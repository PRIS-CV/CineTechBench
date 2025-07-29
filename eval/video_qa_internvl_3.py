import numpy as np
import torch
import os
import json
import re
import time
import math
import random
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from datetime import datetime
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Any, Optional
import argparse

"""
Video Question Answering Script using InternVL-3-8B

How to Run:
    python video_qa_internvl_3.py --json_path /path/to/your/video_annotation.json --video_path /path/to/your/video_folder

Output files:
    - results-InternVL-3-8B.json: Contains detailed results for each video
    - summary-InternVL-3-8B.json: Contains overall accuracy and statistics

Use other open-source models:
    To integrate a different model, you'll need to modify the call_model function and the relevant model loading sections of the code
"""

# --- Configuration Constants ---

# Model name for display and output file naming
MODEL_NAME = "InternVL-3-8B"
# Path to the locally saved model directory
MODEL_PATH = "/mnt/sdc/model_zoo/InternVL3-8B"
# If model didn't answer in right format, retry up to this many times
MAX_ANSWER_RETRIES = 3

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Video QA evaluation with Qwen2.5-VL.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        '-j', '--json_path',
        type=str,
        required=True,
        help="Path to the input JSON file containing video annotations and QA data."
    )
    parser.add_argument(
        '-v', '--video_path',
        type=str,
        required=True,
        help="Path to the folder containing the video files."
    )
    return parser

def load_video_data(json_path:str) -> List[Dict[str, Any]]:

    """
    Loads video QA data from a JSON file formatted as a list of dictionaries.

    Args:
        json_path: The path to the input JSON file.

    Returns:
        A list of dictionaries, where each dictionary contains the
        processed information for one video item.
    """

    print(f"🔄 Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: JSON file not found at {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not decode JSON from {json_path}. Please check its format.")
        return []

    items = []
    # The new JSON is a list of objects, so we iterate through it directly.
    for item in data:
        qa = item.get('QA', {})
        if 'prompt' in qa and 'answer' in qa:
            items.append({
                # Mapping new keys from the JSON to the keys expected by the script
                'VideoFilename': item.get('video_name', 'N/A'),
                'ID': item.get('id', 'N/A'),
                'movie_id': item.get('movie_id', 'N/A'),
                'Original_Type': item.get('type', 'N/A'),
                'Classification': item.get('classification', 'N/A'),
                'Prompt': qa['prompt'],
                'GroundTruth': qa['answer'],
                'AnswerMapping': extract_answer_mapping(qa['prompt'])
            })
    print(f"✅ Loaded {len(items)} items successfully.")
    return items

def extract_answer_mapping(prompt: str) -> Dict[str, str]:
    """
    Extracts the mapping from answer letter (A, B, C, D) to the shot
    type string from the prompt text.

    Args:
        prompt: The full text of the prompt sent to the model.

    Returns:
        A dictionary mapping the option letter to its description.
        Example: {'A': 'Pan Left Shot', 'B': 'Crane Shot'}
    """
    mapping = {}
    # Regex to find lines starting with A., B., C., D. and capture the text following it
    # Adjust regex if the format is different (e.g., includes parentheses)
    matches = re.findall(r'([A-D])\.\s*(.*)', prompt)
    for letter, text in matches:
        # Clean up the extracted text, removing potential trailing periods or spaces
        mapping[letter.upper()] = text.strip()
    return mapping

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.


# set the max number of tiles in `max_num`
generation_config = dict(max_new_tokens=1024, do_sample=True)


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    print(f"Video length: {max_frame} frames")
    fps = float(vr.get_avg_fps())
    print(f"FPS: {fps}")

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def call_model(
    video_path: str,
    question: str,
    model: AutoModel,
    tokenizer: AutoTokenizer
) -> str:

    pixel_values, num_patches_list = load_video(video_path, num_segments=1, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    return response

def extract_answer(raw_text: str) -> str:
    """
    Extracts a single letter answer (A, B, C, or D) from the model's raw output.

    Args:
        raw_text: The raw string response from the Gemini model.

    Returns:
        The uppercase letter if found, otherwise 'N/A'.
    """
    if not isinstance(raw_text, str):
        return "N/A"
    # This regex matches an optional opening parenthesis, a single letter (a-d),
    # an optional closing parenthesis, and an optional period.
    match = re.match(r'^\s*\(?([A-Da-d])\)?\.?\s*$', raw_text.strip())
    return match.group(1).upper() if match else "N/A"

# === MAIN WORKFLOW ===

def main():

    """Main function to execute the video QA evaluation workflow."""
    # 1. Setup and Initialization
    parser = setup_arg_parser()
    args = parser.parse_args()

    VIDEO_FOLDER_PATH = args.video_path
    JSON_DATA_PATH = args.json_path

    # Prepare output paths
    script_directory = os.path.dirname(os.path.abspath(__file__))
    RESULT_JSON_PATH = os.path.join(script_directory, f"results-{MODEL_NAME}.json")
    SUMMARY_JSON_PATH = os.path.join(script_directory, f"summary-{MODEL_NAME}.json")
    print(f"▶️ Script starting. Results will be saved to: {RESULT_JSON_PATH}")

 # <<< MODIFIED: All model loading is now inside main() to prevent global scope errors >>>
    print("🧠 Loading InternVL model and tokenizer... (This may take a while)")
    try:

        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto").eval()
            
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
        print("✅ Model and tokenizer loaded successfully.")

    except Exception as e:
        print(f"❌ Failed to load model or tokenizer. Error: {e}")
        return
    


    # 2. Load Data
    data = load_video_data(JSON_DATA_PATH)
    if not data:
        print("❌ No data to process. Exiting.")
        return

    # 3. Process Each Video
    results = []
    total = len(data)
    print("\n--- Starting Video Processing ---")
    print("Idx | Video | GT | Pred | Status")
    print("-" * 40)

    for idx, item in enumerate(data, 1):
        vid_filename = item['VideoFilename']
        video_path = os.path.join(VIDEO_FOLDER_PATH, vid_filename)
        prompt = item['Prompt']
        gt = item['GroundTruth']
        cls = item['Classification']
        answer_mapping = item['AnswerMapping'] # Get the stored answer mapping

        if not os.path.isfile(video_path):
            raw = ""
            pred = "ERROR"
            status = "File Not Found"
        else:
            raw = call_model(video_path, prompt,model, tokenizer)
            pred = extract_answer(raw)
            tries = 0
            while pred == 'N/A' and tries < MAX_ANSWER_RETRIES:
                print(f"🔄 Retrying answer for {vid_filename} ({tries+1}/{MAX_ANSWER_RETRIES})")
                raw = call_model(video_path, prompt,model, tokenizer)
                pred = extract_answer(raw)
                tries += 1
            status = 'Completed' if pred not in ['N/A','ERROR'] else 'Error'
        time.sleep(random.uniform(0.5, 1.5)) # Random sleep to avoid rate limiting
        # Save current result
        results.append({
            'VideoFilename': vid_filename,
            'ID': item.get('ID'),
            'movie_id': item.get('movie_id'),
            'Original_Type': item['Original_Type'],
            'Classification': cls,
            'Prompt': prompt,
            'GroundTruth': gt,
            'AnswerMapping': answer_mapping, # Store the mapping in results
            'Pred': pred,
            'Raw': raw,
            'Status': status
        })
        with open(RESULT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({'Results': results}, f, indent=4)
        # Print progress
        print(f"{idx}/{total} | {vid_filename} | {gt} | {pred} | {status}")

        # Note: Real-time JSON writing and incremental accuracy calculations are removed from the loop
        # They will be calculated and saved after the loop finishes.

    print("\nTesting complete. Calculating accuracies...")

    # --- Calculate Accuracies ---

    correct_overall = sum(1 for r in results if r.get('Status') == 'Completed' and r['Pred'] == r['GroundTruth'])
    valid_overall = sum(1 for r in results if r.get('Status') == 'Completed')
    acc_total = (correct_overall / valid_overall) if valid_overall > 0 else 0.0

    single_results = [r for r in results if r.get('Classification') == 'Single' and r.get('Status') == 'Completed']
    single_valid = len(single_results)
    single_correct = sum(1 for r in single_results if r['Pred'] == r['GroundTruth'])
    acc_single = (single_correct / single_valid) if single_valid > 0 else 0.0

    comb_results = [r for r in results if r.get('Classification') == 'Combinational' and r.get('Status') == 'Completed']
    comb_valid = len(comb_results)
    comb_correct = sum(1 for r in comb_results if r['Pred'] == r['GroundTruth'])
    acc_comb = (comb_correct / comb_valid) if comb_valid > 0 else 0.0

    # Define the sets of types for categories and groups (based on Original_Type)
    fixed_types = {'Fixed Shot'}
    dolly_types = {'Dolly In Shot', 'Dolly Out Shot'}
    tilt_types = {'Tilt Up Shot', 'Tilt Down Shot'}
    pan_types = {'Pan Left Shot', 'Pan Right Shot'}
    trucking_specific_types = {'Trucking Left Shot', 'Trucking Right Shot'} # For "Trucking Shot" category
    crane_types = {'Crane Shot'}
    rolling_types = {'Rolling Clockwise Shot', 'Rolling Counterclockwise Shot'}
    zoom_types = {'Zoom In Shot', 'Zoom Out Shot'}
    translation_group_types = dolly_types | trucking_specific_types | crane_types | {'Tracking Shot'} # For "Translation" group
    rotation_group_types = tilt_types | pan_types | rolling_types # For "Rotation" group

    # Initialize counts for new categories
    category_counts = {
        'Fixed': {'total': 0, 'correct': 0},
        'Dolly': {'total': 0, 'correct': 0},
        'Tilt': {'total': 0, 'correct': 0},
        'Pan': {'total': 0, 'correct': 0},
        'Trucking': {'total': 0, 'correct': 0}, # This is for the Trucking (Left+Right) category
        'Crane': {'total': 0, 'correct': 0},
        'Rolling': {'total': 0, 'correct': 0},
        'Zoom': {'total': 0, 'correct': 0},
        'Translation': {'total': 0, 'correct': 0}, # This is for the Translation group
        'Rotation': {'total': 0, 'correct': 0} # This is for the Rotation group
    }

    # Populate counts for new categories from completed single results
    for result in single_results: # Iterate only through completed single shots
        original_type = result['Original_Type']
        is_correct = (result['Pred'] == result['GroundTruth'])

        if original_type in fixed_types:
            category_counts['Fixed']['total'] += 1
            if is_correct:
                category_counts['Fixed']['correct'] += 1
        if original_type in dolly_types:
            category_counts['Dolly']['total'] += 1
            if is_correct:
                category_counts['Dolly']['correct'] += 1
        if original_type in tilt_types:
            category_counts['Tilt']['total'] += 1
            if is_correct:
                category_counts['Tilt']['correct'] += 1
        if original_type in pan_types:
            category_counts['Pan']['total'] += 1
            if is_correct:
                category_counts['Pan']['correct'] += 1
        if original_type in trucking_specific_types: # Check against specific Trucking types
            category_counts['Trucking']['total'] += 1
            if is_correct:
                category_counts['Trucking']['correct'] += 1
        if original_type in crane_types:
            category_counts['Crane']['total'] += 1
            if is_correct:
                category_counts['Crane']['correct'] += 1
        if original_type in rolling_types:
            category_counts['Rolling']['total'] += 1
            if is_correct:
                category_counts['Rolling']['correct'] += 1
        if original_type in zoom_types:
            category_counts['Zoom']['total'] += 1
            if is_correct:
                category_counts['Zoom']['correct'] += 1
        if original_type in translation_group_types: # Check against Translation group types
            category_counts['Translation']['total'] += 1
            if is_correct:
                category_counts['Translation']['correct'] += 1
        if original_type in rotation_group_types: # Check against Rotation group types
            category_counts['Rotation']['total'] += 1
            if is_correct:
                category_counts['Rotation']['correct'] += 1

    # Calculate accuracies for new categories
    accuracy_fixed = (category_counts['Fixed']['correct'] / category_counts['Fixed']['total']) if category_counts['Fixed']['total'] > 0 else 0.0
    accuracy_dolly = (category_counts['Dolly']['correct'] / category_counts['Dolly']['total']) if category_counts['Dolly']['total'] > 0 else 0.0
    accuracy_tilt = (category_counts['Tilt']['correct'] / category_counts['Tilt']['total']) if category_counts['Tilt']['total'] > 0 else 0.0
    accuracy_pan = (category_counts['Pan']['correct'] / category_counts['Pan']['total']) if category_counts['Pan']['total'] > 0 else 0.0
    accuracy_trucking = (category_counts['Trucking']['correct'] / category_counts['Trucking']['total']) if category_counts['Trucking']['total'] > 0 else 0.0
    accuracy_crane = (category_counts['Crane']['correct'] / category_counts['Crane']['total']) if category_counts['Crane']['total'] > 0 else 0.0
    accuracy_rolling = (category_counts['Rolling']['correct'] / category_counts['Rolling']['total']) if category_counts['Rolling']['total'] > 0 else 0.0
    accuracy_zoom = (category_counts['Zoom']['correct'] / category_counts['Zoom']['total']) if category_counts['Zoom']['total'] > 0 else 0.0
    accuracy_translation = (category_counts['Translation']['correct'] / category_counts['Translation']['total']) if category_counts['Translation']['total'] > 0 else 0.0
    accuracy_rotation = (category_counts['Rotation']['correct'] / category_counts['Rotation']['total']) if category_counts['Rotation']['total'] > 0 else 0.0


    # --- Update Summary and Print Final Results ---

    summary = {
        'Timestamp': datetime.now().isoformat(),
        'TotalVideosInData': total,
        'ProcessedCompletedVideos': valid_overall,
        'OverallCorrect': correct_overall,
        'AccuracyOverall': f"{acc_total:.2%}",

        'FixedValid_SingleOnly': category_counts['Fixed']['total'],
        'FixedCorrect_SingleOnly': category_counts['Fixed']['correct'],
        'AccuracyFixed_SingleOnly': f"{accuracy_fixed:.2%}",

        'TranslationValid_SingleOnly': category_counts['Translation']['total'],
        'TranslationCorrect_SingleOnly': category_counts['Translation']['correct'],
        'AccuracyTranslation_SingleOnly': f"{accuracy_translation:.2%}",

        'RotationValid_SingleOnly': category_counts['Rotation']['total'],
        'RotationCorrect_SingleOnly': category_counts['Rotation']['correct'],
        'AccuracyRotation_SingleOnly': f"{accuracy_rotation:.2%}",

        'ZoomValid_SingleOnly': category_counts['Zoom']['total'],
        'ZoomCorrect_SingleOnly': category_counts['Zoom']['correct'],
        'AccuracyZoom_SingleOnly': f"{accuracy_zoom:.2%}",

        'SingleValid': single_valid,
        'SingleCorrect': single_correct,
        'AccuracySingle': f"{acc_single:.2%}",

        'CombinationalValid': comb_valid,
        'CombinationalCorrect': comb_correct,
        'AccuracyCombinational': f"{acc_comb:.2%}",

        'DollyValid_SingleOnly': category_counts['Dolly']['total'],
        'DollyCorrect_SingleOnly': category_counts['Dolly']['correct'],
        'AccuracyDolly_SingleOnly': f"{accuracy_dolly:.2%}",

        'TiltValid_SingleOnly': category_counts['Tilt']['total'],
        'TiltCorrect_SingleOnly': category_counts['Tilt']['correct'],
        'AccuracyTilt_SingleOnly': f"{accuracy_tilt:.2%}",

        'PanValid_SingleOnly': category_counts['Pan']['total'],
        'PanCorrect_SingleOnly': category_counts['Pan']['correct'],
        'AccuracyPan_SingleOnly': f"{accuracy_pan:.2%}",

        'TruckingValid_SingleOnly': category_counts['Trucking']['total'],
        'TruckingCorrect_SingleOnly': category_counts['Trucking']['correct'],
        'AccuracyTrucking_SingleOnly': f"{accuracy_trucking:.2%}",

        'CraneValid_SingleOnly': category_counts['Crane']['total'],
        'CraneCorrect_SingleOnly': category_counts['Crane']['correct'],
        'AccuracyCrane_SingleOnly': f"{accuracy_crane:.2%}",

        'RollingValid_SingleOnly': category_counts['Rolling']['total'],
        'RollingCorrect_SingleOnly': category_counts['Rolling']['correct'],
        'AccuracyRolling_SingleOnly': f"{accuracy_rolling:.2%}"

    }

    # Print final summary
    print("\n--- Overall Accuracy ---")
    print(f"Total Videos in Data: {summary['TotalVideosInData']}, Processed (Completed Status): {summary['ProcessedCompletedVideos']}")
    print(f"Overall Correct Predictions (Completed Status): {summary['OverallCorrect']}")
    print(f"Overall Accuracy (Completed Status): {summary['AccuracyOverall']}")

    print("\n--- Classification Accuracy (Completed items only) ---")
    print(f"Single Shots Valid: {summary['SingleValid']}, Correct: {summary['SingleCorrect']}, Accuracy: {summary['AccuracySingle']}")
    print(f"Combinational Shots Valid: {summary['CombinationalValid']}, Correct: {summary['CombinationalCorrect']}, Accuracy: {summary['AccuracyCombinational']}")

    print("\n--- Single Shot Type Accuracy (Completed Single shots only) ---")
    print(f"Fixed Shot Valid: {summary['FixedValid_SingleOnly']}, Correct: {summary['FixedCorrect_SingleOnly']}, Accuracy: {summary['AccuracyFixed_SingleOnly']}")
    print(f"Dolly Shot Valid: {summary['DollyValid_SingleOnly']}, Correct: {summary['DollyCorrect_SingleOnly']}, Accuracy: {summary['AccuracyDolly_SingleOnly']}")
    print(f"Tilt Shot Valid: {summary['TiltValid_SingleOnly']}, Correct: {summary['TiltCorrect_SingleOnly']}, Accuracy: {summary['AccuracyTilt_SingleOnly']}")
    print(f"Pan Shot Valid: {summary['PanValid_SingleOnly']}, Correct: {summary['PanCorrect_SingleOnly']}, Accuracy: {summary['AccuracyPan_SingleOnly']}")
    print(f"Trucking Shot Valid: {summary['TruckingValid_SingleOnly']}, Correct: {summary['TruckingCorrect_SingleOnly']}, Accuracy: {summary['AccuracyTrucking_SingleOnly']}")
    print(f"Crane Shot Valid: {summary['CraneValid_SingleOnly']}, Correct: {summary['CraneCorrect_SingleOnly']}, Accuracy: {summary['AccuracyCrane_SingleOnly']}")
    print(f"Rolling Shot Valid: {summary['RollingValid_SingleOnly']}, Correct: {summary['RollingCorrect_SingleOnly']}, Accuracy: {summary['AccuracyRolling_SingleOnly']}")
    print(f"Zoom Shot Valid: {summary['ZoomValid_SingleOnly']}, Correct: {summary['ZoomCorrect_SingleOnly']}, Accuracy: {summary['AccuracyZoom_SingleOnly']}")

    print("\n--- Single Shot Group Accuracy (Completed Single shots only) ---")
    print(f"Translation Group Valid: {summary['TranslationValid_SingleOnly']}, Correct: {summary['TranslationCorrect_SingleOnly']}, Accuracy: {summary['AccuracyTranslation_SingleOnly']}")
    print(f"Rotation Group Valid: {summary['RotationValid_SingleOnly']}, Correct: {summary['RotationCorrect_SingleOnly']}, Accuracy: {summary['AccuracyRotation_SingleOnly']}")


    # Write final results to JSON
    with open(SUMMARY_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump({'Summary': summary}, f, indent=4)

if __name__ == '__main__':
    main()
