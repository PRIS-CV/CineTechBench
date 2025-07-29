import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
import time
import random
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

"""
Video Question Answering Script using Qwen2.5-VL-7B-Intstruct

How to Run:
    python video_qa_qwen2.5_vl.py --json_path /path/to/your/video_annotation.json --video_path /path/to/your/video_folder

Output files:
    - results-Qwen2.5-VL-7B-Instruct.json: Contains detailed results for each video
    - summary-Qwen2.5-VL-7B-Instruct.json: Contains overall accuracy and statistics

Use other open-source models:
    To integrate a different model, you'll need to modify the call_model function and the relevant model loading sections of the code
"""

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="cuda"
# )

# --- Configuration Constants ---

# Model name for display and output file naming
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
# Path to the locally saved model directory
MODEL_PATH = "/PATH_TO_YOUR/Qwen2.5-VL-7B-Instruct"
# If model didn't answer in right format, retry up to this many times
MAX_ANSWER_RETRIES = 3

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

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


def call_model(
    video_path: str,
    question: str,

) -> str:
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": question},
            ],
        }
    ]


    # In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return(output_text[0])

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
            raw = call_model(video_path, prompt)
            pred = extract_answer(raw)
            tries = 0
            while pred == 'N/A' and tries < MAX_ANSWER_RETRIES:
                print(f"🔄 Retrying answer for {vid_filename} ({tries+1}/{MAX_ANSWER_RETRIES})")
                raw = call_model(video_path, prompt)
                pred = extract_answer(raw.text)
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
