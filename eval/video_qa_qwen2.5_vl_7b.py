from openai import OpenAI
import cv2
import os
import json
import math
import base64
import re
import time
import random
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import base64

"""

Video Question Answering Script using Qwen2.5-VL-7B. 

Since Qwen2.5-VL-7B dont support video input, this script extracts frames from the video and sends them to the model along with the question.

If the open-source model supports video input, you can refer to video_qa_internvl_3_8b.py

How to Run:
    python video_qa_qwen2.5_vl_7b.py --json_path /path/to/your/video_annotation.json --video_path /path/to/your/video_folder

Output files:
    - Frames: Extracted frames from videos will be saved in a folder named "Frames" in the script directory.
    - results-Qwen2.5-VL-7B.json: Contains detailed results for each video
    - summary-Qwen2.5-VL-7B.json: Contains overall accuracy and statistics
    
"""

# Model name, used for naming the output files
MODEL_NAME = "Qwen2.5-VL-7B"
# Host url, used for API calls
HOST_URL= f"http://SERVER_HOST:SERVER_PORT/v1"

# Retry logic parameters
MAX_API_RETRIES = 5
API_WAIT_SECONDS = 5
MAX_ANSWER_RETRIES = 2

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
FRAME_OUTPUT_FOLDER_PATH = os.path.join(script_directory, "Frames")

# === HELPER FUNCTIONS ===

def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Video QA evaluation with model.",
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

def encode_frame(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"❌ Error encoding frame: {e}")
        return None

def load_video_data(json_path: str) -> List[Dict[str, Any]]:
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
    # Regex to find lines starting with A., B., C., or D. and capture the text.
    matches = re.findall(r'([A-D])\.\s*(.*)', prompt)
    for letter, text in matches:
        mapping[letter.upper()] = text.strip()
    return mapping

def extract_and_save_frames(video_path, output_base):
    frames_b64, paths = [], []
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_base, name)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return [], [], 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 150
    duration = total / fps
    num_to_sample = max(math.ceil(duration * 1), 4)
    if total < num_to_sample:
        indices = list(range(total))
    else:
        step = (total - 1) / (num_to_sample - 1)
        indices = sorted({int(round(i * step)) for i in range(num_to_sample)})
    idx, count = 0, 0
    while cap.isOpened() and count < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if idx == indices[count]:
            b64 = encode_frame(frame)
            if b64:
                frames_b64.append(b64)
            p = os.path.join(out_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
            count += 1
        idx += 1
    cap.release()
    return frames_b64, len(paths)


def call_model(
    frames: List[str],
    question: str,
    max_retries: int = MAX_API_RETRIES,
    wait_seconds: int = API_WAIT_SECONDS
) -> str:
    """
    Sends a video and a question to the model and gets a response.
    Includes retry logic for handling transient API errors.

    Args:
        frames: The frames captured from the video.
        question: The prompt/question for the model.
        client: The initialized model client.
        max_retries: The maximum number of times to retry the API call.
        wait_seconds: The number of seconds to wait between retries.

    Returns:
        The text response from the model, or an error message if all retries fail.
    """
    client = OpenAI(base_url=HOST_URL, api_key="None")

    if not frames:
        return "ERROR: No frames"
    
    messages = [{"role":"user","content":[{"type":"text","text":question}]}]

    messages[0]['content'].extend(
    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}} for b64 in frames
    )

    models = client.models.list()
    model = models.data[0].id
    attempt = 0
    while attempt < max_retries:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=150
            )
            return resp.choices[0].message.content or ""

        except Exception as e:
            attempt += 1
            print(f"[Retry {attempt}/{max_retries}] Error: {e}")
            time.sleep(wait_seconds)
    return "ERROR: Max retries exceeded"

def extract_answer(raw_text: str) -> str:
    """
    Extracts a single letter answer (A, B, C, or D) from the model's raw output.

    Args:
        raw_text: The raw string response from the model.

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
    print("Idx | Video | Frames | GT | Pred | Status")
    print("-" * 40)

    for idx, item in enumerate(data, 1):
        vid_filename = item['VideoFilename']
        video_path = os.path.join(VIDEO_FOLDER_PATH, vid_filename)
        prompt = item['Prompt']
        gt = item['GroundTruth']
        cls = item['Classification']
        answer_mapping = item['AnswerMapping'] # Get the stored answer mapping
        frames, n = extract_and_save_frames(video_path, FRAME_OUTPUT_FOLDER_PATH)
        if not os.path.isfile(video_path):
            raw = ""
            pred = "ERROR"
            status = "File Not Found"
        else:
            raw = call_model(frames, prompt)
            pred = extract_answer(raw)
            tries = 0
            while pred == 'N/A' and tries < MAX_ANSWER_RETRIES:
                print(f"🔄 Retrying answer for {vid_filename} ({tries+1}/{MAX_ANSWER_RETRIES})")
                raw = call_model(frames, prompt)
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
        print(f"{idx}/{total} | {vid_filename} | {n} | {gt} | {pred} | {status}")

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

