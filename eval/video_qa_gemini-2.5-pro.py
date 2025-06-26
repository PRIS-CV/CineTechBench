import os
import json
import re
import time
import random
from datetime import datetime
# 请根据实际使用的 Gemini SDK 调整导入
from google import genai
from google.genai import types
import base64
# === 配置区 ===

JSON_DATA_PATH = r""
VIDEO_FOLDER_PATH = r""
client = genai.Client(api_key="xxxxxxxxxxxxx")
# Gemini 模型参数
MODEL_NAME = "Gemini-2.5-Pro"

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
print(f"Script Path: {script_directory}")

RESULT_JSON_PATH = os.path.join(script_directory, f"results-{MODEL_NAME}-added.json")
SUMMARY_JSON_PATH = os.path.join(script_directory, f"summary-{MODEL_NAME}-added.json")
print (f"Result JSON Path: {RESULT_JSON_PATH}")

MAX_RETRIES = 10 # Placeholder
WAIT_SECONDS = 5 # Placeholder
ANSWER_RETRIES = 2 # Placeholder

# === HELPER FUNCTIONS ===

def load_video_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = []
    for fname, details in data.items():
        qa = details.get('QA', {})
        if 'prompt' in qa and 'answer' in qa:
            items.append({
                'VideoFilename': fname,
                'Original_Type': details.get('Original_Type', 'N/A'),
                'Classification': details.get('Classification', 'N/A'),
                'Prompt': qa['prompt'],
                'GroundTruth': qa['answer'],
                'AnswerMapping': extract_answer_mapping(qa['prompt']) # Extract answer mapping
            })
    return items

def extract_answer_mapping(prompt):
    """Extracts the mapping from answer letter (A, B, C, D) to shot type string from the prompt."""
    mapping = {}
    # Regex to find lines starting with A., B., C., D. and capture the text following it
    # Adjust regex if the format is different (e.g., includes parentheses)
    matches = re.findall(r'([A-D])\.\s*(.*)', prompt)
    for letter, text in matches:
        # Clean up the extracted text, removing potential trailing periods or spaces
        mapping[letter.upper()] = text.strip()
    return mapping


# def GEMINI(video_path, question, client, max_retries=MAX_RETRIES, wait_seconds=WAIT_SECONDS):
#     # Keep your original GEMINI function here.
#     pass # Placeholder

def Model(video_path, question, client, max_retries=MAX_RETRIES, wait_seconds=WAIT_SECONDS):
    video_bytes = open(video_path, 'rb').read()
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.models.generate_content(
                model='models/gemini-2.5-pro-preview-03-25',
                contents=types.Content(
                    parts=[
                        types.Part(text=question),
                        types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'))
                    ]
                )
            )
            print(response.text)
            return response.text
        except Exception as e:
            attempt += 1
            print(f"[Retry {attempt}/{max_retries}] Error: {e}")
            time.sleep(wait_seconds)
    return "ERROR: Max retries exceeded"

def extract_answer(raw):
    m = re.match(r'^\(?([A-Ga-g])\)?\.?$', raw.strip())
    return m.group(1).upper() if m else "N/A"

# === 主流程 ===

def main():
    # Ensure client is initialized if not globally
    # client = genai.Client(api_key="YOUR_API_KEY") # Uncomment and replace with your API Key if not global

    data = load_video_data(JSON_DATA_PATH)
    results = []
    total = len(data)
    print("Idx | Video | GT | Pred | Status") # Simplified print during processing

    for idx, item in enumerate(data, 1):
        vid = item['VideoFilename']
        path = os.path.join(VIDEO_FOLDER_PATH, vid)
        prompt = item['Prompt']
        gt = item['GroundTruth']
        cls = item['Classification']
        answer_mapping = item['AnswerMapping'] # Get the stored answer mapping

        if not os.path.isfile(path):
            raw = ""
            pred = "ERROR"
            status = "File Not Found"
        else:
            raw = Model(path, prompt, client)
            pred = extract_answer(raw)
            tries = 0
            while pred == 'N/A' and tries < ANSWER_RETRIES:
                print(f"🔄 Retrying answer for {vid} ({tries+1}/{ANSWER_RETRIES})")
                raw = Model(path, prompt, client)
                pred = extract_answer(raw.text)
                tries += 1
            status = 'Completed' if pred not in ['N/A','ERROR'] else 'Error'
        time.sleep(random.uniform(0.5, 1.5)) # Random sleep to avoid rate limiting
        # Save current result
        results.append({
            'VideoFilename': vid,
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
        print(f"{idx}/{total} | {vid} | {gt} | {pred} | {status}")

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
        'AccuracyOverall': f"{acc_total:.2%}\n",

        'FixedValid_SingleOnly': category_counts['Fixed']['total'],
        'FixedCorrect_SingleOnly': category_counts['Fixed']['correct'],
        'AccuracyFixed_SingleOnly': f"{accuracy_fixed:.2%}\n",

        'TranslationValid_SingleOnly': category_counts['Translation']['total'],
        'TranslationCorrect_SingleOnly': category_counts['Translation']['correct'],
        'AccuracyTranslation_SingleOnly': f"{accuracy_translation:.2%}\n",

        'RotationValid_SingleOnly': category_counts['Rotation']['total'],
        'RotationCorrect_SingleOnly': category_counts['Rotation']['correct'],
        'AccuracyRotation_SingleOnly': f"{accuracy_rotation:.2%}\n",

        'ZoomValid_SingleOnly': category_counts['Zoom']['total'],
        'ZoomCorrect_SingleOnly': category_counts['Zoom']['correct'],
        'AccuracyZoom_SingleOnly': f"{accuracy_zoom:.2%}\n",

        'SingleValid': single_valid,
        'SingleCorrect': single_correct,
        'AccuracySingle': f"{acc_single:.2%}\n",

        'CombinationalValid': comb_valid,
        'CombinationalCorrect': comb_correct,
        'AccuracyCombinational': f"{acc_comb:.2%}\n",

        'DollyValid_SingleOnly': category_counts['Dolly']['total'],
        'DollyCorrect_SingleOnly': category_counts['Dolly']['correct'],
        'AccuracyDolly_SingleOnly': f"{accuracy_dolly:.2%}\n",

        'TiltValid_SingleOnly': category_counts['Tilt']['total'],
        'TiltCorrect_SingleOnly': category_counts['Tilt']['correct'],
        'AccuracyTilt_SingleOnly': f"{accuracy_tilt:.2%}\n",

        'PanValid_SingleOnly': category_counts['Pan']['total'],
        'PanCorrect_SingleOnly': category_counts['Pan']['correct'],
        'AccuracyPan_SingleOnly': f"{accuracy_pan:.2%}\n",

        'TruckingValid_SingleOnly': category_counts['Trucking']['total'],
        'TruckingCorrect_SingleOnly': category_counts['Trucking']['correct'],
        'AccuracyTrucking_SingleOnly': f"{accuracy_trucking:.2%}\n",

        'CraneValid_SingleOnly': category_counts['Crane']['total'],
        'CraneCorrect_SingleOnly': category_counts['Crane']['correct'],
        'AccuracyCrane_SingleOnly': f"{accuracy_crane:.2%}\n",

        'RollingValid_SingleOnly': category_counts['Rolling']['total'],
        'RollingCorrect_SingleOnly': category_counts['Rolling']['correct'],
        'AccuracyRolling_SingleOnly': f"{accuracy_rolling:.2%}\n"

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

