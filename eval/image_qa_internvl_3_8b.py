import base64
import os
import json
import re
from datetime import datetime
from openai import OpenAI
import time
import random
import argparse # Imported argparse

"""

Image Question Answering Script using InternVL-3-8B.

How to Run:
    python image_qa_internvl_3_8b.py --json_path /path/to/your/image_annotation.json --image_path /path/to/your/image_folder

Output files: Contains the results for each image and summary statistics in each dimension.
    - results-InternVL-3-8B-{dimension1}.json
    - results-InternVL-3-8B-{dimension2}.json
    - results-InternVL-3-8B-{dimension3}.json
    ....

"""

# --- Configuration Area: Paths & API Key ---
# Please modify these paths to match your local setup.

# 1. MODEL_NAME and HOST_URL
# These variables define the model to be used and the API endpoint.
# Specify the model you want to use for the evaluation.
# Specify the HOST_URL for the API endpoint.
# You can change this to other compatible models and model codes if needed.

MODEL_NAME = 'InternVL-3-8B'
HOST_URL= f"http://SERVER_HOST:SERVER_PORT/v1"

# 2. BASE_DATASET_PATH
# This path will now be set via a command-line argument.
# This placeholder is kept for reference but will be overwritten.
BASE_DATASET_PATH = "YOUR_SINGLE_IMAGE_FOLDER_PATH_HERE"

# 3. ANNOTATION_JSON_PATH
# This path will now be set via a command-line argument.
# This placeholder is kept for reference but will be overwritten.
ANNOTATION_JSON_PATH = "PATH_TO_YOUR/image_annotation.json"


# Retry logic parameters
MAX_API_RETRIES = 5

# --- Main Configuration ---

# List of all dimensions to be tested. The script will iterate through this list.
# These names must match the "Category" field in your JSON file.

ALL_DIMENSIONS = ["Scale","Angle","Composition","Lighting","Colors","Focal Lengths"]

# This is the original options map you provided.
# It is used as a fallback by the `extract_answer` function.
# Note: In Python, if a dictionary has duplicate keys, the last value assigned to that key is used.

OPTIONS_MAP = {
    "Scale": {
        "A": "Extreme Close-Up",
        "B": "Close-Up",
        "C": "Medium Close-Up",
        "D": "Medium Shot",
        "E": "Medium Long Shot",
        "F": "Long Shot",
        "G": "Extreme Long Shot"
    },
    "Angle": {
        "A": "High Angle Shot",
        "B": "Low Angle Shot",
        "C": "Bird's Eye View",
        "D": "Worm's Eye View",
        "A": "Diagonal Angle",
        "B": "Profile Shot",
        "C": "Back Shot"
    },
    "Composition":{
        "A":"Symmetrical",
        "B":"Central",
        "C":"Diagonal",
        "D":"Rule of Thirds",
        "E":"Framing",
        "F":"Curved Line",
        "G":"Horizontal"
    },
    "Lighting":{
        "A":"High Key",
        "B":"Low Key",
        "A":"Hard Light",
        "B":"Soft Light",
        "A":"Back Light",
        "B":"Side Light",
        "C":"Top Light"
    },
    "Colors":{
        "A":"Red",
        "B":"Yellow",
        "C":"Blue",
        "D":"Green",
        "E":"Purple",
        "F":"Black and White"
    },
    "Focal Lengths":{
        "A":"Standard Lens",
        "B":"Medium Focal Length",
        "C":"Telephoto Lens",
        "D":"Fisheye Lens",
        "E":"Macro Lengs"
    }
}

#  base 64 encoding format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def load_data(json_path, dimension_name):
    """
    Loads and filters data from the main JSON annotation file.

    Args:
        json_path (str): The path to the consolidated JSON file.
        dimension_name (str): The dimension (e.g., "Scale", "Angle") to filter for.

    Returns:
        list: A list of data items for the specified dimension.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    # Filter the data to return only items matching the requested dimension
    return [item for item in all_data if item.get('Category') == dimension_name]

def compute_accuracy(results):
    """
    Calculates the accuracy of the model's answers.

    Args:
        results (list): A list of result dictionaries.

    Returns:
        tuple: A tuple containing (accuracy, correct_count, total_valid_samples).
    """
    valid_choices = set("ABCDEFG")
    # Filter out results where the model failed or gave an invalid letter
    valid_results = [r for r in results if r['ModelAnswer'].upper() in valid_choices]
    correct = sum(1 for r in valid_results if r['ModelAnswer'].upper() == r['GroundTruthAnswer'].upper())
    total = len(valid_results)
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def extract_answer(raw_output, options_dict, max_retry=2, retry_func=None):
    """
    Extracts the choice (e.g., 'A', 'B') from the model's raw text output.

    Args:
        raw_output (str): The raw text response from the model.
        options_dict (dict): The map of options for the current question dimension.
        max_retry (int): The maximum number of retries if the answer is invalid.
        retry_func (callable): The function to call to get a new model output on retry.

    Returns:
        str: The extracted answer ('A', 'B', etc.) or 'N/A' if not found.
    """
    # First, try to find a single capital letter A-G in the output.
    match = re.search(r'\b[A-Ga-g]\b', raw_output)
    if match:
        return match.group().upper()

    # If no letter is found, fall back to searching for keywords from the options.
    for key, keyword in options_dict.items():
        if keyword.lower() in raw_output.lower():
            return key

    # If the answer is still not found and retries are available, ask again.
    if retry_func and max_retry > 0:
        print("🔁 Invalid response. Retrying question...")
        new_output = retry_func()
        return extract_answer(new_output, options_dict, max_retry - 1, retry_func)

    return "N/A" # Return 'N/A' if no valid answer can be found.


def get_model_response(img_path,question):
    """
    Sends the image and question to the model and gets a response.

    Args:
        img_path (str): The file path of the image to be analyzed.
        question (str): The question prompt for the model.

    Returns:
        str: The text response from the model.
    """

    client = OpenAI(base_url=HOST_URL, api_key="None")
    models = client.models.list()
    model = models.data[0].id  # Use the first available model
    base64_image = encode_image(img_path)
    attempt = 0
    while attempt < MAX_API_RETRIES:
        try:
            completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type":"text","text": "You are a helpful assistant."}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}, 
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            )
            return(completion.choices[0].message.content)
        except Exception as e:
            attempt += 1
            print(f"[Retry {attempt}/{MAX_API_RETRIES}] Error: {e}")
    return "ERROR: Max retries exceeded"

def save_results_to_json(results, path, summary):
    """
    Saves the full list of results and a summary to a JSON file.

    Args:
        results (list): The list of all processed items.
        path (str): The file path to save the JSON to.
        summary (dict): A summary dictionary of the test run.
    """
    # Combine results and summary into a single structure
    final_output = {
        "TestInfo": {
            "EvaluatedDimension": summary.get("EvaluatedDimension"),
            "ModelUsed": MODEL_NAME,
            "TestTimestamp": datetime.now().isoformat()
        },
        "Summary": summary,
        "Results": results
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

def main(dimension):
    """
    The main function to run the evaluation for a single dimension.
    """
    # Define the path where results for the current dimension will be saved.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(script_dir, f"results-{MODEL_NAME}-{dimension}.json")

    # Load all items for the current dimension
    # It now uses the global ANNOTATION_JSON_PATH which is set from the command line
    items = load_data(ANNOTATION_JSON_PATH, dimension)
    if not items:
        print(f"⚠️ No items found for dimension '{dimension}' in {ANNOTATION_JSON_PATH}. Skipping.")
        return

    total_items = len(items)
    processed_results = []

    # Check if a results file already exists to resume from where we left off.
    if os.path.exists(result_path):
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                processed_results = existing_data.get("Results", [])
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing results file {result_path}. Starting fresh.")
            processed_results = []
            
    processed_filenames = {item['ImageName'] for item in processed_results}
    print(f"🔎 Starting test for dimension: {dimension} | Total: {total_items} | Already processed: {len(processed_filenames)}")

    # Main loop to process each image
    for idx, item in enumerate(items):
        if item['Image_name'] in processed_filenames:
            continue
        
        # It now uses the global BASE_DATASET_PATH which is set from the command line
        image_path = os.path.join(BASE_DATASET_PATH, item['Image_name'])
        question = item['QA']['Prompt']
        ground_truth_answer = item['QA']['Answer']

        # Ensure the image file exists before proceeding
        if not os.path.exists(image_path):
            print(f"❌ Image not found, skipping: {image_path}")
            continue
        
        try:
            # Define a retry function to be called by extract_answer if needed
            def ask_again():
                return get_model_response(image_path, question)

            raw_output = get_model_response(image_path, question)
            model_answer = extract_answer(raw_output, OPTIONS_MAP.get(dimension, {}), retry_func=ask_again)

        except Exception as e:
            print(f"❌ An error occurred on image {idx+1} ({item['Image_name']}): {e}")
            model_answer = "ERROR"
            raw_output = str(e)

        result = {
            "ImageName": item['Image_name'],
            "Category": item['Category'],
            "Annotation": item['Annotation'],
            "GroundTruthAnswer": ground_truth_answer,
            "ModelAnswer": model_answer,
            "RawOutput": raw_output,
            "Timestamp": datetime.now().isoformat()
        }

        processed_results.append(result)
        
        # Save progress incrementally after each item
        # This prevents data loss if the script is interrupted
        temp_summary = {"Note": "This is an intermediate save. Final summary will be at the end."}
        save_results_to_json(processed_results, result_path, temp_summary)

        is_correct = model_answer == ground_truth_answer
        correctness_symbol = "✔️" if is_correct else "❌"
        print(f"{correctness_symbol} [{len(processed_results)}/{total_items}] {item['Image_name']} | GT: {ground_truth_answer} | Pred: {model_answer}")
        
        # Add a random delay to avoid hitting API rate limits
        time.sleep(random.uniform(1.5, 2.5))

    # Final calculation and summary
    accuracy, correct, total_valid = compute_accuracy(processed_results)
    accuracy_str = f"{accuracy:.2%}"
    
    summary = {
        "EvaluatedDimension": dimension,
        "TotalSamples": len(processed_results),
        "ValidSamplesForAccuracy": total_valid,
        "Correct": correct,
        "Accuracy": accuracy,
        "AccuracyFormatted": accuracy_str,
        "FinishedAt": datetime.now().isoformat()
    }

    # Save the final results with the complete summary
    save_results_to_json(processed_results, result_path, summary)
    
    print(f"\n📊 {dimension} Accuracy: {accuracy_str} ({correct}/{total_valid})")
    print(f"✅ Evaluation for '{dimension}' complete. Results saved to {result_path}")


if __name__ == "__main__":
    # --- Argument Parsing ---
    # Setup the parser to accept command-line arguments for file paths.
    parser = argparse.ArgumentParser(
        description="Run image QA evaluation with Qwen2.5-VL-7B.",
        formatter_class=argparse.RawTextHelpFormatter  # Ensures help text formatting is clean
    )
    parser.add_argument(
        '--json_path',
        required=True,
        type=str,
        help="Full path to your consolidated JSON annotation file.\n(e.g., /path/to/your/image_annotation.json)"
    )
    parser.add_argument(
        '--image_path',
        required=True,
        type=str,
        help="Path to the SINGLE FOLDER that contains ALL of your image files.\n(e.g., /path/to/your/images/)"
    )
    args = parser.parse_args()

    # --- Overwrite Global Path Variables ---
    # The global path variables are updated with the values from the command line.
    ANNOTATION_JSON_PATH = args.json_path
    BASE_DATASET_PATH = args.image_path

    # Loop through each dimension and run the main evaluation function
    for dim in ALL_DIMENSIONS:
        main(dim)
        print("-" * 50)
    
    print("\n🎉 All dimensions have been processed.")