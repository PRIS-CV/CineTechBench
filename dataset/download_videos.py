import json
import os
import subprocess

# ==============================================================================
# IMPORTANT NOTICE
# ==============================================================================
#
# This code and its associated tools are intended for academic research purposes only.
# Any use of this code for commercial purposes or non-academic research is strictly prohibited.
# The user assumes all liability for any consequences arising from the use of this code.
#
# ==============================================================================

YOUR_BROWSER_NAME = 'edge'
# --- Configuration Section ---
script_directory = os.path.dirname(os.path.abspath(__file__))
# 1. Path definitions. The script assumes all necessary files (executables, json, cookies)
#    are in the same directory as the script itself.
json_file_path = os.path.join(script_directory, 'video_annotation.json')
yt_dlp_path = os.path.join(script_directory, 'yt-dlp.exe')
ffmpeg_path = os.path.join(script_directory, 'ffmpeg.exe')
cookies_file_path = os.path.join(script_directory, 'cookies.txt') 
# 2. The folder name to save videos into (the script will create it automatically).
save_folder = os.path.join(script_directory, 'Download_videos')

# --- Main Program ---

print("--- Starting bulk video download task ---")

# 检查所有必需文件是否存在
if not os.path.exists(yt_dlp_path):
    print(f"❌ Fatal Error: yt_dlp.exe not found in the script directory! Please download it and place it here.")
    exit()
if not os.path.exists(ffmpeg_path):
    print(f"❌ Fatal Error: ffmpeg.exe not found in the script directory!")
    exit()

print(f"✅ Successfully located yt-dlp: {yt_dlp_path}")
print(f"✅ Successfully located FFmpeg: {ffmpeg_path}")

# Check for and create the video save folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"✅ Folder '{os.path.basename(save_folder)}' has been created.")

# Read and parse the JSON file
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        video_data = json.load(f)
    print(f"✅ Successfully read and parsed '{os.path.basename(json_file_path)}', found {len(video_data)} video entries.")
except FileNotFoundError:
    print(f"❌ Error reading or parsing JSON file: '{os.path.basename(json_file_path)}'。")
    exit()

# Loop through and download videos
for i, item in enumerate(video_data):
    try:
        video_link = item['link']
        video_name = item['video_name']
        video_id = item['id']
    except KeyError as e:
        print(f"⚠️ Skipping item {i+1} , missing key field:: {e}")
        continue

    print("-" * 40)
    print(f"▶️ Processing video {i+1}/{len(video_data)} (ID: {video_id})")
    
    output_path_template = os.path.join(save_folder, video_name)
    
    if os.path.exists(output_path_template):
        print(f"🟢 File '{video_name}' already exists, skipping download.")
        continue

    command = [
        yt_dlp_path,  
        '--cookies', cookies_file_path,
        '--ffmpeg-location', ffmpeg_path,
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', output_path_template,
        video_link
    ]

    print(f"🔗 Link: {video_link}")
    print(f"🏷️ Saving as: {video_name}")
    print("⏳ Calling yt-dlp to download and merge, please wait...")

    try:

        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"✅ Successfully downloaded and merged: {video_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Failed to download or merge video (ID: {video_id}) ")
        print(f"   yt-dlp Error Output:\n   {e.stderr.strip()}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during download: {e}")

print("-" * 40)
print("\n🎉 All tasks completed!")