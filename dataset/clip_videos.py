import json
import os
import subprocess
import re

# --- Configuration Area ---
DOWNLOADED_VIDEOS_FOLDER = './raw'
# Changed folder name to reflect high-quality re-encoding
CLIPPED_VIDEOS_FOLDER = './clips'

# --- Script Setup ---
script_directory = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(script_directory, 'ffmpeg.exe')
ffprobe_path = os.path.join(script_directory, 'ffprobe.exe')
json_file_name = './annotation/video_annotation.json'
json_file_path = os.path.join(script_directory, json_file_name)
download_folder_abs_path = os.path.join(script_directory, DOWNLOADED_VIDEOS_FOLDER)
clipped_folder_abs_path = os.path.join(script_directory, CLIPPED_VIDEOS_FOLDER)

# --- Helper Functions ---
def get_frame_rate(video_file_path):
    if not os.path.exists(ffprobe_path):
        print(f"❌ CRITICAL ERROR: ffprobe.exe not found at {ffprobe_path}")
        return None
    command = [
        ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        video_file_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        frame_rate_str = result.stdout.strip()
        if '/' in frame_rate_str:
            num, den = map(float, frame_rate_str.split('/'))
            return num / den if den != 0 else None
        return float(frame_rate_str)
    except Exception as e:
        print(f"🟡 WARNING: Error getting frame rate for {os.path.basename(video_file_path)}: {e}")
        return None

def timecode_to_seconds(timecode_str, frame_rate):
    try:
        parts = timecode_str.split(':')
        if len(parts) == 3: # mm:ss:ff
            m, s, ff = map(int, parts)
            if frame_rate is None or frame_rate <= 0:
                print("🟡 WARNING: Invalid frame rate for timecode conversion. Using frames as 0.")
                return m * 60 + s
            return m * 60 + s + (ff / frame_rate)
        else:
            print(f"🟡 WARNING: Timecode '{timecode_str}' not in mm:ss:ff. Returning None.")
            return None
    except ValueError:
        print(f"🟡 WARNING: Could not parse timecode '{timecode_str}'. Returning None.")
        return None

# --- Main Program ---
print("--- Starting batch video clipping (re-encoding for HIGHEST quality) ---")

if not os.path.exists(ffmpeg_path) or not os.path.exists(ffprobe_path):
    print(f"❌ CRITICAL ERROR: ffmpeg.exe or ffprobe.exe not found in {script_directory}")
    exit()
if not os.path.isdir(download_folder_abs_path):
    print(f"❌ ERROR: Source folder '{DOWNLOADED_VIDEOS_FOLDER}' not found.")
    exit()
if not os.path.exists(clipped_folder_abs_path):
    os.makedirs(clipped_folder_abs_path)
    print(f"✅ Output folder '{CLIPPED_VIDEOS_FOLDER}' created.")

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        video_data = json.load(f)
    print(f"✅ Loaded {len(video_data)} entries from '{json_file_name}'.")
except Exception as e:
    print(f"❌ ERROR: Failed to load or parse JSON file '{json_file_name}': {e}")
    exit()

processed_count = 0
skipped_count = 0
error_count = 0

for i, item in enumerate(video_data):
    try:
        video_name = item['video_name']
        start_frame_tc = item['start_frame']
        end_frame_tc = item['end_frame']
        video_id = item.get('id', 'N/A')
    except KeyError as e:
        print(f"⚠️ Skipping entry {i+1} (ID: {video_id}), missing key: {e}")
        skipped_count += 1
        continue

    print("-" * 50)
    print(f"▶️ Processing video {i+1}/{len(video_data)} (ID: {video_id}) - {video_name}")

    original_video_path = os.path.join(download_folder_abs_path, video_name)
    clipped_video_path = os.path.join(clipped_folder_abs_path, video_name)

    if not os.path.exists(original_video_path):
        print(f"🟡 WARNING: Source file '{original_video_path}' not found. Skipping.")
        skipped_count += 1
        continue
    if os.path.exists(clipped_video_path):
        print(f"🟢 Clipped file '{video_name}' already exists. Skipping.")
        skipped_count += 1
        continue

    frame_rate = get_frame_rate(original_video_path)
    if frame_rate is None:
        print(f"🔴 ERROR: Cannot get frame rate for '{video_name}'. Skipping.")
        error_count += 1
        continue
    print(f"   Frame rate: {frame_rate:.2f} fps")

    start_total_seconds = timecode_to_seconds(start_frame_tc, frame_rate)
    end_total_seconds = timecode_to_seconds(end_frame_tc, frame_rate)

    if start_total_seconds is None or end_total_seconds is None:
        print(f"🔴 ERROR: Invalid timecode(s) for '{video_name}'. Skipping.")
        error_count += 1
        continue

    duration_seconds = end_total_seconds - start_total_seconds
    if duration_seconds <= 0:
        print(f"🔴 ERROR: Calculated duration is zero or negative for '{video_name}' (Start: {start_total_seconds:.2f}s, End: {end_total_seconds:.2f}s). Skipping.")
        error_count += 1
        continue
    
    print(f"   Original timecodes: Start {start_frame_tc}, End {end_frame_tc}")
    print(f"   Converted: Start {start_total_seconds:.3f}s, Duration {duration_seconds:.3f}s (End at {end_total_seconds:.3f}s)")
    print(f"   Output path: {clipped_video_path}")
    print(f"⏳ Calling ffmpeg to clip (re-encoding for highest quality, this will be SLOW)...")
    
    # --- FFmpeg command for HIGHEST QUALITY RE-ENCODING ---
    command = [
        ffmpeg_path,
        '-i', original_video_path,
        '-ss', str(start_total_seconds),  # Start time in seconds
        '-t', str(duration_seconds),      # Duration of the clip in seconds
        
        # Video encoding settings for near-lossless quality
        '-c:v', 'libx264',          # Video codec: H.264
        '-preset', 'slower',        # Slower preset for better compression/quality. 'veryslow' is even slower.
        '-crf', '18',               # Constant Rate Factor (quality). 17-18 is often considered visually lossless. Lower is better quality.
        '-pix_fmt', 'yuv420p',      # Pixel format for broad compatibility
        
        # Audio encoding settings for high quality
        '-c:a', 'aac',              # Audio codec: AAC
        '-b:a', '320k',             # Audio bitrate: 320kbps (high quality)
        
        '-y',                             # Overwrite output without asking
        '-loglevel', 'warning',           # Show warnings and errors from ffmpeg
        clipped_video_path
    ]
    
    # If you still prefer to use -to (end timestamp) instead of -t (duration):
    # command = [
    #     ffmpeg_path,
    #     '-i', original_video_path,
    #     '-ss', str(start_total_seconds),
    #     '-to', str(end_total_seconds), 
    #     '-c:v', 'libx264', '-preset', 'slower', '-crf', '18', '-pix_fmt', 'yuv420p',
    #     '-c:a', 'aac', '-b:a', '320k',
    #     '-y', '-loglevel', 'warning',
    #     clipped_video_path
    # ]

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            if os.path.exists(clipped_video_path) and os.path.getsize(clipped_video_path) > 100:
                print(f"✅ Successfully re-encoded (high quality) and saved: {video_name}")
                processed_count += 1
            else:
                print(f"🔴 ERROR: ffmpeg reported success, but output file '{video_name}' is empty or too small after re-encoding.")
                print(f"   ffmpeg stdout:\n   {result.stdout.strip()}")
                print(f"   ffmpeg stderr:\n   {result.stderr.strip()}")
                error_count += 1
        else:
            print(f"❌ ERROR: Failed to re-encode video (ID: {video_id}, File: {video_name}).")
            print(f"   ffmpeg stdout:\n   {result.stdout.strip()}")
            print(f"   ffmpeg stderr:\n   {result.stderr.strip()}")
            error_count += 1
    except Exception as e:
        print(f"❌ Unknown Python error during clipping (ID: {video_id}, File: {video_name}): {e}")
        error_count += 1

print("-" * 50)
print("\n🎉 All clipping tasks completed!")
print(f"   Successfully processed: {processed_count} videos")
print(f"   Skipped: {skipped_count} videos")
print(f"   Errors occurred: {error_count} videos")

if error_count > 0:
    print("\n🔴 NOTE: Some errors occurred. Please check the ffmpeg output above for details.")