
# CineTechBench Bulk Video Downloader

This script automates the process of downloading videos specified in a JSON file using `yt-dlp`. It is designed to be self-contained, requiring all necessary executables and configuration files to be in the same directory as the script. This ensures it works across different machines without relying on system-wide installations.

## Synopsis

This Python script reads video metadata from a 'CineTechBench_Video_Annotation.json' file, which contains video IDs, names, and source links. For each entry, it invokes `yt-dlp` to download the best quality MP4 video and audio streams and then uses FFmpeg to merge them into a single MP4 file. The script requires browser cookies to access videos that may be behind a login wall and saves the final output into a 'Download_videos' subfolder, skipping any videos that have already been downloaded.

## Features

-   **Bulk Downloading**: Reads a JSON file to download multiple videos in one go.
-   **Cross-Platform**: Automatically detects the OS (Windows, Linux, macOS) to use the correct executable names.
-   **Self-Contained**: Does not depend on system PATH. All required tools (`yt-dlp`, `ffmpeg`) are used from the script's local directory.
-   **High-Quality Downloads**: Fetches the best available MP4 video and audio streams and merges them.
-   **Cookie Support**: Can use browser cookies to download videos from sites requiring login.
-   **Skips Existing Files**: Checks if a video already exists before downloading to save time and bandwidth.
-   **Automatic Folder Creation**: Creates the destination folder if it doesn't exist.

## Prerequisites

Before running the script, you must set up the following file structure. All listed files and folders (except for `Download_videos`, which is created automatically) must be placed in the **same directory**.

/your_project_folder/
├── download_videos.py                  <-- The main script
├── video_annotation.json               <-- The JSON file with video info
├── yt-dlp.exe                          <-- (For Windows)
├── yt-dlp                              <-- (For Linux/macOS)
├── ffmpeg.exe                          <-- (For Windows)
├── ffprobe.exe                         <-- (For Windows)
├── ffmpeg                                <-- (For Linux/macOS)
├── ffprobe                               <-- (For Linux/macOS)
├── cookies.txt                         <-- Your browser cookies file
└── /Download_videos/                   <-- This folder will be created by the script

### 1. `yt-dlp` Executable

This tool is essential for downloading videos.
-   Go to the [yt-dlp GitHub Releases page](https://github.com/yt-dlp/yt-dlp/releases/latest).
-   Download the correct executable for your operating system (`yt-dlp.exe` for Windows, `yt-dlp` for Linux/macOS).
-   Place the file in your project folder.

### 2. `ffmpeg` and `ffprobe` Executables

These tools are required by `yt-dlp` to merge separate video and audio files into a single MP4.
-   **For Windows:**
    1.  Download a pre-built FFmpeg release. A good source is [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (get the "full" release zip).
    2.  Extract the downloaded `.zip` file.
    3.  Go into the `bin` folder.
    4.  Copy `ffmpeg.exe` and `ffprobe.exe` and paste them into your project folder.
-   **For Linux/macOS:**
    1.  Download a pre-built static release from [johnvansickle.com](https://johnvansickle.com/ffmpeg/).
    2.  Extract the `.tar.xz` archive.
    3.  Copy the `ffmpeg` and `ffprobe` files from the extracted directory into your project folder.
    4.  **Important**: You must give them execute permissions. Open a terminal in your project folder and run:
        ```bash
        chmod +x ffmpeg
        chmod +x ffprobe
        chmod +x yt-dlp
        ```

### 3. `cookies.txt` File

Cookies are necessary to download videos from sites that require you to be logged in (e.g., certain Bilibili or YouTube videos).
1.  Install a browser extension that can export cookies in the `Netscape` format. A recommended one is **"Get cookies.txt LOCALLY"** (available for [Chrome](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) and [Firefox](https://addons.mozilla.org/en-US/firefox/addon/get-cookies-txt-locally/)).
2.  Navigate to the website where the videos are hosted (e.g., `bilibili.com`).
3.  Click the extension's icon in your browser toolbar.
4.  Click the "Export" button. This will download a `.txt` file.
5.  Rename the downloaded file to `cookies.txt` and place it in your project folder.

### 4. `video_annotation` File

This is the input file containing the list of videos to download. It must be in the project folder and contain a JSON array where each object has at least an `id`, `video_name`, and `link`.

## How to Use

1.  **Verify Setup**: Ensure all the required files from the "Prerequisites" section are correctly placed in one folder.
2.  **Open Terminal**: Open a command prompt (CMD/PowerShell on Windows) or terminal (on Linux/macOS).
3.  **Navigate to Folder**: Use the `cd` command to navigate to your project folder.
    ```bash
    cd path/to/your_project_folder
    ```
4.  **Run the Script**: Execute the script using Python.
    ```bash
    python download_videos.py
    ```
5.  **Check Output**: The script will begin processing the JSON file. Download progress will be displayed in the terminal. The final video files will appear in the `Download_videos` subfolder.

## Disclaimer

This code and its associated tools are intended for **academic research purposes only**.

Any use of this code for commercial purposes or non-academic research is strictly prohibited. The user assumes all liability for any consequences arising from the use of this code, including but not limited to copyright infringement and violations of terms of service of the websites from which the content is downloaded. The developers of this script are not responsible for the user's actions.