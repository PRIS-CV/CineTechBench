## FFmpeg Setup Instructions

FFmpeg is a crucial component required for video processing operations performed by our scripts, such as clipping and re-encoding videos. This guide provides installation steps for Windows, macOS, and Linux.

**Important Note for Python Scripts:** The Python scripts developed in our project (e.g., `clip_videos.py`) currently expect the `ffmpeg` (or `ffmpeg.exe` on Windows) and `ffprobe` (or `ffprobe.exe` on Windows) executables to be located **in the same directory as the Python script itself**. Please follow the OS-specific instructions below, and then ensure these executables are copied to your script's project folder if needed.

### Windows

**Step 1: Download FFmpeg Files**

1.  Open your web browser and go to the FFmpeg builds page on gyan.dev. This is a popular website for Windows users to get the latest and most stable FFmpeg versions.
    * **Direct Link**: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2.  On the page, locate the "release" builds section.
3.  Click on the link named `ffmpeg-release-full.7z` to start the download. This is the most feature-complete version and includes `ffmpeg.exe` and `ffprobe.exe`.

**Step 2: Extract the Files**

The file you downloaded is a `.7z` archive, which requires extraction software.

1.  If you don't have an extraction tool, we recommend downloading and installing **7-Zip** (it's completely free).
    * **7-Zip Official Website**: [https://www.7-zip.org/](https://www.7-zip.org/)
2.  Once 7-Zip is installed (or if you already have an archiver that handles .7z), find the `ffmpeg-release-full.7z` file you downloaded.
3.  Right-click on it, select "7-Zip" (or your archiver's option) from the context menu, and then choose "Extract to "ffmpeg-......\\"" (or a similar option like "Extract Here"). This will create a new folder in the current location.

**Step 3: Find and Copy `ffmpeg.exe` and `ffprobe.exe`**

1.  Navigate into the newly extracted folder (e.g., `ffmpeg-7.0-release-full`).
2.  You will see a subfolder named `bin`. Double-click to enter it.
3.  Inside the `bin` folder, you will find several `.exe` files, including `ffmpeg.exe` and `ffprobe.exe`.
4.  Select both `ffmpeg.exe` and `ffprobe.exe`. Right-click on them and choose "Copy".

**Step 4: Paste into the Script's Folder**

1.  Return to the folder where you have stored your Python script (e.g., `clip_videos.py`) and your JSON file (e.g., `CineTechBench_Video_Annotation.json`).
2.  In an empty space within this folder, right-click and select "Paste".

Your project folder should now contain at least:
* Your Python script (e.g., `clip_videos.py`)
* Your JSON file (e.g., `CineTechBench_Video_Annotation.json`)
* `ffmpeg.exe`
* `ffprobe.exe`
* (And `yt-dlp.exe` if your project also involves downloading videos with it using our scripts)

### macOS

1.  **Using Homebrew (Recommended):**
    * Open your Terminal (you can find it in Applications > Utilities).
    * If you don't have Homebrew installed, you can install it by pasting the command found on the official Homebrew website: [https://brew.sh/](https://brew.sh/)
    * Once Homebrew is installed, install FFmpeg by running the following command in your Terminal:
        ```bash
        brew install ffmpeg
        ```
    * This command installs FFmpeg and its related tools, including `ffmpeg` and `ffprobe`, system-wide. They are typically installed in a location like `/opt/homebrew/bin/` (for Apple Silicon Macs) or `/usr/local/bin/` (for Intel-based Macs).

2.  **Making FFmpeg accessible to the script:**
    * After installation, to make `ffmpeg` and `ffprobe` usable by our Python script (which expects them in its own directory), you need to copy these executables into your script's project folder.
    * First, find where they were installed by typing in the Terminal:
        ```bash
        which ffmpeg
        which ffprobe
        ```
    * Then, copy these files to your script's folder. For example, if `which ffmpeg` showed `/opt/homebrew/bin/ffmpeg` and your script is in `~/Projects/MyVideoProject`, you would run:
        ```bash
        cp /opt/homebrew/bin/ffmpeg ~/Projects/MyVideoProject/
        cp /opt/homebrew/bin/ffprobe ~/Projects/MyVideoProject/
        ```

### Linux

FFmpeg is available in the default repositories of most Linux distributions. You can install it using your distribution's package manager.

1.  **For Debian/Ubuntu-based distributions (e.g., Ubuntu, Mint):**
    * Open your Terminal.
    * It's good practice to update your package list first:
        ```bash
        sudo apt update
        ```
    * Then, install FFmpeg (this will usually include `ffmpeg` and `ffprobe`):
        ```bash
        sudo apt install ffmpeg
        ```

2.  **For Fedora-based distributions:**
    * Open your Terminal.
    * Install FFmpeg:
        ```bash
        sudo dnf install ffmpeg
        ```
    * Note: If `ffmpeg` is not found, you might need to enable the RPM Fusion repository first. Instructions can be found on the RPM Fusion website.

3.  **For other distributions (e.g., Arch Linux, openSUSE):**
    * Please use your distribution's specific package manager.
        * Arch Linux: `sudo pacman -S ffmpeg`
        * openSUSE: `sudo zypper install ffmpeg`

4.  **Making FFmpeg accessible to the script:**
    * These commands typically install FFmpeg system-wide (e.g., in `/usr/bin/` or `/usr/local/bin/`).
    * Similar to macOS, to use `ffmpeg` and `ffprobe` with our Python script (which expects them in its own directory), you'll need to copy them into your script's project folder.
    * Find their location:
        ```bash
        which ffmpeg
        which ffprobe
        ```
    * Then, copy them. For example, if `which ffmpeg` showed `/usr/bin/ffmpeg` and your script is in `~/Projects/MyVideoProject`:
        ```bash
        cp /usr/bin/ffmpeg ~/Projects/MyVideoProject/
        cp /usr/bin/ffprobe ~/Projects/MyVideoProject/
        ```

---

**Alternative for macOS and Linux (More Advanced Users):**

Instead of copying the `ffmpeg` and `ffprobe` executables into the script's directory after a system-wide installation, you could modify the Python script. The script can be updated to search for these executables in the system's PATH environment variable first (e.g., using Python's `shutil.which('ffmpeg')` and `shutil.which('ffprobe')`) before looking in its own directory. This would make the script directly use the system-wide installation without needing local copies. However, the provided scripts are currently set up for the bundled approach.