# Benchmark Preparation 
This is a introduction on how to prepare our benchmark.

## Download annotation from huggingface

```bash
cd dataset
mkdir annotations
# If you are cannot assess huggingface
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download Xinran0906/CineTechBench --local-dir ./annotation
```

## Download the raw videos 
Please download the raw videos the according to the instruction[here](./download_video.md).

```bash
mkdir raw
python download_videos.py
```

## Clip the raw videos 
Please first prepare the ffmpeg according to the instruction [here](./ffmpeg.md). Then execute the following script to clip the raw video.

```bash
mkdir clips
python clip_videos.py
```

## Download the images
Please download the images according to the image links in `./annotation/image_annotation.json`
