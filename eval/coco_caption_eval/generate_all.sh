#!/usr/bin/env bash
set -euo pipefail

# COCO 格式文件固定路径
COCO_PATH="our_anno_video/right_anno/video_description_coco.json"
# 输出目录
OUTPUT_DIR="our_res_video/right_res"
mkdir -p "${OUTPUT_DIR}"

# 模型列表
models=(
  "Doubao-1.5-Vision-Pro"
  "Gemini-2.0-Flash"
  "Gemini-2.5-Pro"
  "gemma3-4b-it"
  "Glm-4V-Plus"
  "GPT-4o"
  "InternVL2.5-8B"
  "InternVL3-8B"
  "Kimi-VL-A3B-Instruct"
  "Llama-3.2-11B-Vision"
  "LLaVA-NeXT-8B"
  "LlaVA-OneVision-7B"
  "MiniCPM-V-2.6"
  "Phi-3.5-Vision-Instruct"
  "Qwen-VL-Plus"
  "Qwen2.5-Omini-7B"
  "Qwen2.5-VL-7B"
  "LLaVA-NeXT-Video-7B"
)

for m in "${models[@]}"; do
  echo "⏳ Processing model: $m"

  # 1. 构造该模型的 res_path
  RES_PATH="our_res_video/models_generated_video_description/${m}-CinematicShotStyle-Video-Description.json"
  if [[ ! -f "${RES_PATH}" ]]; then
    echo "!!! Warning: ${RES_PATH} not found, skipping."
    continue
  fi

  # Define the full output file path including filename
  OUTPUT_PATH="${OUTPUT_DIR}/${m}_coco.json"

  # 2. 调用 Python 脚本，输出到固定文件
  python zyx_tiaozheng/res_proceses.py \
    --coco_path "${COCO_PATH}" \
    --res_path  "${RES_PATH}" \
    --output_dir "${OUTPUT_PATH}"

  echo "✅ Generated → ${OUTPUT_PATH}"
done

echo "🎉 All done!"
