#!/usr/bin/env bash
set -euo pipefail

COCO_PATH="anno/processed/description_coco.json"
OUTPUT_DIR="res/processed"
mkdir -p "${OUTPUT_DIR}"

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

  RES_PATH="res/models_generated_image_description/${m}-CinematicShotStyle-Image-Description.json"
  if [[ ! -f "${RES_PATH}" ]]; then
    echo "!!! Warning: ${RES_PATH} not found, skipping."
    continue
  fi

  OUTPUT_PATH="${OUTPUT_DIR}/${m}_coco.json"

  python create/res_proceses.py \
    --coco_path "${COCO_PATH}" \
    --res_path  "${RES_PATH}" \
    --output_dir "${OUTPUT_PATH}"

  echo "✅ Generated → ${OUTPUT_PATH}"
done

echo "🎉 All done!"
