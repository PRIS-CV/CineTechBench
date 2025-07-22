#!/usr/bin/env bash

# 如果在其他目录下执行，请先 cd 到脚本所在目录
# cd /path/to/your/coco-caption

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
  "image_captions"
  "LLaVA-NeXT-Video-7B"
)

for model in "${models[@]}"; do
  echo "⏳ 正在处理模型: $model"
  python cocoEval.py --model "$model"
  echo "✅ 模型 $model 处理完毕"
  echo
done

echo "🎉 全部模型评估完成！"
