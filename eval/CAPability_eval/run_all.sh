#!/usr/bin/env bash
# 这是处理测评输入数据的
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



# 循环执行
for m in "${models[@]}"; do
  echo "⏳ 正在处理模型: $m"
  python split_iference_by_model.py \
    "$m" \
    ./our_anno_video \
    ./our_inference_video/models_generated_video_description \
    ./our_inference_video/tiaozheng_description
done

echo "🎉 全部模型处理完毕！"
