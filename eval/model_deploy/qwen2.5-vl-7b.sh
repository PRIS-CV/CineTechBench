CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path PATH_TO_YOUR/Qwen2.5-VL-7B-Instruct \
  --chat-template qwen2-vl \
  --host YOUR_SERVER_HOST \
  --context-len 32000 \
  --port YOUR_SERVER_PORT \