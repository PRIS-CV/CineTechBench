CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path /YOUR_PATH_TO/Qwen2.5-VL-7B-Instruct \
  --chat-template qwen2-vl \
  --host YOUR_HOST \
  --context-len 32000 \
  --port YOUR_PORT \