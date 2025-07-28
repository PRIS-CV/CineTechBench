#!/bin/bash

INFER_DIR="inference/processed"
GT_ROOT="anno"
GT_ROOT_video="anno_video"
SAVE_BASE="results"

# 1) 列出所有模型并打印
echo "Found models:"
for dir in "$INFER_DIR"/*; do
  model=$(basename "$dir")
  echo "  - $model"
done

# 2) 依次对每个模型调用 eval2.py
for dir in "$INFER_DIR"/*; do
  model=$(basename "$dir")
  echo ""
  echo "=== Evaluating $model ==="
  python3 evaluation/eval2.py \
    --caption_file_root "$INFER_DIR/$model" \
    --gt_file_root        "$GT_ROOT" \
    --gt_file_root_video        "$GT_ROOT_video" \
    --save_root           "$SAVE_BASE/$model" 
done
