# CAPability 评估说明

## 概述

本文档介绍如何设置和运行 CAPability 视觉描述评估流程，帮助您快速完成模型输出与人工标注的对比评估。

## 环境要求

- conda activate capability

- #终端执行 
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890
(capability) (base) wangxinran@scaling:/mnt/sdb/zhangyuxuan/CAPability-main$ bash eval_run_all.sh


## 前期准备



## 使用方法

请使用以下命令模板运行评估脚本，根据实际路径和参数需求进行调整：

```bash
python3 evaluation/eval.py \
  --caption_file_root our_inference/tiaozheng_description/Doubao-1.5-Vision-Pro \
  --gt_file_root        our_anno \
  --save_root           our_evaluation/Doubao-1.5-Vision-Pro \
  --tasks               ["Scale"]

```

### 参数说明

- `--caption_file_root`：模型生成的 caption `jsonl` 文件所在目录前缀，如 `inference/output/gemini-1.5-pro`
- `--gt_file_root`：人工标注真值目录，默认为 `annotations`
- `--save_root`：评估结果输出前缀，脚本运行后会生成 `<save_root>.metrics.json` 或 `<save_root>.summary.json` 等文件
- `--tasks`：要评估的维度列表，使用 `all` 表示全部，或逗号分隔多项，例如 `object_color,scene`
- `--num_process`：并行进程数，设置为 `0` 表示单进程运行
- `--eval_model`：用于评估的 OpenAI 模型名称，如 `gpt-4-turbo`、`gpt-4.1-nano`
- `--max_retry_times`：API 调用的最大重试次数，避免因网络或限流失败
- `--max_allow_missing`：允许的最大缺失预测数，超过则跳过该维度

## 示例(原来代码的 已废除)


### 示例 1：仅评估 "scene", "camera_angle","action", "camera_movement"

```bash
python evaluation/eval.py \
  --caption_file_root inference/output/gemini-1.5-pro \
  --gt_file_root     annotations \
  --save_root        evaluation_test/output/gemini-1.5-pro \
  --tasks            "scene,camera_angle,action,camera_movement" \
  --num_process      0 \
  --eval_model       gpt-4.1-nano \
  --max_retry_times  5 \
  --max_allow_missing 5
```
这样就好：
```bash
python evaluation/eval.py \
  --tasks            "scene,camera_angle,action,camera_movement" 
```
