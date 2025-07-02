#!/usr/bin/env python3
import json
import os

# 输入 / 输出设置
INPUT_JSON  = "image_description.json"
OUTPUT_DIR  = "anno"                       # ← 新增：统一输出目录
METRICS     = ["Scale", "Angle", "Composition",
               "Colors", "Lighting", "Focal Lengths"]

def normalize_task_name(metric: str) -> str:
    """把指标名转成子目录前缀，例如 Focal Lengths → focal_lengths"""
    return metric.lower().replace(" ", "_")

def main():
    # 1. 读入整张 JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 为每个 metric 打开一个输出文件（覆盖写入）
    writers = {}
    for m in METRICS:
        fn = os.path.join(OUTPUT_DIR, f"{m}.jsonl")  
        writers[m] = open(fn, "w", encoding="utf-8")

    # 4. 遍历每个 image 条目
    for file_id, attrs in data.items():
        for metric in METRICS:
            val = attrs.get(metric)
            # 跳过缺失、空值或 N/A
            if not val or val == "N/A":
                continue

            task = metric
            record = {
                "file_id":   file_id,
                "file_path": os.path.join(
                    "data",
                    normalize_task_name(metric),
                    f"{file_id}.jpg"
                ),
                "annotation": val,
                "task":      task,
                "data_type": "image"
            }
            # 写入一行 JSONL
            writers[metric].write(json.dumps(record, ensure_ascii=False) + "\n")

    # 5. 关闭所有文件
    for w in writers.values():
        w.close()

    print(f"✅ 拆分完成，6 个 .jsonl 文件已保存到 ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
