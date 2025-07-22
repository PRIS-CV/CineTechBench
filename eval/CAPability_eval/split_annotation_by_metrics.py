import json
import os
from pathlib import Path
from typing import Dict, TextIO, Optional

# 输入 / 输出
INPUT_JSON = "image_description_annotation.json"
OUT_DIR = "anno" 
METRICS = [
    "Scale", "Angle", "Composition",
    "Colors", "Lighting", "Focal Lengths"
]

# 错别字 / 别名映射
ALIAS_MAP = {
    "compostion": "composition",
    "focal_lengths": "focal lengths",
}

def normalize_metric_name(metric: str) -> str:
    """小写、去空格后用于比较"""
    return metric.lower().replace(" ", "")

def canonical_metric(raw: str) -> Optional[str]:
    """
    把 Category 字段映射到 METRICS 的“正名”；
    若无效则返回 None。
    """
    key = normalize_metric_name(raw)
    key = ALIAS_MAP.get(key, key)
    for m in METRICS:
        if normalize_metric_name(m) == key:
            return m
    return None

def normalize_task_name(metric: str) -> str:
    """生成子文件夹名 / file_path 前缀"""
    return metric.lower().replace(" ", "_")

def prepare_writers(metrics) -> Dict[str, TextIO]:
    os.makedirs(OUT_DIR, exist_ok=True) 
    writers = {}
    for m in metrics:
        path = os.path.join(OUT_DIR, f"{m}.jsonl") 
        writers[m] = open(path, "w", encoding="utf-8")
    return writers

def main() -> None:
    # 1) 读入列表格式 JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as fp:
        records = json.load(fp)

    # 2) 输出文件
    writers = prepare_writers(METRICS)

    # 3) 遍历每条记录
    for rec in records:
        metric = canonical_metric(rec.get("Category", ""))
        if metric is None:
            continue
        val = rec.get("Annotation")
        if not val or val == "N/A":
            continue
        file_id = rec.get("Image_name")
        if not file_id:
            continue

        record = {
            "file_id":   file_id,
            "file_path": os.path.join("data", normalize_task_name(metric), file_id),
            "annotation": val,
            "task":      metric,
            "data_type": "image"
        }
        writers[metric].write(json.dumps(record, ensure_ascii=False) + "\n")

    # 4) 关闭文件
    for w in writers.values():
        w.close()

    print("✅ 拆分完成，生成 6 个 .jsonl 文件。")

if __name__ == "__main__":
    main()