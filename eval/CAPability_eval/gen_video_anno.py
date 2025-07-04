#!/usr/bin/env python3
import json
import os

def convert_to_jsonl(input_json_path: str, output_dir: str = "anno_vedio"):
    """
    读取 video_description.json，将 key 作为 file_id，Type 作为 annotation，生成 JSONL 文件。
    输出文件 Movement.jsonl 将保存在 output_dir 目录下。
    记录字段：
      - file_id
      - file_path (格式: data/video/<file_id>)
      - annotation
      - task ("Movement")
      - data_type ("video")
    """
    # 读取原始 JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl_path = os.path.join(output_dir, "Movement.jsonl")

    unique_annotations = set()

    # 写 JSONL
    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for file_id, info in data.items():
            annotation = info.get('Type', '')
            unique_annotations.add(annotation)

            record = {
                "file_id":   file_id,
                "file_path": f"data/video/{file_id}",
                "annotation": annotation,
                "task":      "Movement",
                "data_type": "video"
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 输出汇总信息
    print("✅ 已生成 JSONL 文件:", output_jsonl_path)
    print("Unique annotations:")
    for ann in sorted(unique_annotations):
        print(f"- {ann}")

if __name__ == "__main__":
    convert_to_jsonl("video_description.json")
