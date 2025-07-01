#!/usr/bin/env python3
import json
import os
import sys

#METRICS = ["Scale", "Angle", "Composition", "Colors", "Lighting", "Focal Lengths"]
METRICS = ["Movement"]

def load_anno_ids(anno_dir):
    """读取 anno 输入目录下每个 metric.jsonl，返回 {metric: set(file_id)}"""
    ids = {}
    for m in METRICS:
        path = os.path.join(anno_dir, f"{m}.jsonl")
        if not os.path.isfile(path):
            print(f"⚠️ 注释文件不存在：{path}", file=sys.stderr)
            ids[m] = set()
            continue
        with open(path, encoding="utf-8") as f:
            s = set(json.loads(line)["file_id"] for line in f)
        ids[m] = s
    return ids

def split_descriptions(model_name, anno_dir, inf_dir, out_dir):
    # 1. 读入 Cinematic 描述 JSON
    inf_path = os.path.join(inf_dir, f"{model_name}-CinematicShotStyle-Video-Description.json")
    if not os.path.isfile(inf_path):
        print(f"❌ 找不到描述文件：{inf_path}", file=sys.stderr)
        sys.exit(1)
    with open(inf_path, encoding="utf-8") as f:
        desc_data = json.load(f)

    # 2. 加载每个维度的 file_id 集合
    anno_ids = load_anno_ids(anno_dir)

    # 3. 定义并检查输出目录，已存在则跳过
    target_dir = os.path.join(out_dir, model_name)
    if os.path.exists(target_dir):
        print(f"⚠️ 目标目录已存在，跳过生成：{target_dir}", file=sys.stderr)
        return

    # 4. 创建输出目录
    os.makedirs(target_dir, exist_ok=True)

    # 5. 对每个维度，筛选并写入新的 jsonl
    for m in METRICS:
        out_path = os.path.join(target_dir, f"{m}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fout:
            for file_id, attrs in desc_data.items():
                if file_id not in anno_ids[m]:
                    continue
                caption = attrs.get("Description")
                if not caption:
                    continue
                record = {
                    "file_id": file_id,
                    "caption": caption
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✅ 已写入 {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python3 split_iference_by_model.py \
              <模型名称> <anno_INPUT_PATH> <inf_INPUT_PATH> <OUTPUT_PATH>",
              file=sys.stderr)
        sys.exit(1)

    model_name, anno_input, inf_input, output_base = sys.argv[1:]
    split_descriptions(model_name, anno_input, inf_input, output_base)
