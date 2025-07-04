#!/usr/bin/env python3
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO-style image captions JSON from custom descriptions."
    )
    parser.add_argument(
        "--coco_path", "-c",
        type=str,
        required=True,
        help="路径到 COCO 格式 JSON 文件，需包含 'images' 列表"
    )
    parser.add_argument(
        "--res_path", "-r",
        type=str,
        required=True,
        help="路径到包含描述的 JSON 文件"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="输出路径"
    )
    args = parser.parse_args()

    coco_path = args.coco_path
    res_path = args.res_path
    output_path = args.output_dir

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read the 'images' list from the COCO-format file
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Load custom descriptions
    with open(res_path, 'r', encoding='utf-8') as f:
        res = json.load(f)

    # Build mapping from file_name to image_id
    file2id = {img['file_name']: img['id'] for img in coco.get('images', [])}

    # Generate output list
    output_list = []
    for file_name, attrs in res.items():
        img_id = file2id.get(file_name)
        if img_id is None:
            print(f"!!! {file_name} 未在 {coco_path} 的 images 中找到")
            continue

        raw = attrs.get('Description', '')
        caption = (raw[0] if isinstance(raw, list) and raw else raw).strip()

        output_list.append({
            "image_id": img_id,
            "caption": caption
        })

    # Write output JSON list
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成 → {output_path}")

if __name__ == "__main__":
    main()
