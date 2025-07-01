import os
import json

# 输入、输出路径
input_path = 'our_anno_video/original/video_description.json'
output_dir = 'our_anno_video/right_anno'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'video_description_coco.json')

# 载入原始数据
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

images = []
annotations = []
ann_id = 1
img_id = 1

# 遍历每张图
for file_name, attrs in data.items():
    # images 列表
    images.append({
        "id": img_id,
        "file_name": file_name
    })
    # 处理 Description → caption
    desc = attrs.get("Description", "").strip()
    # 如果未来有多条 caption，可改为：captions = attrs.get("Description", [])
    captions = [desc]  # 目前整段当一条 caption

    for cap in captions:
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "caption": cap
        })
        ann_id += 1

    img_id += 1

# 构造 COCO 格式字典
coco_fmt = {
    "images": images,
    "annotations": annotations,
    "type": "captions"
}

# 写出结果
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(coco_fmt, f, ensure_ascii=False, indent=2)

print(f"✅ 已输出 → {output_path}")
