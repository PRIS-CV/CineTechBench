import os
import json

# Input and output paths
input_path = 'anno/original/description.json'
output_dir = 'anno/processed'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'description_coco.json')

# Load raw data
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

images = []
annotations = []
ann_id = 1
img_id = 1

# Iterate over each image
for file_name, attrs in data.items():
    # images list
    images.append({
        "id": img_id,
        "file_name": file_name
    })
    # Convert Description → caption
    desc = attrs.get("Description", "").strip()
    # If there are multiple captions in the future, you can change to: captions = attrs.get("Description", [])
    captions = [desc]  # Currently treat the whole paragraph as one caption

    for cap in captions:
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "caption": cap
        })
        ann_id += 1

    img_id += 1

# Build dictionary in COCO format
coco_fmt = {
    "images": images,
    "annotations": annotations,
    "type": "captions"
}

# Write output
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(coco_fmt, f, ensure_ascii=False, indent=2)

print(f"✅ 已输出 → {output_path}")
