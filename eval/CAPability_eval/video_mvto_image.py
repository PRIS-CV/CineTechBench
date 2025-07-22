import os
import shutil

src_base = "our_inference_video/tiaozheng_description"
dst_base = "our_inference/tiaozheng_description_2"

for model_name in os.listdir(src_base):
    src_model_path = os.path.join(src_base, model_name)
    dst_model_path = os.path.join(dst_base, model_name)

    if not os.path.isdir(src_model_path):
        continue  # 忽略非目录

    # 创建目标模型目录（如果不存在）
    if not os.path.exists(dst_model_path):
        os.makedirs(dst_model_path)
        print(f"✅ 创建目录: {dst_model_path}")

    # 遍历模型目录下的所有文件/子目录
    for item in os.listdir(src_model_path):
        src_item_path = os.path.join(src_model_path, item)
        dst_item_path = os.path.join(dst_model_path, item)

        # 如果目标已存在该项，跳过
        if os.path.exists(dst_item_path):
            print(f"⚠️ 已存在: {dst_item_path}，跳过")
            continue

        # 文件或目录复制
        if os.path.isdir(src_item_path):
            shutil.copytree(src_item_path, dst_item_path)
        else:
            shutil.copy2(src_item_path, dst_item_path)
        print(f"📁 复制: {src_item_path} ➜ {dst_item_path}")
