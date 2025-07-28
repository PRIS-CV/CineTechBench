import os
import json

# 根目录
base_dir = 'xxxx'

# 最终汇总字典
aggregated = {}

# 遍历 each 模型文件夹
for model_name in os.listdir(base_dir):
    summary_path = os.path.join(base_dir, model_name, 'data', 'summary.json')
    if not os.path.isfile(summary_path):
        continue

    # 读取原始 summary.json
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 重命名并存入汇总
    aggregated[model_name] = {
        'HR':  data.get('average_hit_rate'),
        'AP':  data.get('average_precision'),
        'AR':  data.get('average_recall'),
        'F1':  data.get('average_f1_score'),
    }

# 将汇总结果写入文件
out_path = os.path.join(base_dir, 'aggregated_summary_add_video.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(aggregated, f, indent=2, ensure_ascii=False)

print(f"已生成汇总文件：{out_path}")
