0 环境：
conda env create -f capability_conda.yml
conda activate capability
pip install -r requirements.txt

1 运行split_annotation_by_metrics.py, 得到anno文件夹中的6个jsonl的gt
2 把models_generated_image_description放到inference文件夹下面，运行process_all.sh获取正确格式的输入
3 运行eval_run_all.sh进行测评