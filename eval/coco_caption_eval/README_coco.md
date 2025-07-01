0 环境配置:
conda env create -f coco_conda.yml
conda activate coco
pip install -r requirements.txt


1 用create文件夹中的anno_process.py生成正确格式的annotation
2 用generate_all.sh生成正确格式的result
3 用eval_run_all.sh测评所有结果