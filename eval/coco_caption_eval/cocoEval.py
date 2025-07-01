# ---- Code cell ----

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import argparse
import os
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# ---- Code cell ----
def main():
    parser = argparse.ArgumentParser(
        description="Run COCO evaluation and plot CIDEr histogram for a given model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="模型名称，例如 Doubao-1.5-Vision-Pro"
    )
    args = parser.parse_args()
    model = args.model

    # 固定的标注文件
    annFile = "our_anno_video/right_anno/video_description_coco.json"
    # 各模型对应的结果 JSON
    resFile = f"our_res_video/right_res/{model}_coco.json"
    # COCO Eval 输出的各图片评估结果
    evalImgsFile = f"our_eval_res_video/{model}/evalImgsFile.json"
    # COCO Eval 输出的汇总结果
    evalFile = f"our_eval_res_video/{model}/evalFile.json"
    # 绘制并保存 CIDEr 分数直方图的路径
    saveimage = f"our_eval_res_video/{model}/cider_histogram.png"

    # 确保输出目录存在
    out_dir = os.path.dirname(evalImgsFile)
    os.makedirs(out_dir, exist_ok=True)

    # Debug 打印路径，确认无误
    print("▶️  Annotation file:   ", annFile)
    print("▶️  Results file:      ", resFile)
    print("▶️  EvalImgs output:   ", evalImgsFile)
    print("▶️  Eval summary file: ", evalFile)
    print("▶️  Histogram image:   ", saveimage)


    # ---- Code cell ----
    # create coco object and cocoRes object
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    print(f"coco{coco}")
    print(f"cocoRes{cocoRes}")

    # ---- Code cell ----
    # create cocoEval object by taking coco and cocoRes

    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    # cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    #不训练SPICE了 太慢，直接在eval.py里改了
    cocoEval.evaluate()

    # ---- Code cell ----
    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print( '%s: %.3f'%(metric, score))

    # ---- Code cell ----
    # demo how to use evalImgs to retrieve low score result
    evals = [eva for eva in cocoEval.evalImgs if eva['CIDEr']<30]
    print ('ground truth captions')
    imgId = evals[1]['image_id']
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

    print ('\n')
    print ('generated caption (CIDEr score %0.1f)'%(evals[0]['CIDEr']))
    annIds = cocoRes.getAnnIds(imgIds=imgId)
    anns = cocoRes.loadAnns(annIds)
    coco.showAnns(anns)

    '''
    img = coco.loadImgs(imgId)[0]
    try:
        I = io.imread(f'{dataDir}/images/{dataType}/{img["file_name"]}')
        plt.imshow(I); plt.axis('off'); plt.show()
    except FileNotFoundError as e:
        print('⚠️ 找不到图片：', e.filename)
    '''

    # ---- Code cell ----
    # plot score histogram
    ciderScores = [eva['CIDEr'] for eva in cocoEval.evalImgs]
    plt.hist(ciderScores)
    plt.title('Histogram of CIDEr Scores', fontsize=20)
    plt.xlabel('CIDEr score', fontsize=20)
    plt.ylabel('result counts', fontsize=20)
    import numpy as np
    print("最小值、最大值：", np.min(ciderScores), np.max(ciderScores))
    print("去重后有多少种值：", len(set(ciderScores)))


    # 保存图到文件（格式可选 png/jpg/pdf，dpi 可调）
    plt.tight_layout()  # 防止标题和标签被裁剪
    plt.savefig(saveimage, dpi=200)


    #plt.show()

    # ---- Code cell ----
    # save evaluation results to ./results folder
    json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
    json.dump(cocoEval.eval,     open(evalFile, 'w'))

    # ---- Code cell ----

if __name__ == "__main__":
    main()









'''
# ---- Code cell ----
# set up file names and paths
dataDir  = '.'
dataType = 'val2014'
algName  = 'fakecap'

# 标注文件
annFile = f"{dataDir}/annotations/captions_{dataType}.json"

# 结果文件
subtypes = ['results', 'evalImgs', 'eval']
resFile, evalImgsFile, evalFile = [
    f"{dataDir}/results/captions_{dataType}_{algName}_{subtype}.json"
    for subtype in subtypes
]
print(f"annFile={annFile}")
print(f"resFile={resFile}")


# ---- Code cell ----
#自定义1
annFile = "our_anno/right_anno/image_description_coco.json"
resFile = "our_res/right_res/image_captions_coco.json"
evalImgsFile = "our_eval_res/evalImgsFile.json"
evalFile = "our_eval_res/evalFile.json"
saveimage = "our_eval_res/cider_histogram.png"
print(f"annFile={annFile}")
print(f"resFile={resFile}")
print(f"evalImgsFile={evalImgsFile}")
print(f"evalFile={evalFile}")
'''