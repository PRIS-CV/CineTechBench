<p align="center" style="border-radius: 10px">
  <img src="asset/logov2.png" width="35%" alt="logo"/>
</p>

# 📽️ CineTechBench: A Benchmark for Cinematographic Technique Understanding and Generation

<div align="center">
<a href="https://pris-cv.github.io/CineTechBench/"><img src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a> &ensp;
<a href="http://arxiv.org/abs/2505.15145"><img src="https://img.shields.io/static/v1?label=Arxiv&message=Paper&color=red&logo=arxiv"></a> &ensp;
<a href="https://www.alphaxiv.org/overview/2505.15145v1"><img src="https://img.shields.io/static/v1?label=alphaXiv&message=Blog&color=red&logo=arxiv"></a> &ensp;
<a href="https://huggingface.co/datasets/Xinran0906/CineTechBench"><img src="https://img.shields.io/static/v1?label=Dataset&message=CineTechBench&color=yellow&logo=huggingface"></a> &ensp;
</div>


## 🔥 News
CineTechBench has been accepted at NeurIPS 2025 as a poster!

## 👀 Introduction
We present CineTechBench, a pioneering benchmark founded on precise, manual annotation by seasoned cinematography experts across key cinematography dimensions. Our benchmark covers seven essential aspects—shot scale, shot angle, composition, camera movement, lighting, color, and focal length—and includes over 600 annotated movie images and 120 movie clips with clear cinematographic techniques.




<div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
<!--   <img src="asset/tax.png" width="30%" alt="tax"/> -->
  <img src="asset/bench_compare.png" width="100%" alt="bench_compare"/>
</div>




## 📌 TODO
- [x] Video extraction script for movie clips
- [x] Camera trajectory similarity calculation script
- [x] Movie image link organization and documentation
- [x] Video Question-answering evaluation script
- [x] Image Question-answering evaluation script
- [x] Description evaluation script


## Prepare Benchmark
Due to the copyright, we cannot distributed the movie clips and images directly, here we provide [instructions](dataset/README.md) to download and preprocess the data in our benchmark. We upload the all image links in `image_annotation` file in our [CineTechBench HF Repo](https://huggingface.co/datasets/Xinran0906/CineTechBench).


## 💾 Environment

Create the conda environment:
```bash
conda create -n ctbench python=3.11 -y
conda activate ctbench
```

Install pytorch (e.g, cuda 12.4) and transformers
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.3
```

Install flash-attn
```
pip install flash-attn
```

Please prepare another conda environment following the instruction in [MonST3R](https://monst3r-project.github.io/) for estimating camera trajectory from input video.


## 📊 Evaluation

**Image Cinematographic Tech Dimension Question Answering**

We provide an example to evaluate Gemini-2.5-Pro on image dimensions, e.g, shot angle, lighting, focal length, ..., QA.
```
python image_qa_gemini_2.5_pro.py --json_path /path/to/your/image_annotation.json --image_path /path/to/your/image_folder
```

**Video Camera Movement Question Answering**

We provide an example to evaluate Gemini-2.5-Pro on camera movement QA.
```
python video_qa_gemini_2.5_pro.py --json_path /path/to/your/video_annotation.json --video_path /path/to/your/video_folder
```

**CineTech Description**

We provide code to evaluate MLLMs on description generation on metrics in CAPability and MSCOCO, see the instructions for [CAPability](eval/CAPability_eval) and [COCO](eval/coco_caption_eval).

**Video Camera Movement Generation**

Before evaluation, you should first prepare the generated videos and the original film clips. Then use [MonST3R](https://monst3r-project.github.io/) to estimate their camera trajectory. The result folder should be arranged like:

```text
- original_clips
  - result for movie clip 1 
  - result for movie clip 2
- wani2v_ct
  - result for generated movie clip 1 
  - result for generated movie clip 2
```

After preparing the camera trajectory estimation results, please use `eval/eval_ct.sh` to summary the results.



## 💽 Copyright
We fully respect the copyright of all films and do not use any clips for commercial purposes. Instead of distributing or hosting video content, we only provide links to publicly available, authorized sources (e.g., official studio or distributor channels). All assets are credited to their original rights holders, and our use of these links falls under fair‐use provisions for non‐commercial, academic research.



## 🤗 Acknowledgements
We would like to thank the contributors to the [Wan2.1](https://github.com/Wan-Video/Wan2.1), [FramePack](https://github.com/lllyasviel/FramePack), [CamI2V](https://github.com/ZGCTroy/CamI2V), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [LMDeploy](https://github.com/InternLM/lmdeploy), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V), [MovieNet](https://movienet.github.io/#), [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2), [MonST3R](https://monst3r-project.github.io/), [CAPability](https://capability-bench.github.io/) for their open research. We also wish to acknowledge [IMDb](https://www.imdb.com/) for its comprehensive movie database and the [MOVIECLIPS](https://www.youtube.com/@MOVIECLIPS) YouTube channel for its vast collection of high-quality clips, which were instrumental to our work.

## 📮 Contant

If you have any question please feel free to mail to wangxr@bupt.edu.cn.


## 🔗 Citation
```Text
@misc{wang2025cinetechbenchbenchmarkcinematographictechnique,
      title={CineTechBench: A Benchmark for Cinematographic Technique Understanding and Generation}, 
      author={Xinran Wang and Songyu Xu and Xiangxuan Shan and Yuxuan Zhang and Muxi Diao and Xueyan Duan and Yanhua Huang and Kongming Liang and Zhanyu Ma},
      year={2025},
      eprint={2505.15145},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.15145}, 
}
```


## 📄 License and Data Usage Policy

### General Terms

By downloading, accessing, or using this dataset, you acknowledge that you have read, understood, and agree to be bound by all the terms and conditions of this agreement.

### 📜 Core License

The Dataset is released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International ([CC BY-NC-ND 4.0](https://spdx.org/licenses/CC-BY-NC-ND-4.0)) License**.

This means you are free to copy and redistribute the material in any medium or format under the following terms:

* **ATTRIBUTION** — You must give appropriate credit by citing our original research paper, provide a link to the license, and indicate if any changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

* **NON-COMMERCIAL** — You may not use the Dataset for commercial purposes. This includes any use where the primary purpose is for commercial advantage or monetary compensation. The Dataset is intended for academic and research use only.

* **NO DERIVATIVES** — If you remix, transform, or build upon the material, you may not distribute the modified material. You are permitted to share and redistribute the Dataset only in its original, unmodified form.

### 🔗 Disclaimer Regarding Third-Party Content

This Dataset does not host or distribute any copyrighted video or image files. The Dataset consists solely of metadata (such as annotations, descriptions, and question-answer pairs) and publicly available hyperlinks to the original content, which remains on third-party platforms.

We do not claim ownership of any linked media. All rights to the original visual content belong to their respective copyright holders. Users are solely responsible for adhering to the terms of service, copyright policies, and licensing agreements of the source platforms when accessing or using the linked content.

### ⚖️ No Warranty and Limitation of Liability

> THE DATASET IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
>
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE DATASET.

### ⚠️ General Disclaimer

You are solely responsible for any legal liability arising from your improper use of the Dataset. We reserve the right to terminate your access to the Dataset at any time if you fail to comply with these terms.
