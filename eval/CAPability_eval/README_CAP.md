# CAPability Caption Evaluation

This repository provides an **end-to-end pipeline** for evaluating image-captioning models on **CAPability** .

---

## 📂 Directory Layout

```text
CAPability_eval/
├── anno/                         # GT files (created in Step 1)
├── anno_video/                   # GT for video captions (optional)
├── evaluation/                   # CAPability metric implementation
│   ├── eval2.py                  # ⬅︎ main evaluation entry
│   ├── OPTIONS.py                # dimension-set for images
│   ├── OPTIONS_video.py          # dimension-set for videos
│   └── prompt2.py                # LLM prompt template
│
├── capability_conda.yml          # Reproducible conda environment
├── eval_run_all.sh               # ⬅︎ Step 3: call evaluation/eval2.py per model
├── process_all.sh                # Step 2: Convert raw outputs → eval input
├── split_annotation_by_metrics.py# Step 1: Generate GT by metric
├── split_inference_by_model.py    # helper for process_all.sh
├── image_description_annotation.json
├── requirements.txt              # Extra Python dependencies
└── README_CAP.md                 # (this file)

```

---

## 🚀 Quick Start

### 0  Environment Setup

```bash
# ➊ Create and activate the conda environment
conda env create -f capability_conda.yml
conda activate capability

# ➋ Install any additional Python packages
pip install -r requirements.txt
```

> **Note**  All subsequent commands assume your working directory is
> `CAPability_eval/`.

### 1  Generate CAPability-Style Annotations
**Place** the `image_description_annotation.json` inside `CAPability_eval/` .
```bash
python split_annotation_by_metrics.py
```

This script slices `image_description_annotation.json` into six JSONL files
(Scale, Angle, Composition, Colors, Lighting, Focal Lengths) under `anno/`.

### 2  Convert Model Outputs → Evaluation Input

1. **Place** the `models_generated_image_description/` folder (containing each model’s raw image descriptions) inside `inference/` (create `inference/` alongside this README if it does not yet exist).
2. Run the helper script:

```bash
bash ./process_all.sh
```

The script processes **only** the model names hard-coded inside the file.
If you add new model directories to `inference/`, remember to update that list before rerunning.
Standardised inputs are written to `inputs/` (created automatically).

---

## 3  Evaluate All Models

```bash
bash ./eval_run_all.sh
```

The script will

1. **Enumerate** every processed model directory under `inference/processed/`.
2. **Invoke** `python evaluation/eval2.py` for each model with the following args

   ```text
   --caption_file_root   inference/processed/<model_name>
   --gt_file_root        anno/                 # image GT
   --gt_file_root_video  anno_video/           # video GT (if any)
   --save_root           evaluation/<model_name>
   ```
3. **Write** result JSON files per model to `evaluation/<model_name>/`

