# CineTechBench – COCO‑Style Caption Evaluation

This repository provides an **end‑to‑end pipeline** for evaluating image‑captioning models on **CineTechBench** using the official MS‑COCO metrics.

---

## 📂 Directory Layout

```text
CineTechBench/
└── eval/
    └── coco_caption_eval/
        ├── create/               # Annotation‑creation utilities
        │   ├── anno_process.py
        │   └── res_proceses.py   # (helper for result‑file post‑processing)
        ├── cocoEval.py           # Thin wrapper on COCOEvalCap
        ├── coco_conda.yml        # Reproducible conda environment
        ├── eval_run_all.sh       # Evaluate every model in one go
        ├── generate_all.sh       # Convert raw outputs → COCO result JSON
        ├── README_coco.md        # (this file)
        └── requirements.txt      # Extra Python dependencies
```

---

## 🚀 Quick Start

### 0  Environment Setup

```bash
# ➊ Create and activate the conda environment
conda env create -f coco_conda.yml
conda activate coco

# ➋ Install any additional Python packages
pip install -r requirements.txt
```

> **Note**  All subsequent commands assume your working directory is
> `CineTechBench/eval/coco_caption_eval`.

### 1  Generate COCO‑Style Annotations

```bash
python create/anno_process.py
```

This script converts your custom descriptions into the standard COCO annotation schema.

### 2  Convert Model Outputs → Result JSON

1. **Place** the `models_generated_image_description/` folder (which contains each model’s raw image descriptions) inside `res/` (create `res/` alongside this README if it does not yet exist).
2. Run the helper script:

```bash
bash ./generate_all.sh
```

The script processes **only** the model names that are hard‑coded inside the script. If you add new model directories to `res/`, remember to update that list before rerunning. Standardised outputs are written to `results/` (automatically created).

### 3  Evaluate All Models

```bash
bash ./eval_run_all.sh
```

For every result JSON, you will see the standard COCO metrics (BLEU‑1‑4, METEOR, ROUGE‑L, CIDEr) in `coco_res/`.

---
