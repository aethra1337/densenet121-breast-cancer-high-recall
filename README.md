# Breast Cancer Detection on Mammography with DenseNet121 + MONAI

A research-oriented, high-recall binary classification pipeline for breast lesions on mammography, built with **PyTorch**, **MONAI**, and **DenseNet121**.

> [!WARNING]
> This project is for research and educational use only. It is **not** a medical device and must **not** be used for clinical diagnosis, screening, or treatment decisions. Any clinical use requires prospective validation, regulatory review, and oversight by qualified clinicians.

---

## Overview

Breast cancer screening is a high-cost error domain: a false negative can delay diagnosis, while a false positive creates unnecessary anxiety and follow-up procedures. This project therefore optimizes for **malignant recall (sensitivity)** rather than raw accuracy.

The released checkpoint is a **MONAI DenseNet121** model that classifies single-channel mammogram crops as **benign** or **malignant**. It was trained on combined **mass and calcification** cases from the public **CBIS-DDSM** dataset, with a decision threshold tuned toward high sensitivity.

| Item | Detail |
| :--- | :--- |
| Task | Binary classification: benign vs. malignant |
| Primary dataset | CBIS-DDSM (mass + calcification, combined) |
| Input | 512 x 512, single-channel mammogram ROI |
| Architecture | MONAI `DenseNet121`, 2D, 1 input channel, 2-class head |
| Parameters | 6.95M |
| Training precision | Mixed precision (AMP) |
| Supporting tools | scikit-learn, pandas, OpenCV (CLAHE experiments) |
| Operating threshold | 0.35 malignant probability |
| Checkpoint | `densenet121_breast_cancer_high_recall.pth` |

OpenCV is only needed for the MIAS/CLAHE experiments; the released evaluation and training scripts do not import it.

---

## Results

Reported test-set performance for the released checkpoint:

| Metric | Score |
| :--- | :--- |
| ROC-AUC | **0.77** |
| Malignant recall (sensitivity) | **88%** (155 / 177 malignant cases) |
| Benign precision | **83%** |
| Operating threshold | **0.35** |

**How the headline metrics are defined**

- **Malignant recall** = `TP / (TP + FN)` — the share of malignant cases that were correctly flagged.
- **Benign precision** = `TN / (TN + FN)` — among cases predicted as benign, the share that were truly benign. In a two-class setting this is also the negative predictive value (NPV).
- **ROC-AUC** is threshold-independent and measures ranking quality across all thresholds.
- The **0.35 threshold** is the high-recall operating point used for the reported results. It should be re-calibrated on a site-specific validation set before any real-world use.

An AUC of 0.77 indicates moderate discrimination. The high recall at 0.35 is achieved by accepting more false positives — an intentional trade-off for a screening-oriented workflow, not a claim of clinical-grade accuracy.

---

## Technical Highlights

### 1. High-recall optimization

- Weighted `CrossEntropyLoss` with a **2.5x penalty** on the malignant class.
- Decision threshold lowered to **0.35**, prioritizing sensitivity over generic classification accuracy.
- Evaluation explicitly reports recall, benign precision, specificity, PPV/NPV, and a full threshold sweep instead of accuracy alone.

### 2. Overfitting control and checkpoint selection

- Validation metrics monitored across fine-tuning epochs.
- The best checkpoint is retained based on validation performance before saturation/overfitting.
- A custom `TrainingTimer` tracked epoch duration, including data loading and validation time.

### 3. MONAI preprocessing and augmentation

- MONAI transforms for loading, channel handling, resizing, and intensity scaling (GPU-capable when applied to CUDA tensors).
- Training augmentations include `RandFlip`, `RandRotate`, and contrast adjustment.
- Evaluation uses deterministic resizing and intensity scaling; optional zero-mean normalization is available via `--normalize` for pipelines that used it during training.

### 4. Domain-shift analysis on MIAS

The model was additionally stress-tested on the **MIAS** dataset to study cross-dataset generalization. CBIS-DDSM contains **cropped region-of-interest (ROI) patches**, while MIAS contains **full-field mammograms**. That structural difference — plus differences in acquisition, resolution, and contrast — creates a substantial domain shift. MIAS experiments used **OpenCV-based CLAHE** preprocessing to improve local contrast, but the two settings are not directly comparable. External performance should be expected to degrade without domain adaptation and re-calibration.

---

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── test_eval.py
├── train.py
├── mammo_data.py
├── mammo_metrics.py
├── densenet121_breast_cancer_high_recall.pth
├── LICENSE
└── docs/
    └── MODEL_CARD.md
```

The dataset and test split are **not** included. Download CBIS-DDSM separately and prepare the layout described below.

This release contains the trained checkpoint, a reference training loop, and the evaluation path. The original training notebook/configuration and the exact CBIS-DDSM split are not included, so the reported metrics may not be reproduced exactly without those experiment records.

---

## Setup

Python 3.10–3.12 is recommended for the widest PyTorch/MONAI compatibility. On Windows, if `python` points to a newer version, create the environment explicitly with `py -3.12 -m venv .venv`.

The released scripts were smoke-tested with Python 3.12, PyTorch 2.14.1, and MONAI 1.6.1 on CPU. GPU/AMP behavior follows the standard PyTorch CUDA path.

```bash
git clone https://github.com/aethra1337/densenet121-breast-cancer-high-recall.git
cd densenet121-breast-cancer-high-recall
```

Create an environment and install dependencies:

```bash
# Windows (adjust the version if needed)
py -3.12 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU training or evaluation, install the PyTorch build that matches your CUDA driver first, following the official selector at [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining requirements.

---

## Data Layout

### Option A — class folders

```text
data/
├── train/
│   ├── benign/
│   └── malignant/
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

Each class folder may contain nested subdirectories. Images should be patient-disjoint across the three splits; do not place different views of the same patient in both train and test.

Supported formats include PNG, JPEG, BMP, TIFF, and DICOM (`.dcm`). Images are resized to 512 x 512 and scaled to `[0, 1]` by default.

### Option B — CSV manifest (evaluation)

```csv
image,label
images/case_0001.png,benign
images/case_0002.png,malignant
images/case_0003.png,BENIGN_WITHOUT_CALLBACK
```

Labels may be `benign` / `malignant`, `0` / `1`, or CBIS-DDSM-style strings such as `BENIGN_WITHOUT_CALLBACK`. The parser also accepts `normal` as benign.

---

## Training

`train.py` is a reference implementation of the training methodology described above: weighted cross-entropy with a 2.5x malignant weight, MONAI augmentation, AMP on CUDA, validation-based checkpoint selection, early stopping, and a `TrainingTimer` for epoch timing.

```bash
python train.py \
  --train-dir data/train \
  --val-dir data/val \
  --output-dir outputs/train
```

The script saves:

```text
outputs/train/
├── best_model.pth   # selected by validation recall (default), AUC, or loss
└── history.json     # per-epoch losses, metrics, and wall-clock time
```

`best_model.pth` is directly compatible with `test_eval.py`:

```bash
python test_eval.py \
  --checkpoint outputs/train/best_model.pth \
  --data-dir data/test
```

### Training options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--train-dir` | `data/train` | Training class-folder root |
| `--val-dir` | `data/val` | Validation class-folder root |
| `--output-dir` | `outputs/train` | Checkpoint and history directory |
| `--epochs` | `20` | Maximum training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--malignant-weight` | `2.5` | Class weight for malignant in cross-entropy |
| `--threshold` | `0.35` | Threshold used for validation metrics |
| `--monitor` | `recall` | Checkpoint criterion: `recall`, `auc`, or `loss` |
| `--patience` | `5` | Early-stopping patience; `0` disables |
| `--grad-clip` | `1.0` | Gradient-norm clipping; `0` disables |
| `--pretrained` | off | Initialize with ImageNet DenseNet121 weights |
| `--normalize` | off | Add zero-mean intensity normalization |
| `--seed` | `42` | Random seed |

The reference loop is intentionally conservative. It does not claim to reproduce the released checkpoint's exact metrics without the original data split, augmentation probabilities, and training schedule. It keeps preprocessing/augmentation in DataLoader workers and uses CUDA AMP for the model; move the transforms onto CUDA tensors if you specifically want GPU-side preprocessing.

---

## Run Evaluation

### Class-folder layout

```bash
python test_eval.py --data-dir data/test
```

### CSV manifest

```bash
python test_eval.py \
  --manifest data/test.csv \
  --image-dir data/images \
  --image-col image \
  --label-col label
```

### Useful options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `densenet121_breast_cancer_high_recall.pth` | Model weights |
| `--threshold` | `0.35` | Malignant-probability cutoff |
| `--target-recall` | none | Report the highest threshold meeting this recall target |
| `--normalize` | off | Add `NormalizeIntensity(nonzero=True, channel_wise=True)` after `ScaleIntensity` |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--batch-size` | `8` | Evaluation batch size |
| `--num-workers` | `4` | DataLoader workers; use `0` if your platform has issues |
| `--limit` | none | Evaluate only the first N images (smoke test) |
| `--output-dir` | `outputs/eval` | Directory for metrics and predictions |

Example with a recall target and a CPU-only run:

```bash
python test_eval.py --data-dir data/test --device cpu --target-recall 0.90
```

### Outputs

The evaluator writes:

```text
outputs/eval/
├── metrics.json          # metrics, confusion matrix, run configuration
├── predictions.csv       # per-image label, malignant probability, prediction
└── threshold_sweep.csv   # recall/precision trade-off across thresholds
```

A console summary includes ROC-AUC, PR-AUC, malignant recall, benign precision, specificity, PPV, NPV, balanced accuracy, and the confusion matrix. Threshold sweep results are for analysis only: **choose the operating threshold on a validation split, never on the test set.**

---

## Interpreting the Threshold

At a low threshold, more cases cross into "malignant" and recall rises, but false positives and radiologist workload also rise. At a high threshold, benign precision improves but cancers are missed. The 0.35 operating point reflects the project's screening-oriented preference for sensitivity; it is not a universal clinical recommendation.

The sweep output makes this trade-off explicit:

```text
threshold  malignant_recall  benign_precision  specificity  ...
0.20       ...
0.35       ...
0.50       ...
```

Use it to select a threshold that matches a prospective clinical workflow and its acceptable false-positive rate.

---

## Limitations

- **Retrospective public data.** Results come from a single public dataset and are not evidence of prospective clinical performance.
- **ROI vs. full-field shift.** CBIS-DDSM ROI patches and MIAS full mammograms differ structurally; cross-dataset results are a stress test, not a like-for-like benchmark.
- **Calibration is dataset-specific.** A probability of 0.35 does not transfer automatically to another scanner, population, or preprocessing pipeline.
- **Moderate discrimination.** ROC-AUC 0.77 leaves substantial room for error; high recall is purchased with false positives.
- **No confidence intervals.** With 177 malignant cases, the recall estimate carries meaningful statistical uncertainty.
- **No clinical deployment.** The repository does not include regulatory clearance, prospective trials, or radiologist-in-the-loop validation.
- **Preprocessing must match training.** If your training pipeline used zero-mean normalization, pass `--normalize`; otherwise the checkpoint will see a different intensity distribution.

---

## Suggested Next Steps

1. Validate on an external, multi-site mammography cohort.
2. Report patient-level (not image-level) splits to avoid leakage between views of the same patient.
3. Calibrate probabilities (e.g., temperature scaling) before threshold selection.
4. Evaluate subgroup performance across age, breast density, and acquisition device.
5. Add explainability (Grad-CAM) and reader-study evaluation before any clinical pilot.

---

## Acknowledgements

- **CBIS-DDSM** — Curated Breast Imaging Subset of DDSM, publicly available via TCIA.
- **MONAI** — medical imaging transforms and network implementations.
- **PyTorch** — deep learning framework.
- **DenseNet** — Huang et al., *Densely Connected Convolutional Networks*.

A clinical-safety-focused model card is available in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## License

The source code in this repository is released under the [MIT License](LICENSE). The CBIS-DDSM and MIAS datasets remain subject to their own licenses and terms of use. The checkpoint is provided for research use only and is **not** a medical device.
