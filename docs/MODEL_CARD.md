# Model Card: DenseNet121 Mammography Classifier

## Model details

| Item | Value |
| :--- | :--- |
| Model name | `densenet121_breast_cancer_high_recall` |
| Version | 1.0 |
| Last updated | 2026-10-22 |
| Architecture | MONAI `DenseNet121`, 2D |
| Input | 1 channel, 512 x 512 mammogram ROI |
| Output | 2-class softmax: benign, malignant |
| Parameters | 6,949,634 |
| Checkpoint | `densenet121_breast_cancer_high_recall.pth` |
| Operating threshold | 0.35 malignant probability |
| Framework | PyTorch + MONAI |

## Intended use

Research and educational experiments in high-sensitivity breast lesion classification on cropped mammography regions of interest. The model is intended to support retrospective analysis, threshold studies, and domain-shift experiments.

## Out-of-scope use

- Clinical diagnosis, screening, triage, or treatment planning.
- Autonomous interpretation of mammograms.
- Full-field mammography without domain adaptation and re-calibration.
- Populations, scanners, or acquisition protocols not represented in the training data.
- Any use without qualified clinical oversight and regulatory review.

## Training data and methodology

The model was trained on combined mass and calcification cases from the public **CBIS-DDSM** dataset. Inputs are single-channel region-of-interest patches resized to 512 x 512.

Documented training choices:

- Weighted `CrossEntropyLoss` with a **2.5x** malignant-class weight.
- Mixed precision (AMP) on CUDA.
- MONAI augmentation: `RandFlip`, `RandRotate`, and contrast adjustment.
- Validation-based checkpoint selection before overfitting/saturation.
- Custom `TrainingTimer` for epoch wall-clock timing, including data loading and validation.
- Decision threshold set to **0.35** to favor sensitivity.

The original training notebook, exact split, and full schedule are not included in this release. The reference `train.py` reproduces the documented methodology but may not reproduce the reported metrics exactly.

## Evaluation

Reported results on the project's retrospective CBIS-DDSM test split:

| Metric | Value |
| :--- | :--- |
| ROC-AUC | 0.77 |
| Malignant recall (sensitivity) | 88% (155 / 177) |
| Benign precision (NPV) | 83% |
| Operating threshold | 0.35 |

Definitions:

- Malignant recall = `TP / (TP + FN)`.
- Benign precision = `TN / (TN + FN)`.
- ROC-AUC is threshold-independent.

These are retrospective, single-dataset results. They are not evidence of prospective clinical performance, and no external multi-site validation is claimed.

## Domain shift

The model was stress-tested on **MIAS** full-field mammograms. CBIS-DDSM contains cropped ROI patches, while MIAS contains full-field images with different acquisition, resolution, and contrast characteristics. CLAHE preprocessing was used in the MIAS experiments, but the settings are not directly comparable. Performance degradation under this shift is expected; site-specific re-calibration and external validation are required.

## Limitations and risks

- **False positives:** High recall is achieved by accepting more false positives, which increases follow-up burden.
- **Calibration:** A 0.35 probability threshold is dataset-specific and does not transfer automatically.
- **Leakage risk:** Image-level splits can leak patient identity across views; patient-level splitting is required.
- **Bias:** Public datasets may not represent current scanners, populations, breast densities, or acquisition protocols.
- **Annotation quality:** Lesion labels and ROI boundaries inherit errors from the source dataset.
- **Moderate discrimination:** ROC-AUC 0.77 leaves substantial room for error.
- **Statistical uncertainty:** No confidence intervals are reported; 177 malignant cases leave meaningful uncertainty around the recall estimate.
- **No clinical evidence:** No prospective trial, reader study, or regulatory clearance exists.

## Ethical and legal notes

This model is **not a medical device**. It must not be used for clinical diagnosis or screening decisions. Any clinical research use requires ethics approval, data-use compliance, human oversight, and prospective validation. CBIS-DDSM and MIAS remain subject to their own dataset licenses and terms.

## Reproduction

```bash
python test_eval.py --data-dir data/test
```

See the main [`README.md`](../README.md) for setup, data layout, training, and threshold-sweep details.
