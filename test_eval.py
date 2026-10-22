#!/usr/bin/env python3
"""Evaluate the high-recall DenseNet121 mammography checkpoint.

The script reports malignant recall, benign precision, ROC-AUC, specificity,
PPV/NPV, a confusion matrix, and a threshold sweep. It supports either a
class-folder test set or a CSV manifest.

Research use only. Not a medical device.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from monai.networks.nets import DenseNet121
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from mammo_data import (
    MammogramDataset,
    build_eval_transform,
    build_samples_from_folders,
    build_samples_from_manifest,
    check_missing_files,
)
from mammo_metrics import (
    SCALAR_METRICS,
    compute_metrics,
    format_ratio,
    json_safe,
)

DEFAULT_CHECKPOINT = Path(__file__).resolve().with_name(
    "densenet121_breast_cancer_high_recall.pth"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MONAI DenseNet121 checkpoint for high-recall benign vs "
            "malignant mammography classification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_eval.py --data-dir data/test\n"
            "  python test_eval.py --manifest data/test.csv --image-dir data/images\n"
            "  python test_eval.py --data-dir data/test --device cpu --target-recall 0.90\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the .pth checkpoint (default: the checkpoint beside this script).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/test",
        help="Root containing benign/ and malignant/ folders (default: data/test).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional CSV manifest. If supplied, --data-dir is not used.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Base directory for relative image paths in --manifest.",
    )
    parser.add_argument(
        "--image-col",
        default="image",
        help="Image-path column name in --manifest (default: image).",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Label column name in --manifest (default: label).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Malignant-probability cutoff (default: 0.35).",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Report the highest threshold in the sweep meeting this recall target.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Square input resolution (default: 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers; use 0 if your platform has issues (default: 4).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or mps (default: auto).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed precision.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Add NormalizeIntensity(nonzero=True, channel_wise=True) after "
            "ScaleIntensity. Use this if training used zero-mean normalization."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N images (smoke test).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval",
        help="Directory for metrics.json, predictions.csv, and threshold_sweep.csv.",
    )

    args = parser.parse_args(argv)

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.target_recall is not None and not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in the interval (0, 1]")
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    return args


def resolve_device(name: str) -> torch.device:
    name = name.strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return device


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 2.0 does not accept weights_only.
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[warn] weights_only load failed ({exc}); retrying with weights_only=False")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and any(
                torch.is_tensor(value) for value in candidate.values()
            ):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format in {path}: expected a state_dict, "
            f"got {type(checkpoint).__name__}"
        )

    state: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if isinstance(key, str) and torch.is_tensor(value):
            state[_strip_prefix(key)] = value

    if not state:
        raise ValueError(f"No tensor parameters found in checkpoint: {path}")
    return state


def build_model() -> DenseNet121:
    return DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)


# ---------------------------------------------------------------------------
# Metrics and reporting
# ---------------------------------------------------------------------------


def build_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fixed_threshold: float,
) -> pd.DataFrame:
    thresholds = {
        round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05)
    }
    thresholds.add(round(float(fixed_threshold), 2))

    rows = []
    for threshold in sorted(thresholds):
        metrics = compute_metrics(y_true, y_prob, threshold)
        rows.append(
            {
                "threshold": threshold,
                "malignant_recall": metrics["malignant_recall"],
                "benign_precision": metrics["benign_precision"],
                "specificity": metrics["specificity"],
                "malignant_precision": metrics["malignant_precision"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
            }
        )
    return pd.DataFrame(rows)


def print_summary(
    metrics: dict[str, float | int],
    threshold: float,
    n_benign: int,
    n_malignant: int,
) -> None:
    roc_auc = float(metrics["roc_auc"])
    pr_auc = float(metrics["pr_auc"])
    roc_text = f"{roc_auc:.4f}" if math.isfinite(roc_auc) else "n/a"
    pr_text = f"{pr_auc:.4f}" if math.isfinite(pr_auc) else "n/a"

    print()
    print("Evaluation summary")
    print("------------------")
    print(f"Images:                    {n_benign + n_malignant} "
          f"({n_benign} benign, {n_malignant} malignant)")
    print(f"Operating threshold:       {threshold:.2f}")
    print(f"ROC-AUC:                   {roc_text}")
    print(f"PR-AUC (malignant):        {pr_text}")
    print(f"Malignant recall:          {format_ratio(metrics['malignant_recall'])} "
          f"({metrics['tp']}/{metrics['tp'] + metrics['fn']})")
    print(f"Benign precision (NPV):    {format_ratio(metrics['benign_precision'])} "
          f"({metrics['tn']}/{metrics['tn'] + metrics['fn']})")
    print(f"Specificity:               {format_ratio(metrics['specificity'])}")
    print(f"Malignant precision (PPV): {format_ratio(metrics['malignant_precision'])}")
    print(f"Accuracy:                  {format_ratio(metrics['accuracy'])}")
    print(f"Balanced accuracy:         {format_ratio(metrics['balanced_accuracy'])}")
    print()
    print("Confusion matrix (rows = truth, columns = prediction)")
    print("                     pred benign   pred malignant")
    print(f"  true benign        {metrics['tn']:>10}   {metrics['fp']:>14}")
    print(f"  true malignant     {metrics['fn']:>10}   {metrics['tp']:>14}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)

    if args.manifest:
        samples = build_samples_from_manifest(
            args.manifest,
            args.image_dir,
            args.image_col,
            args.label_col,
        )
    else:
        samples = build_samples_from_folders(args.data_dir)

    if args.limit is not None:
        samples = samples[: args.limit]

    check_missing_files(samples)

    n_benign = sum(1 for _, label in samples if label == 0)
    n_malignant = sum(1 for _, label in samples if label == 1)

    use_amp = device.type == "cuda" and not args.no_amp

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device} (AMP: {'on' if use_amp else 'off'})")
    print(f"Images:     {len(samples)} ({n_benign} benign, {n_malignant} malignant)")
    print(f"Threshold:  {args.threshold:.2f}")
    print(f"Preprocess: ScaleIntensity{' + NormalizeIntensity' if args.normalize else ''}")

    dataset = MammogramDataset(
        samples,
        build_eval_transform(args.size, args.normalize),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = build_model()
    state_dict = load_state_dict(checkpoint_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint does not match MONAI DenseNet121("
            "spatial_dims=2, in_channels=1, out_channels=2). "
            "Make sure you are using the released state_dict."
        ) from exc

    model.to(device)
    model.eval()
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model:      MONAI DenseNet121 ({n_parameters:,} parameters)")

    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc="Evaluating", unit="batch"):
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)

            batch_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            probabilities.append(batch_probabilities.cpu().numpy())
            labels.append(batch_labels.numpy())

    y_prob = np.concatenate(probabilities).astype(np.float64)
    y_true = np.concatenate(labels).astype(np.int64)

    metrics = compute_metrics(y_true, y_prob, args.threshold)
    print_summary(metrics, args.threshold, n_benign, n_malignant)

    print()
    print("Per-class report")
    print("----------------")
    print(
        classification_report(
            y_true,
            (y_prob >= args.threshold).astype(np.int64),
            labels=[0, 1],
            target_names=["benign", "malignant"],
            digits=4,
            zero_division=0,
        )
    )

    sweep = build_sweep(y_true, y_prob, args.threshold)
    print("Threshold sweep")
    print("---------------")
    display = sweep.copy()
    for column in (
        "malignant_recall",
        "benign_precision",
        "specificity",
        "malignant_precision",
        "accuracy",
        "balanced_accuracy",
    ):
        display[column] = display[column].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:.3f}"
        )
    display["threshold"] = display["threshold"].map(lambda value: f"{value:.2f}")
    print(display.to_string(index=False))

    selected_threshold: float | None = None
    if args.target_recall is not None:
        eligible = sweep[sweep["malignant_recall"] >= args.target_recall]
        if eligible.empty:
            print(
                f"\nNo threshold in the sweep reaches recall >= "
                f"{args.target_recall:.2f}."
            )
        else:
            # Prefer the highest threshold that still meets the recall target:
            # it keeps the most benign cases out of the malignant pathway.
            row = eligible.sort_values("threshold").iloc[-1]
            selected_threshold = float(row["threshold"])
            benign_text = (
                "n/a"
                if pd.isna(row["benign_precision"])
                else f"{row['benign_precision']:.3f}"
            )
            print(
                f"\nHighest threshold with recall >= {args.target_recall:.2f}: "
                f"{selected_threshold:.2f} "
                f"(recall={row['malignant_recall']:.3f}, "
                f"benign precision={benign_text})"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted = (y_prob >= args.threshold).astype(np.int64)
    predictions = pd.DataFrame(
        {
            "path": [str(path) for path, _ in samples],
            "label": y_true,
            "prob_malignant": y_prob,
            "predicted": predicted,
            "correct": (y_true == predicted).astype(int),
        }
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)

    payload = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "amp": use_amp,
        "preprocessing": {
            "size": args.size,
            "intensity": "scale+normalize" if args.normalize else "scale",
        },
        "threshold": args.threshold,
        "target_recall": args.target_recall,
        "selected_threshold": selected_threshold,
        "n_samples": int(len(y_true)),
        "n_benign": n_benign,
        "n_malignant": n_malignant,
        "metrics": {name: metrics[name] for name in SCALAR_METRICS},
        "confusion_matrix": {
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)

    print(f"\nSaved outputs to {output_dir.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_evaluation(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
