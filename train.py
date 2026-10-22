#!/usr/bin/env python3
"""Reference training loop for the high-recall DenseNet121 model.

The defaults reflect the engineering choices described in the README:
weighted cross-entropy (2.5x on malignant), MONAI augmentation, AMP on CUDA,
validation-based checkpoint selection, and a TrainingTimer for epoch timing.

The released checkpoint was produced by the project's original experiment.
This script is a reproducible reference, not a claim that the exact reported
metrics will be reproduced without the original split and configuration.

Research use only. Not a medical device.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from torch.utils.data import DataLoader
from tqdm import tqdm

from mammo_data import (
    MammogramDataset,
    build_eval_transform,
    build_samples_from_folders,
    build_train_transform,
    check_missing_files,
)
from mammo_metrics import compute_metrics, format_ratio, json_safe


class TrainingTimer:
    """Track wall-clock time per epoch, including CUDA synchronization."""

    def __init__(self) -> None:
        self.epoch_times: list[float] = []
        self._started_at: float | None = None

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._started_at = time.perf_counter()

    def stop(self) -> float:
        if self._started_at is None:
            raise RuntimeError("TrainingTimer.stop() called before start()")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._started_at
        self.epoch_times.append(elapsed)
        self._started_at = None
        return elapsed

    @property
    def mean(self) -> float:
        return sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MONAI DenseNet121 for high-recall mammography classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python train.py --train-dir data/train --val-dir data/val\n"
            "  python train.py --epochs 20 --batch-size 8 --malignant-weight 2.5\n"
        ),
    )
    parser.add_argument("--train-dir", default="data/train", help="Training class-folder root.")
    parser.add_argument("--val-dir", default="data/val", help="Validation class-folder root.")
    parser.add_argument("--output-dir", default="outputs/train", help="Checkpoint/history directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--malignant-weight",
        type=float,
        default=2.5,
        help="CrossEntropyLoss weight for the malignant class (default: 2.5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Operating threshold used for validation metrics (default: 0.35).",
    )
    parser.add_argument(
        "--monitor",
        choices=("recall", "auc", "loss"),
        default="recall",
        help="Validation quantity used for checkpoint selection (default: recall).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience in epochs; 0 disables early stopping.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient-norm clip value; 0 disables clipping.",
    )
    parser.add_argument("--size", type=int, default=512, help="Square input resolution.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply NormalizeIntensity after ScaleIntensity and augmentation.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize DenseNet121 with ImageNet weights when available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.malignant_weight <= 0:
        parser.error("--malignant-weight must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.patience < 0:
        parser.error("--patience cannot be negative")
    if args.grad_clip < 0:
        parser.error("--grad-clip cannot be negative")

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        # Compatibility with older PyTorch releases.
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0

    progress = tqdm(loader, desc="Train", unit="batch", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
        progress.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}")

    return running_loss / seen if seen else float("nan")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> tuple[float, dict[str, float | int]]:
    model.eval()
    running_loss = 0.0
    seen = 0
    probabilities: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc="Val", unit="batch", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        batch_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
        probabilities.append(batch_probabilities.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    y_prob = np.concatenate(probabilities).astype(np.float64)
    y_true = np.concatenate(labels_list).astype(np.int64)
    metrics = compute_metrics(y_true, y_prob, threshold)
    return running_loss / max(seen, 1), metrics


def monitor_score(
    metrics: dict[str, float | int],
    val_loss: float,
    monitor: str,
) -> tuple[float, ...]:
    recall = float(metrics["malignant_recall"])
    auc = float(metrics["roc_auc"])
    recall = recall if math.isfinite(recall) else -math.inf
    auc = auc if math.isfinite(auc) else -math.inf

    if monitor == "loss":
        return (-val_loss,)
    if monitor == "auc":
        return (auc,)
    # Recall first; AUC then breaks ties between thresholds that catch the same
    # number of cancers. Lower validation loss breaks a remaining tie.
    return (recall, auc, -val_loss)


def check_split(name: str, samples: list[tuple[Path, int]]) -> None:
    labels = {label for _, label in samples}
    if labels != {0, 1}:
        raise ValueError(
            f"The {name} split must contain both benign (0) and malignant (1) images. "
            f"Found labels: {sorted(labels)}"
        )


def run_training(args: argparse.Namespace) -> None:
    train_samples = build_samples_from_folders(args.train_dir)
    val_samples = build_samples_from_folders(args.val_dir)
    check_missing_files(train_samples)
    check_missing_files(val_samples)
    check_split("training", train_samples)
    check_split("validation", val_samples)

    train_paths = {path.resolve() for path, _ in train_samples}
    val_paths = {path.resolve() for path, _ in val_samples}
    overlap = train_paths & val_paths
    if overlap:
        raise ValueError(
            f"{len(overlap)} image(s) appear in both train and validation splits. "
            "Create a patient-level, non-overlapping split before training."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    train_dataset = MammogramDataset(
        train_samples,
        build_train_transform(args.size, args.normalize),
    )
    val_dataset = MammogramDataset(
        val_samples,
        build_eval_transform(args.size, args.normalize),
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = DenseNet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        pretrained=args.pretrained,
    )
    model.to(device)

    class_weights = torch.tensor(
        [1.0, args.malignant_weight],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(use_amp)
    timer = TrainingTimer()

    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device:     {device} (AMP: {'on' if use_amp else 'off'})")
    print(f"Model:      MONAI DenseNet121 ({n_parameters:,} parameters)")
    print(f"Train:      {len(train_samples)} images")
    print(f"Validation: {len(val_samples)} images")
    print(f"Threshold:  {args.threshold:.2f}")
    print(f"Monitor:    {args.monitor}")
    print(f"Class weights: benign=1.0, malignant={args.malignant_weight}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.json"

    history: list[dict] = []
    best_score: tuple[float, ...] | None = None
    best_epoch = 0
    best_metrics: dict[str, float | int] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        timer.start()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            args.grad_clip,
        )
        val_loss, metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            args.threshold,
        )
        elapsed = timer.stop()

        auc = float(metrics["roc_auc"])
        auc_text = f"{auc:.3f}" if math.isfinite(auc) else "n/a"
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"recall {format_ratio(metrics['malignant_recall'])} | "
            f"AUC {auc_text} | "
            f"benign precision {format_ratio(metrics['benign_precision'])} | "
            f"{elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": metrics,
                "seconds": elapsed,
            }
        )
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(history), handle, indent=2)

        score = monitor_score(metrics, val_loss, args.monitor)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": metrics,
                    "threshold": args.threshold,
                    "monitor": args.monitor,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  Saved new best checkpoint at epoch {epoch}: {best_path}")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: no improvement for "
                f"{args.patience} epoch(s)."
            )
            break

    if best_metrics is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {best_path.resolve()}")
    print(f"Best validation recall: {format_ratio(best_metrics['malignant_recall'])}")
    print(f"Best validation ROC-AUC: {float(best_metrics['roc_auc']):.4f}")
    print(f"Best validation benign precision: {format_ratio(best_metrics['benign_precision'])}")
    print(f"Mean epoch time: {timer.mean:.1f}s")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_training(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
