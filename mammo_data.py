"""Shared data loading and MONAI transforms for mammography experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    NormalizeIntensity,
    RandAdjustContrast,
    RandFlip,
    RandRotate,
    Resize,
    ScaleIntensity,
    Transform,
)
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".dcm",
    ".dicom",
}


class EnsureSingleChannel(Transform):
    """Collapse an RGB/RGBA image to one channel while keeping channels first."""

    def __call__(self, img: Any) -> Any:
        if img.ndim == 4:
            if img.shape[0] == 1 and img.shape[-1] in (3, 4):
                return img.mean(dim=-1)
            raise ValueError(f"Expected a 2D grayscale image, got shape {tuple(img.shape)}")

        if img.ndim != 3:
            raise ValueError(f"Expected a 3D channel-first image, got shape {tuple(img.shape)}")

        if img.shape[0] == 1:
            return img
        if img.shape[0] in (3, 4):
            return img.mean(dim=0, keepdim=True)
        if img.shape[-1] in (3, 4):
            return img.mean(dim=-1).unsqueeze(0)
        raise ValueError(f"Expected a grayscale or RGB image, got shape {tuple(img.shape)}")


def _base_steps(size: int, normalize: bool) -> list[Any]:
    steps: list[Any] = [
        LoadImage(image_only=True),
        EnsureChannelFirst(channel_dim="no_channel"),
        EnsureSingleChannel(),
        Resize(spatial_size=(size, size), mode="bilinear"),
        ScaleIntensity(minv=0.0, maxv=1.0),
    ]
    if normalize:
        steps.append(NormalizeIntensity(nonzero=True, channel_wise=True))
    return steps


def build_eval_transform(size: int, normalize: bool = False) -> Compose:
    """Deterministic preprocessing used for validation, test, and inference."""
    return Compose(_base_steps(size, normalize))


def build_train_transform(size: int, normalize: bool = False) -> Compose:
    """Preprocessing plus the augmentations described in the project README."""
    steps = _base_steps(size, normalize=False)
    steps.extend(
        [
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate(range_x=10.0, prob=0.3, keep_size=True),
            RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
        ]
    )
    if normalize:
        # Apply intensity normalization after augmentation so that the final
        # distribution matches the deterministic evaluation transform.
        steps.append(NormalizeIntensity(nonzero=True, channel_wise=True))
    return Compose(steps)


class MammogramDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: Compose,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        try:
            image = self.transform(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {path}") from exc

        if hasattr(image, "as_tensor"):
            image = image.as_tensor()
        image = torch.as_tensor(image, dtype=torch.float32)

        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] != 1:
            raise RuntimeError(
                f"Expected one channel after preprocessing, got {tuple(image.shape)} for {path}"
            )

        return image, torch.tensor(label, dtype=torch.long)


def iter_images(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def find_class_dir(root: Path, class_name: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    direct = root / class_name
    if direct.is_dir():
        return direct

    for child in root.iterdir():
        if child.is_dir() and child.name.casefold() == class_name:
            return child

    raise FileNotFoundError(
        f"Could not find a '{class_name}' directory under {root}. "
        f"Expected {root / class_name}."
    )


def parse_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)

    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return int(value)

    text = str(value).strip().lower()

    try:
        numeric = float(text)
        if numeric in (0.0, 1.0):
            return int(numeric)
    except ValueError:
        pass

    if text in {"benign", "normal", "negative", "non-malignant", "nonmalignant"}:
        return 0
    if text.startswith("benign"):
        # Includes CBIS-DDSM labels such as BENIGN_WITHOUT_CALLBACK.
        return 0

    if text in {"malignant", "cancer", "positive", "tumor", "tumour"}:
        return 1
    if text.startswith("malignant"):
        return 1

    raise ValueError(f"Unrecognized label {value!r}. Use benign/malignant or 0/1.")


def build_samples_from_folders(data_dir: str | Path) -> list[tuple[Path, int]]:
    root = Path(data_dir)
    samples: list[tuple[Path, int]] = []

    for class_name, label in (("benign", 0), ("malignant", 1)):
        class_dir = find_class_dir(root, class_name)
        paths = list(iter_images(class_dir))
        if not paths:
            print(f"[warn] No supported images found in {class_dir}")
        samples.extend((path, label) for path in paths)

    if not samples:
        raise FileNotFoundError(
            f"No supported images found under {root}. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return samples


def build_samples_from_manifest(
    manifest_path: str | Path,
    image_dir: str | Path | None,
    image_col: str,
    label_col: str,
) -> list[tuple[Path, int]]:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    frame = pd.read_csv(manifest)
    missing_columns = [name for name in (image_col, label_col) if name not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Manifest {manifest} is missing column(s): {', '.join(missing_columns)}. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )

    base = Path(image_dir) if image_dir else manifest.parent
    samples: list[tuple[Path, int]] = []

    for row_index, row in frame.iterrows():
        raw_path = row[image_col]
        if pd.isna(raw_path):
            raise ValueError(f"Missing image path in {manifest}, row {row_index}")

        path = Path(str(raw_path))
        if not path.is_absolute():
            path = base / path

        samples.append((path, parse_label(row[label_col])))

    if not samples:
        raise FileNotFoundError(f"Manifest {manifest} contains no rows")
    return samples


def check_missing_files(samples: list[tuple[Path, int]]) -> None:
    missing = [path for path, _ in samples if not path.is_file()]
    if not missing:
        return

    preview = "\n".join(f"  - {path}" for path in missing[:10])
    suffix = "\n  ..." if len(missing) > 10 else ""
    raise FileNotFoundError(
        f"{len(missing)} image file(s) were not found:\n{preview}{suffix}"
    )
