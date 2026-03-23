"""
Dataset loader for pattern piece images.

Loads labeled pattern piece images for training the CNN classifier.
Supports directory-based label structure:

    data/
    ├── pants/
    │   ├── front/
    │   │   ├── img001.jpg
    │   │   └── img002.png
    │   └── back/
    │       └── img003.jpg
    ├── dress/
    │   ├── front/
    │   └── back/
    └── ...

Each image is labeled with:
    - garment_type: from top-level directory name
    - piece_name: from subdirectory name

Additional labels (fold, grain, notch/dart counts) come from an optional
annotations JSON file.

Requires: torch, torchvision, Pillow
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
    from PIL import Image

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .classifier import (
    GARMENT_TYPES,
    PIECE_NAMES,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PatternDataset requires torch and torchvision. "
            "Install with: pip install torch torchvision"
        )


# ── Annotation schema ───────────────────────────────────────────────────────

DEFAULT_ANNOTATION = {
    "has_fold_line": False,
    "has_grain_line": True,
    "notch_count": 0,
    "dart_count": 0,
}


def load_annotations(path: str | Path) -> dict[str, dict]:
    """
    Load per-image annotations from a JSON file.

    Expected format:
        {
            "img001.jpg": {
                "has_fold_line": true,
                "has_grain_line": true,
                "notch_count": 3,
                "dart_count": 1
            },
            ...
        }

    Args:
        path: Path to annotations JSON file.

    Returns:
        Dict mapping image filename to annotation dict.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class PatternDataset(Dataset if HAS_TORCH else object):
    """
    PyTorch Dataset for pattern piece images.

    Scans a directory tree for images organized by garment_type/piece_name.
    Returns tensors and multi-task label dicts for training the classifier.

    Args:
        root_dir: Path to data root (contains garment_type subdirs).
        annotations_file: Optional path to JSON annotations for fold/grain/notch/dart.
        augment: Whether to apply data augmentation (for training).
    """

    def __init__(
        self,
        root_dir: str | Path,
        annotations_file: str | Path | None = None,
        augment: bool = False,
    ):
        _require_torch()
        self.root_dir = Path(root_dir)

        # Load optional annotations
        self.annotations = {}
        if annotations_file:
            self.annotations = load_annotations(annotations_file)

        # Build sample list: (image_path, garment_type, piece_name)
        self.samples: list[tuple[Path, str, str]] = []
        self._scan_directory()

        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def _scan_directory(self) -> None:
        """Walk the directory tree and collect image samples."""
        if not self.root_dir.exists():
            return

        for garment_dir in sorted(self.root_dir.iterdir()):
            if not garment_dir.is_dir():
                continue
            garment_type = garment_dir.name.lower()
            if garment_type not in GARMENT_TYPES:
                continue

            for piece_dir in sorted(garment_dir.iterdir()):
                if not piece_dir.is_dir():
                    continue
                piece_name = piece_dir.name.lower()
                if piece_name not in PIECE_NAMES:
                    continue

                for img_path in sorted(piece_dir.iterdir()):
                    if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                        self.samples.append((img_path, garment_type, piece_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            (image_tensor, labels_dict)

            labels_dict keys:
                garment_type: int (index into GARMENT_TYPES)
                piece_name: int (index into PIECE_NAMES)
                has_fold_line: float (0.0 or 1.0)
                has_grain_line: float (0.0 or 1.0)
                notch_count: float
                dart_count: float
        """
        img_path, garment_type, piece_name = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)

        # Labels
        garment_idx = GARMENT_TYPES.index(garment_type)
        piece_idx = PIECE_NAMES.index(piece_name)

        # Check for per-image annotations
        ann = self.annotations.get(img_path.name, DEFAULT_ANNOTATION)

        labels = {
            "garment_type": garment_idx,
            "piece_name": piece_idx,
            "has_fold_line": float(ann.get("has_fold_line", False)),
            "has_grain_line": float(ann.get("has_grain_line", True)),
            "notch_count": float(ann.get("notch_count", 0)),
            "dart_count": float(ann.get("dart_count", 0)),
        }

        return tensor, labels

    def summary(self) -> dict:
        """Return a summary of the dataset."""
        garment_counts: dict[str, int] = {}
        piece_counts: dict[str, int] = {}
        for _, g, p in self.samples:
            garment_counts[g] = garment_counts.get(g, 0) + 1
            piece_counts[p] = piece_counts.get(p, 0) + 1

        return {
            "total_images": len(self.samples),
            "garment_types": garment_counts,
            "piece_names": piece_counts,
            "has_annotations": len(self.annotations) > 0,
        }
