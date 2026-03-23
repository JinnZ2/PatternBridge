"""
Training loop for the pattern piece CNN classifier.

Trains PatternClassifierNet on a labeled dataset of pattern piece images.
Uses multi-task loss combining classification and regression heads.

Usage:
    python -m pattern_vision.train \
        --data-dir data/patterns \
        --output weights/pattern_classifier.pt \
        --epochs 30 \
        --batch-size 16

Requires: torch, torchvision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .classifier import PatternClassifierNet
from .dataset import PatternDataset


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "Training requires torch. Install with: pip install torch torchvision"
        )


# ── Loss ─────────────────────────────────────────────────────────────────────


class MultiTaskLoss(nn.Module if HAS_TORCH else object):
    """
    Combined loss for all classifier heads.

    - garment_type, piece_name: CrossEntropyLoss
    - has_fold_line, has_grain_line: BCEWithLogitsLoss
    - notch_count, dart_count: MSELoss (regression)
    """

    def __init__(self):
        _require_torch()
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        # Loss weights — classification heads weighted higher
        self.weights = {
            "garment_type": 2.0,
            "piece_name": 2.0,
            "has_fold_line": 1.0,
            "has_grain_line": 1.0,
            "notch_count": 0.5,
            "dart_count": 0.5,
        }

    def forward(self, outputs: dict, labels: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute weighted multi-task loss.

        Returns:
            (total_loss, individual_losses_dict)
        """
        losses = {}

        losses["garment_type"] = self.ce_loss(
            outputs["garment_type"], labels["garment_type"]
        )
        losses["piece_name"] = self.ce_loss(
            outputs["piece_name"], labels["piece_name"]
        )
        losses["has_fold_line"] = self.bce_loss(
            outputs["has_fold_line"].squeeze(-1), labels["has_fold_line"]
        )
        losses["has_grain_line"] = self.bce_loss(
            outputs["has_grain_line"].squeeze(-1), labels["has_grain_line"]
        )
        losses["notch_count"] = self.mse_loss(
            outputs["notch_count"].squeeze(-1), labels["notch_count"]
        )
        losses["dart_count"] = self.mse_loss(
            outputs["dart_count"].squeeze(-1), labels["dart_count"]
        )

        total = sum(self.weights[k] * v for k, v in losses.items())
        return total, {k: v.item() for k, v in losses.items()}


# ── Collate ──────────────────────────────────────────────────────────────────


def collate_fn(batch):
    """Custom collate to stack label dicts into tensors."""
    images = torch.stack([b[0] for b in batch])
    labels = {}
    keys = batch[0][1].keys()
    for k in keys:
        vals = [b[1][k] for b in batch]
        if k in ("garment_type", "piece_name"):
            labels[k] = torch.tensor(vals, dtype=torch.long)
        else:
            labels[k] = torch.tensor(vals, dtype=torch.float32)
    return images, labels


# ── Training ─────────────────────────────────────────────────────────────────


def train(
    data_dir: str | Path,
    output_path: str | Path = "weights/pattern_classifier.pt",
    annotations_file: str | Path | None = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str | None = None,
) -> dict:
    """
    Train the pattern classifier.

    Args:
        data_dir: Path to labeled image directory.
        output_path: Where to save the trained weights.
        annotations_file: Optional per-image annotations JSON.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        val_split: Fraction of data for validation.
        device: "cpu", "cuda", or "mps". Auto-detected if None.

    Returns:
        Dict with training history (loss, accuracy per epoch).
    """
    _require_torch()

    # Device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    # Dataset
    full_dataset = PatternDataset(
        root_dir=data_dir,
        annotations_file=annotations_file,
        augment=True,
    )
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    print(f"Dataset: {len(full_dataset)} images")
    print(f"  {full_dataset.summary()}")

    # Train/val split
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"  Train: {train_size}, Val: {val_size}")

    # Model
    model = PatternClassifierNet().to(device)
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_garment_acc": [], "val_piece_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        # Validate
        model.eval()
        val_loss = 0.0
        garment_correct = 0
        piece_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(images)
                loss, _ = criterion(outputs, labels)
                val_loss += loss.item()

                garment_pred = outputs["garment_type"].argmax(dim=1)
                piece_pred = outputs["piece_name"].argmax(dim=1)
                garment_correct += (garment_pred == labels["garment_type"]).sum().item()
                piece_correct += (piece_pred == labels["piece_name"]).sum().item()
                total += images.size(0)

        avg_val = val_loss / len(val_loader)
        garment_acc = garment_correct / total if total > 0 else 0
        piece_acc = piece_correct / total if total > 0 else 0

        history["val_loss"].append(avg_val)
        history["val_garment_acc"].append(garment_acc)
        history["val_piece_acc"].append(piece_acc)

        scheduler.step()

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"garment_acc={garment_acc:.2%}  piece_acc={piece_acc:.2%}"
            f"{'  *best*' if avg_val <= best_val_loss else ''}"
        )

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {output_path}")

    return history


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train pattern piece classifier")
    parser.add_argument("--data-dir", required=True, help="Path to labeled images")
    parser.add_argument(
        "--output", default="weights/pattern_classifier.pt",
        help="Output weights path",
    )
    parser.add_argument("--annotations", default=None, help="Annotations JSON file")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        annotations_file=args.annotations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
