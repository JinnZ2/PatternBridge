"""
CNN multi-head classifier for automated pattern piece detection.

Identifies garment type, piece name, and structural features from
pattern piece images without requiring an LLM API call.

Architecture:
    - Backbone: ResNet-18 (pretrained on ImageNet, fine-tuned)
    - Heads:
        1. garment_type: pants | dress | skirt | top | jacket | hat | sock | other
        2. piece_name: front | back | side | sleeve | collar | waistband | other
        3. has_fold_line: binary
        4. has_grain_line: binary
        5. notch_count: 0-10 (regression)
        6. dart_count: 0-6 (regression)

Requires: torch, torchvision, Pillow

Usage:
    classifier = PatternClassifier()
    classifier.load("weights/pattern_classifier.pt")
    result = classifier.predict("patterns/pants_front.jpg")
    # result = {
    #     "garment_type": "pants",
    #     "piece_name": "front",
    #     "has_fold_line": True,
    #     "has_grain_line": True,
    #     "notch_count": 4,
    #     "dart_count": 0,
    #     "confidence": {"garment_type": 0.94, "piece_name": 0.87, ...},
    # }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Label maps ───────────────────────────────────────────────────────────────

GARMENT_TYPES = [
    "pants", "dress", "skirt", "top", "jacket", "hat", "sock", "other",
]

PIECE_NAMES = [
    "front", "back", "side", "sleeve", "collar", "waistband",
    "yoke", "cuff", "pocket", "facing", "other",
]

# Standard input size for the classifier
INPUT_SIZE = 224

# ImageNet normalization (used because backbone is pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Model ────────────────────────────────────────────────────────────────────


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PatternClassifier requires torch and torchvision. "
            "Install with: pip install torch torchvision"
        )


class PatternClassifierNet(nn.Module if HAS_TORCH else object):
    """
    Multi-head ResNet-18 for pattern piece classification.

    Shared backbone with separate heads for each prediction task.
    """

    def __init__(self):
        _require_torch()
        super().__init__()

        # Backbone: ResNet-18 pretrained, remove final FC
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = 512  # ResNet-18 output dimension

        # Classification heads
        self.garment_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(GARMENT_TYPES)),
        )
        self.piece_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(PIECE_NAMES)),
        )

        # Binary heads
        self.fold_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.grain_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Regression heads (count predictions)
        self.notch_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),  # counts are non-negative
        )
        self.dart_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.features(x).flatten(1)
        return {
            "garment_type": self.garment_head(features),
            "piece_name": self.piece_head(features),
            "has_fold_line": self.fold_head(features),
            "has_grain_line": self.grain_head(features),
            "notch_count": self.notch_head(features),
            "dart_count": self.dart_head(features),
        }


# ── Classifier API ───────────────────────────────────────────────────────────


class PatternClassifier:
    """
    High-level classifier for pattern piece images.

    Wraps PatternClassifierNet with image preprocessing, weight loading,
    and result formatting.

    Args:
        device: "cpu", "cuda", or "mps". Auto-detected if None.
    """

    def __init__(self, device: str | None = None):
        _require_torch()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = PatternClassifierNet().to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._loaded = False

    def load(self, weights_path: str | Path) -> None:
        """
        Load trained weights.

        Args:
            weights_path: Path to .pt file with model state dict.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._loaded = True

    def predict(self, image_path: str | Path) -> dict:
        """
        Classify a pattern piece image.

        Args:
            image_path: Path to pattern piece image (JPG, PNG, etc.).

        Returns:
            Dict with predictions and confidence scores:
                garment_type: str
                piece_name: str
                has_fold_line: bool
                has_grain_line: bool
                notch_count: int
                dart_count: int
                confidence: dict[str, float]
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)

        # Decode predictions
        garment_probs = torch.softmax(outputs["garment_type"], dim=1)[0]
        piece_probs = torch.softmax(outputs["piece_name"], dim=1)[0]

        garment_idx = garment_probs.argmax().item()
        piece_idx = piece_probs.argmax().item()

        fold_prob = torch.sigmoid(outputs["has_fold_line"])[0, 0].item()
        grain_prob = torch.sigmoid(outputs["has_grain_line"])[0, 0].item()

        notch_count = max(0, round(outputs["notch_count"][0, 0].item()))
        dart_count = max(0, round(outputs["dart_count"][0, 0].item()))

        return {
            "garment_type": GARMENT_TYPES[garment_idx],
            "piece_name": PIECE_NAMES[piece_idx],
            "has_fold_line": fold_prob > 0.5,
            "has_grain_line": grain_prob > 0.5,
            "notch_count": min(notch_count, 10),
            "dart_count": min(dart_count, 6),
            "confidence": {
                "garment_type": float(garment_probs[garment_idx]),
                "piece_name": float(piece_probs[piece_idx]),
                "has_fold_line": float(fold_prob if fold_prob > 0.5 else 1 - fold_prob),
                "has_grain_line": float(grain_prob if grain_prob > 0.5 else 1 - grain_prob),
            },
        }

    def predict_batch(
        self, image_paths: list[str | Path], batch_size: int = 8
    ) -> list[dict]:
        """
        Classify multiple images.

        Args:
            image_paths: List of image paths.
            batch_size: Images per forward pass.

        Returns:
            List of prediction dicts.
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self.transform(img))
            batch = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch)

            # Decode each result in the batch
            garment_probs = torch.softmax(outputs["garment_type"], dim=1)
            piece_probs = torch.softmax(outputs["piece_name"], dim=1)
            fold_probs = torch.sigmoid(outputs["has_fold_line"])
            grain_probs = torch.sigmoid(outputs["has_grain_line"])

            for j in range(len(batch_paths)):
                g_idx = garment_probs[j].argmax().item()
                p_idx = piece_probs[j].argmax().item()
                f_prob = fold_probs[j, 0].item()
                gr_prob = grain_probs[j, 0].item()
                n_count = max(0, round(outputs["notch_count"][j, 0].item()))
                d_count = max(0, round(outputs["dart_count"][j, 0].item()))

                results.append({
                    "garment_type": GARMENT_TYPES[g_idx],
                    "piece_name": PIECE_NAMES[p_idx],
                    "has_fold_line": f_prob > 0.5,
                    "has_grain_line": gr_prob > 0.5,
                    "notch_count": min(n_count, 10),
                    "dart_count": min(d_count, 6),
                    "confidence": {
                        "garment_type": float(garment_probs[j][g_idx]),
                        "piece_name": float(piece_probs[j][p_idx]),
                        "has_fold_line": float(f_prob if f_prob > 0.5 else 1 - f_prob),
                        "has_grain_line": float(gr_prob if gr_prob > 0.5 else 1 - gr_prob),
                    },
                })

        return results

    def to_vision_result(self, prediction: dict) -> dict:
        """
        Convert a classifier prediction to the same format as
        PatternPromptEvaluator.evaluate() output, so it can be
        passed directly to PatternPiece.from_vision_result().

        Args:
            prediction: Output from predict().

        Returns:
            Dict compatible with PatternPiece.from_vision_result().
        """
        return {
            "piece_name": prediction["piece_name"].upper(),
            "garment_type": prediction["garment_type"],
            "fold_line_present": prediction["has_fold_line"],
            "grain_line_angle_degrees": 0.0 if prediction["has_grain_line"] else None,
            "notch_count": prediction["notch_count"],
            "dart_count": prediction["dart_count"],
            "seam_allowance_inches": 0.625,  # standard 5/8"
            "total_score": sum(prediction["confidence"].values()) / 4 * 100,
            "band_label": "Classifier",
            "pattern_brand": "unknown",
        }
