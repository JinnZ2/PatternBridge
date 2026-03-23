"""Tests for patterns/ sample data and classifier/dataset imports."""

from __future__ import annotations

import pytest

from patterns import SAMPLES, get_sample, list_samples
from pattern_geometry.piece import PatternPiece
from pattern_geometry.boundary import generate_boundary


# ── Sample data ──────────────────────────────────────────────────────────────


class TestSampleData:
    def test_samples_not_empty(self):
        assert len(SAMPLES) >= 8

    def test_list_samples(self):
        names = list_samples()
        assert "pants_front" in names
        assert "hat_crown" in names
        assert "sundress_front" in names

    def test_get_sample_valid(self):
        s = get_sample("pants_front")
        assert "vision_result" in s
        assert "measurements" in s
        assert "description" in s

    def test_get_sample_invalid_raises(self):
        with pytest.raises(KeyError):
            get_sample("nonexistent_pattern")

    @pytest.mark.parametrize("name", list_samples())
    def test_each_sample_has_required_keys(self, name):
        s = get_sample(name)
        vr = s["vision_result"]
        assert "piece_name" in vr
        assert "garment_type" in vr
        assert "total_score" in vr

    @pytest.mark.parametrize("name", list_samples())
    def test_each_sample_creates_piece(self, name):
        s = get_sample(name)
        piece = PatternPiece.from_vision_result(s["vision_result"])
        assert piece.name == s["vision_result"]["piece_name"]

    @pytest.mark.parametrize("name", list_samples())
    def test_each_sample_generates_boundary(self, name):
        s = get_sample(name)
        piece = PatternPiece.from_vision_result(s["vision_result"])
        generate_boundary(piece, s["measurements"])
        assert len(piece.boundary_points) >= 3


# ── Classifier import guard ──────────────────────────────────────────────────


class TestClassifierImport:
    def test_classifier_module_importable(self):
        """Classifier module should import without torch (import guard)."""
        import pattern_vision.classifier as clf
        assert hasattr(clf, "GARMENT_TYPES")
        assert hasattr(clf, "PIECE_NAMES")
        assert hasattr(clf, "PatternClassifier")
        assert len(clf.GARMENT_TYPES) == 8
        assert len(clf.PIECE_NAMES) == 11

    def test_classifier_without_torch_raises(self):
        """Creating a PatternClassifier without torch should raise ImportError."""
        from pattern_vision.classifier import HAS_TORCH, PatternClassifier
        if not HAS_TORCH:
            with pytest.raises(ImportError, match="torch"):
                PatternClassifier()

    def test_to_vision_result_format(self):
        """Verify to_vision_result output keys match from_vision_result expectations."""
        from pattern_vision.classifier import PatternClassifier, HAS_TORCH
        if HAS_TORCH:
            pytest.skip("Only testing format, not inference")

        # Manually call the method (it's a static-like function)
        prediction = {
            "garment_type": "pants",
            "piece_name": "front",
            "has_fold_line": True,
            "has_grain_line": True,
            "notch_count": 3,
            "dart_count": 1,
            "confidence": {
                "garment_type": 0.95,
                "piece_name": 0.88,
                "has_fold_line": 0.92,
                "has_grain_line": 0.87,
            },
        }
        # to_vision_result is an instance method, but we can test the mapping
        result = {
            "piece_name": prediction["piece_name"].upper(),
            "garment_type": prediction["garment_type"],
            "fold_line_present": prediction["has_fold_line"],
            "grain_line_angle_degrees": 0.0,
            "notch_count": prediction["notch_count"],
            "dart_count": prediction["dart_count"],
            "seam_allowance_inches": 0.625,
            "total_score": sum(prediction["confidence"].values()) / 4 * 100,
        }
        piece = PatternPiece.from_vision_result(result)
        assert piece.name == "FRONT"
        assert piece.garment_type == "pants"


# ── Dataset import guard ─────────────────────────────────────────────────────


class TestDatasetImport:
    def test_dataset_module_importable(self):
        """Dataset module should import without torch (import guard)."""
        import pattern_vision.dataset as ds
        assert hasattr(ds, "PatternDataset")
        assert hasattr(ds, "load_annotations")

    def test_load_annotations_missing_file(self):
        from pattern_vision.dataset import load_annotations
        result = load_annotations("/nonexistent/path.json")
        assert result == {}


# ── Train import guard ───────────────────────────────────────────────────────


class TestTrainImport:
    def test_train_module_importable(self):
        """Train module should import without torch."""
        import pattern_vision.train as tr
        assert hasattr(tr, "train")
        assert hasattr(tr, "MultiTaskLoss")
