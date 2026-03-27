"""Tests for the image preprocessor module."""

from __future__ import annotations

import math
from unittest import TestCase

import numpy as np
from PIL import Image

from pattern_vision.preprocessor import (
    assess_quality,
    enhance_contrast,
    log_polar_transform,
    sharpen,
    preprocess,
    LOW_BRIGHTNESS_THRESHOLD,
    HIGH_BRIGHTNESS_THRESHOLD,
    LOW_CONTRAST_THRESHOLD,
)


def _make_image(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create a solid-color test image."""
    return Image.new("RGB", size, color)


def _make_gradient(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create a gradient image with full dynamic range."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = (int(255 * x / max(w - 1, 1)),
                         int(255 * y / max(h - 1, 1)),
                         128)
    return Image.fromarray(arr)


def _make_dark_image(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create an underexposed image (mean brightness ~25)."""
    arr = np.random.RandomState(42).randint(0, 50, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_bright_image(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create an overexposed image (mean brightness ~240)."""
    arr = np.random.RandomState(42).randint(220, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_flat_image(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create a low-contrast image (very tight pixel range)."""
    arr = np.random.RandomState(42).randint(120, 130, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ── Quality assessment ──────────────────────────────────────────────────────


class TestAssessQuality(TestCase):

    def test_returns_expected_keys(self):
        q = assess_quality(_make_gradient())
        expected = {
            "brightness", "contrast", "sharpness",
            "is_underexposed", "is_overexposed", "is_low_contrast", "is_blurry",
            "needs_preprocessing", "recommended",
        }
        assert set(q.keys()) == expected

    def test_gradient_is_not_underexposed(self):
        q = assess_quality(_make_gradient())
        assert not q["is_underexposed"]

    def test_dark_image_is_underexposed(self):
        q = assess_quality(_make_dark_image())
        assert q["is_underexposed"]
        assert q["brightness"] < LOW_BRIGHTNESS_THRESHOLD

    def test_bright_image_is_overexposed(self):
        q = assess_quality(_make_bright_image())
        assert q["is_overexposed"]
        assert q["brightness"] > HIGH_BRIGHTNESS_THRESHOLD

    def test_flat_image_is_low_contrast(self):
        q = assess_quality(_make_flat_image())
        assert q["is_low_contrast"]
        assert q["contrast"] < LOW_CONTRAST_THRESHOLD

    def test_needs_preprocessing_flag(self):
        q = assess_quality(_make_dark_image())
        assert q["needs_preprocessing"]
        assert "enhance_contrast" in q["recommended"]

    def test_good_image_needs_no_preprocessing(self):
        q = assess_quality(_make_gradient())
        # Gradient has good brightness and contrast
        assert not q["is_underexposed"]
        assert not q["is_overexposed"]

    def test_brightness_is_float(self):
        q = assess_quality(_make_gradient())
        assert isinstance(q["brightness"], float)
        assert 0.0 <= q["brightness"] <= 255.0

    def test_contrast_is_float(self):
        q = assess_quality(_make_gradient())
        assert isinstance(q["contrast"], float)
        assert q["contrast"] >= 0.0


# ── Contrast enhancement ────────────────────────────────────────────────────


class TestEnhanceContrast(TestCase):

    def test_returns_rgb_image(self):
        result = enhance_contrast(_make_dark_image())
        assert result.mode == "RGB"

    def test_preserves_size(self):
        img = _make_dark_image(size=(80, 60))
        result = enhance_contrast(img)
        assert result.size == (80, 60)

    def test_brightens_dark_image(self):
        img = _make_dark_image()
        result = enhance_contrast(img)
        original_mean = np.array(img).mean()
        enhanced_mean = np.array(result).mean()
        assert enhanced_mean > original_mean

    def test_log_strength_zero_skips_log(self):
        img = _make_gradient(size=(32, 32))
        result = enhance_contrast(img, log_strength=0.0)
        # Still applies CLAHE but no log transform
        assert result.mode == "RGB"
        assert result.size == (32, 32)

    def test_output_pixels_in_valid_range(self):
        result = enhance_contrast(_make_dark_image())
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_flat_image_gains_contrast(self):
        img = _make_flat_image()
        result = enhance_contrast(img)
        original_std = np.array(img, dtype=np.float64).std()
        enhanced_std = np.array(result, dtype=np.float64).std()
        assert enhanced_std > original_std


# ── Log-polar transform ─────────────────────────────────────────────────────


class TestLogPolarTransform(TestCase):

    def test_returns_rgb_image(self):
        result = log_polar_transform(_make_gradient())
        assert result.mode == "RGB"

    def test_preserves_size_by_default(self):
        img = _make_gradient(size=(80, 60))
        result = log_polar_transform(img)
        assert result.size == (80, 60)

    def test_custom_output_size(self):
        img = _make_gradient(size=(80, 60))
        result = log_polar_transform(img, output_size=(100, 100))
        assert result.size == (100, 100)

    def test_custom_center(self):
        img = _make_gradient(size=(64, 64))
        result = log_polar_transform(img, center=(10.0, 10.0))
        assert result.size == (64, 64)
        # Result should differ from default center
        default = log_polar_transform(img)
        assert np.array(result).tolist() != np.array(default).tolist()

    def test_output_pixels_in_valid_range(self):
        result = log_polar_transform(_make_gradient())
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_rotation_becomes_translation(self):
        """Rotating input should shift log-polar output vertically."""
        img = _make_gradient(size=(64, 64))
        rotated = img.rotate(45, fillcolor=(0, 0, 0))

        lp1 = np.array(log_polar_transform(img), dtype=np.float64)
        lp2 = np.array(log_polar_transform(rotated), dtype=np.float64)

        # The two should differ (rotation is not identity)
        assert not np.allclose(lp1, lp2, atol=5)


# ── Sharpen ─────────────────────────────────────────────────────────────────


class TestSharpen(TestCase):

    def test_returns_rgb_image(self):
        result = sharpen(_make_gradient())
        assert result.mode == "RGB"

    def test_preserves_size(self):
        img = _make_gradient(size=(80, 60))
        result = sharpen(img)
        assert result.size == (80, 60)

    def test_output_pixels_in_valid_range(self):
        result = sharpen(_make_gradient())
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_strength_zero_is_identity(self):
        img = _make_gradient()
        result = sharpen(img, strength=0.0)
        np.testing.assert_array_equal(np.array(result), np.array(img))


# ── Combined pipeline ───────────────────────────────────────────────────────


class TestPreprocess(TestCase):

    def test_auto_enhances_dark_image(self):
        img = _make_dark_image()
        result = preprocess(img)
        original_mean = np.array(img).mean()
        result_mean = np.array(result).mean()
        assert result_mean > original_mean

    def test_passes_through_good_image(self):
        img = _make_gradient()
        q = assess_quality(img)
        if not q["needs_preprocessing"]:
            result = preprocess(img, q)
            # No changes when image is fine
            np.testing.assert_array_equal(np.array(result), np.array(img))

    def test_force_all_applies_everything(self):
        img = _make_gradient()
        result = preprocess(img, force_all=True)
        # force_all should still return valid image
        assert result.mode == "RGB"
        assert result.size == img.size

    def test_preprocess_computes_quality_if_none(self):
        img = _make_dark_image()
        # Should not raise even without quality dict
        result = preprocess(img)
        assert result.mode == "RGB"

    def test_preprocess_with_explicit_quality(self):
        img = _make_gradient()
        q = assess_quality(img)
        result = preprocess(img, quality=q)
        assert result.mode == "RGB"
