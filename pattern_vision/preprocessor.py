"""
Image preprocessor for low-quality pattern piece photos.

Applies optional corrections to improve classifier and vision pipeline accuracy
on images with poor lighting, low contrast, or arbitrary rotation.

Two main capabilities:
    1. **Contrast enhancement** — log transform + adaptive histogram equalization
       for underexposed, washed-out, or low-contrast scans.
    2. **Log-polar transform** — rotation-invariant representation for pieces
       photographed at arbitrary angles.

Usage:
    from pattern_vision.preprocessor import preprocess, assess_quality

    # Auto-detect and fix bad images
    img = Image.open("pattern.jpg")
    quality = assess_quality(img)
    if quality["needs_preprocessing"]:
        img = preprocess(img, quality)

    # Or manually apply specific corrections
    from pattern_vision.preprocessor import enhance_contrast, log_polar_transform
    img = enhance_contrast(img)
    polar = log_polar_transform(img)

Requires: Pillow, numpy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageStat, ImageFilter

# ── Quality thresholds ──────────────────────────────────────────────────────

LOW_BRIGHTNESS_THRESHOLD = 60       # mean pixel value below this → underexposed
HIGH_BRIGHTNESS_THRESHOLD = 210     # mean pixel value above this → overexposed
LOW_CONTRAST_THRESHOLD = 35         # std dev below this → flat / washed out
LOW_SHARPNESS_THRESHOLD = 15.0      # Laplacian variance below this → blurry


# ── Quality assessment ──────────────────────────────────────────────────────


def assess_quality(img: Image.Image) -> dict:
    """
    Evaluate image quality and recommend preprocessing steps.

    Analyzes brightness, contrast, and sharpness to determine which
    corrections (if any) would improve downstream classification.

    Args:
        img: PIL Image (RGB).

    Returns:
        Dict with quality metrics and boolean flags:
            - brightness: float (0-255 mean pixel value)
            - contrast: float (std dev of pixel values)
            - sharpness: float (Laplacian variance)
            - is_underexposed: bool
            - is_overexposed: bool
            - is_low_contrast: bool
            - is_blurry: bool
            - needs_preprocessing: bool (True if any issue detected)
            - recommended: list[str] of correction names to apply
    """
    gray = img.convert("L")
    pixels = np.array(gray, dtype=np.float64)

    brightness = float(pixels.mean())
    contrast = float(pixels.std())

    # Sharpness via Laplacian variance
    laplacian = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[ 0, -1,  0,
                -1,  4, -1,
                 0, -1,  0],
        scale=1,
        offset=128,
    ))
    lap_pixels = np.array(laplacian, dtype=np.float64) - 128.0
    sharpness = float(lap_pixels.var())

    is_underexposed = brightness < LOW_BRIGHTNESS_THRESHOLD
    is_overexposed = brightness > HIGH_BRIGHTNESS_THRESHOLD
    is_low_contrast = contrast < LOW_CONTRAST_THRESHOLD
    is_blurry = sharpness < LOW_SHARPNESS_THRESHOLD

    recommended = []
    if is_underexposed or is_overexposed or is_low_contrast:
        recommended.append("enhance_contrast")
    if is_blurry:
        recommended.append("sharpen")

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "is_underexposed": is_underexposed,
        "is_overexposed": is_overexposed,
        "is_low_contrast": is_low_contrast,
        "is_blurry": is_blurry,
        "needs_preprocessing": len(recommended) > 0,
        "recommended": recommended,
    }


# ── Contrast enhancement ────────────────────────────────────────────────────


def enhance_contrast(
    img: Image.Image,
    *,
    log_strength: float = 1.0,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
) -> Image.Image:
    """
    Enhance image contrast using log transform + adaptive histogram equalization.

    Pipeline:
        1. Log transform — compresses bright regions, lifts shadows.
           Good for underexposed phone photos of patterns.
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization) —
           local contrast enhancement that avoids blowing out highlights.

    Args:
        img: PIL Image (RGB).
        log_strength: Blend factor for log transform (0.0 = skip, 1.0 = full).
        clahe_clip: Clipping limit for CLAHE (higher = more contrast).
        clahe_grid: Grid size for CLAHE tiles.

    Returns:
        Enhanced PIL Image (RGB).
    """
    arr = np.array(img, dtype=np.float64)

    # Step 1: Log transform
    if log_strength > 0:
        # c * log(1 + pixel) — normalized to [0, 255]
        c = 255.0 / np.log1p(255.0)
        log_arr = c * np.log1p(arr)
        arr = (1.0 - log_strength) * arr + log_strength * log_arr

    # Step 2: CLAHE per channel
    arr = _clahe(arr, clip_limit=clahe_clip, grid_size=clahe_grid)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _clahe(
    arr: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Pure numpy implementation — no OpenCV dependency.

    Args:
        arr: float64 array of shape (H, W, C) in [0, 255].
        clip_limit: Maximum slope of the cumulative histogram.
        grid_size: Number of tiles in each dimension.

    Returns:
        Equalized float64 array in [0, 255].
    """
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    result = np.empty_like(arr)
    nbins = 256

    tile_h = max(1, h // grid_size)
    tile_w = max(1, w // grid_size)

    for ch in range(channels):
        channel = arr[:, :, ch]

        # Build lookup tables for each tile
        luts = {}
        for ty in range(grid_size):
            for tx in range(grid_size):
                y0 = ty * tile_h
                y1 = min(y0 + tile_h, h)
                x0 = tx * tile_w
                x1 = min(x0 + tile_w, w)

                tile = channel[y0:y1, x0:x1]
                hist, _ = np.histogram(tile.ravel(), bins=nbins, range=(0, 256))

                # Clip histogram
                n_pixels = tile.size
                clip_val = max(1, int(clip_limit * n_pixels / nbins))
                excess = np.sum(np.maximum(hist - clip_val, 0))
                hist = np.minimum(hist, clip_val)
                hist += excess // nbins

                # Cumulative distribution → LUT
                cdf = hist.cumsum().astype(np.float64)
                cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
                denom = max(1, n_pixels - cdf_min)
                lut = ((cdf - cdf_min) / denom * 255.0).clip(0, 255)
                luts[(ty, tx)] = lut

        # Apply with bilinear interpolation between tiles
        for y in range(h):
            for x in range(w):
                # Which tile center is closest?
                fy = (y - tile_h / 2) / tile_h
                fx = (x - tile_w / 2) / tile_w

                ty0 = max(0, min(grid_size - 1, int(math.floor(fy))))
                ty1 = min(grid_size - 1, ty0 + 1)
                tx0 = max(0, min(grid_size - 1, int(math.floor(fx))))
                tx1 = min(grid_size - 1, tx0 + 1)

                wy = fy - ty0
                wx = fx - tx0
                wy = max(0.0, min(1.0, wy))
                wx = max(0.0, min(1.0, wx))

                val = int(channel[y, x].clip(0, 255))

                # Bilinear blend of 4 tile LUTs
                v00 = luts[(ty0, tx0)][val]
                v01 = luts[(ty0, tx1)][val]
                v10 = luts[(ty1, tx0)][val]
                v11 = luts[(ty1, tx1)][val]

                top = (1 - wx) * v00 + wx * v01
                bot = (1 - wx) * v10 + wx * v11
                result[y, x, ch] = (1 - wy) * top + wy * bot

    if channels == 1:
        result = result[:, :, 0]
    return result


# ── Log-polar transform ─────────────────────────────────────────────────────


def log_polar_transform(
    img: Image.Image,
    *,
    output_size: tuple[int, int] | None = None,
    center: tuple[float, float] | None = None,
) -> Image.Image:
    """
    Convert image to log-polar coordinates for rotation-invariant representation.

    In log-polar space, rotations become vertical translations and scale changes
    become horizontal translations — both easy for CNNs to handle.

    Args:
        img: PIL Image (RGB).
        output_size: (width, height) of output. Defaults to input size.
        center: (cx, cy) center point. Defaults to image center.

    Returns:
        Log-polar transformed PIL Image (RGB).
    """
    arr = np.array(img, dtype=np.float64)
    h, w = arr.shape[:2]

    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center

    if output_size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = output_size

    # Maximum radius from center to corner
    max_radius = math.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2)
    log_max = math.log(max(max_radius, 1.0))

    # Build coordinate maps
    theta = np.linspace(0, 2 * math.pi, out_h, endpoint=False)  # rows → angle
    log_r = np.linspace(0, log_max, out_w, endpoint=False)       # cols → log(radius)

    # Convert back to cartesian for sampling
    r = np.exp(log_r)
    map_x = cx + np.outer(np.cos(theta), r)  # shape: (out_h, out_w)
    map_y = cy + np.outer(np.sin(theta), r)

    # Nearest-neighbor sampling
    ix = np.clip(map_x.astype(int), 0, w - 1)
    iy = np.clip(map_y.astype(int), 0, h - 1)

    if arr.ndim == 3:
        result = arr[iy, ix, :]
    else:
        result = arr[iy, ix]

    return Image.fromarray(result.astype(np.uint8))


# ── Sharpening ──────────────────────────────────────────────────────────────


def sharpen(img: Image.Image, *, strength: float = 1.0) -> Image.Image:
    """
    Sharpen a blurry image using unsharp masking.

    Args:
        img: PIL Image (RGB).
        strength: Sharpening intensity (1.0 = standard, 2.0 = aggressive).

    Returns:
        Sharpened PIL Image (RGB).
    """
    blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
    arr = np.array(img, dtype=np.float64)
    blur_arr = np.array(blurred, dtype=np.float64)

    # Unsharp mask: original + strength * (original - blurred)
    sharpened = arr + strength * (arr - blur_arr)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return Image.fromarray(sharpened, mode="RGB")


# ── Combined pipeline ───────────────────────────────────────────────────────


def preprocess(
    img: Image.Image,
    quality: dict | None = None,
    *,
    force_all: bool = False,
) -> Image.Image:
    """
    Auto-preprocess an image based on quality assessment.

    Args:
        img: PIL Image (RGB).
        quality: Quality dict from assess_quality(). Computed if None.
        force_all: Apply all corrections regardless of quality.

    Returns:
        Preprocessed PIL Image (RGB).
    """
    if quality is None:
        quality = assess_quality(img)

    result = img

    if force_all or "enhance_contrast" in quality.get("recommended", []):
        result = enhance_contrast(result)

    if force_all or "sharpen" in quality.get("recommended", []):
        result = sharpen(result)

    return result
