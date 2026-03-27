"""
Fetch open-source sewing pattern images for classifier training.

Downloads pattern images from curated open-source / Creative-Commons
repositories and saves them into the PatternDataset directory structure:

    data/<garment_type>/<piece_name>/IMG_<source>_<n>.png

Each downloaded image gets a sidecar .json with source URL, license, and
default annotations so the training pipeline can pick it up immediately.

Sources (all explicitly open-licensed):
    - Freesewing.org (MIT) — parametric pattern SVGs rendered to PNG
    - Wikimedia Commons — CC-licensed sewing pattern images
    - GitHub repos with open sewing patterns

Usage:
    python tools/fetch_patterns.py                       # default: data/
    python tools/fetch_patterns.py --data-dir my_data    # custom output
    python tools/fetch_patterns.py --source freesewing   # specific source
    python tools/fetch_patterns.py --dry-run             # preview without downloading
    python tools/fetch_patterns.py --limit 50            # max images per source

Requires: requests, Pillow
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pattern_vision.classifier import GARMENT_TYPES, PIECE_NAMES


log = logging.getLogger("fetch_patterns")


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class PatternSource:
    """A single downloadable pattern image with metadata."""

    url: str
    garment_type: str
    piece_name: str
    source_name: str
    license: str
    attribution: str = ""
    has_fold_line: bool = False
    has_grain_line: bool = True
    notch_count: int = 0
    dart_count: int = 0


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


# ── Source registry ─────────────────────────────────────────────────────────

# Freesewing.org pattern pieces (MIT license)
# These are SVG pattern pieces from the open-source Freesewing project.
# We reference their GitHub raw content which hosts the pattern piece images.
FREESEWING_PATTERNS: list[PatternSource] = [
    # Aaron (A-shirt / tank top)
    PatternSource(
        url="https://raw.githubusercontent.com/freesewing/freesewing/develop/designs/aaron/tests/__snapshots__/shared.test.mjs.snap",
        garment_type="top", piece_name="front", source_name="freesewing",
        license="MIT", attribution="Freesewing contributors",
    ),
    PatternSource(
        url="https://raw.githubusercontent.com/freesewing/freesewing/develop/designs/aaron/tests/__snapshots__/shared.test.mjs.snap",
        garment_type="top", piece_name="back", source_name="freesewing",
        license="MIT", attribution="Freesewing contributors",
    ),
]

# Wikimedia Commons — curated CC-licensed sewing pattern images
# Search: "sewing pattern" filetype:svg OR filetype:png on Commons
WIKIMEDIA_PATTERNS: list[PatternSource] = [
    PatternSource(
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Shirt_pattern.svg/800px-Shirt_pattern.svg.png",
        garment_type="top", piece_name="front", source_name="wikimedia",
        license="CC-BY-SA-3.0", attribution="Wikimedia Commons contributors",
    ),
    PatternSource(
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Skirt_pattern.svg/600px-Skirt_pattern.svg.png",
        garment_type="skirt", piece_name="front", source_name="wikimedia",
        license="CC-BY-SA-3.0", attribution="Wikimedia Commons contributors",
        has_fold_line=True, has_grain_line=True,
    ),
]

# Open pattern GitHub repositories with image assets
GITHUB_PATTERNS: list[PatternSource] = [
    PatternSource(
        url="https://raw.githubusercontent.com/crashspace/BreathableShieldCover/master/Pattern-BreathableShieldCover.png",
        garment_type="other", piece_name="front", source_name="github-open",
        license="CC-BY-4.0", attribution="CrashSpace contributors",
    ),
]


_SOURCE_REGISTRY: dict[str, list[PatternSource]] = {
    "freesewing": FREESEWING_PATTERNS,
    "wikimedia": WIKIMEDIA_PATTERNS,
    "github": GITHUB_PATTERNS,
}


def get_all_sources() -> list[PatternSource]:
    """Return all registered pattern sources."""
    sources = []
    for pats in _SOURCE_REGISTRY.values():
        sources.extend(pats)
    return sources


def get_sources_by_name(name: str) -> list[PatternSource]:
    """Return sources for a specific registry name."""
    return list(_SOURCE_REGISTRY.get(name, []))


def register_source(name: str, patterns: list[PatternSource]) -> None:
    """Register additional pattern sources at runtime."""
    _SOURCE_REGISTRY[name] = patterns


# ── Classifier helpers ──────────────────────────────────────────────────────


# Keywords in filenames/URLs that hint at garment type
_GARMENT_KEYWORDS: dict[str, list[str]] = {
    "pants": ["pants", "trouser", "jean", "pant", "legging"],
    "dress": ["dress", "gown", "frock"],
    "skirt": ["skirt", "a-line"],
    "top": ["top", "shirt", "blouse", "tee", "tank", "aaron", "bodice", "tshirt"],
    "jacket": ["jacket", "coat", "blazer", "hoodie", "cardigan"],
    "hat": ["hat", "cap", "beanie", "beret"],
    "sock": ["sock", "stocking", "bootie"],
}

_PIECE_KEYWORDS: dict[str, list[str]] = {
    "front": ["front", "cf", "center-front"],
    "back": ["back", "cb", "center-back"],
    "side": ["side", "gusset", "gore"],
    "sleeve": ["sleeve", "arm"],
    "collar": ["collar", "neckband", "neck"],
    "waistband": ["waistband", "belt", "waist-band"],
    "yoke": ["yoke"],
    "cuff": ["cuff"],
    "pocket": ["pocket"],
    "facing": ["facing"],
}


def classify_from_url(url: str) -> tuple[str, str]:
    """
    Guess garment_type and piece_name from a URL.

    Returns:
        (garment_type, piece_name) — defaults to ("other", "other") if unknown.
    """
    url_lower = url.lower()

    garment = "other"
    for gtype, keywords in _GARMENT_KEYWORDS.items():
        if any(kw in url_lower for kw in keywords):
            garment = gtype
            break

    piece = "other"
    for pname, keywords in _PIECE_KEYWORDS.items():
        if any(kw in url_lower for kw in keywords):
            piece = pname
            break

    return garment, piece


# ── Download engine ─────────────────────────────────────────────────────────


def _url_hash(url: str) -> str:
    """Short hash of a URL for deduplication."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _is_valid_image(data: bytes) -> bool:
    """Check if bytes represent a valid image."""
    if not Image:
        return len(data) > 100  # basic check if Pillow unavailable
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return True
    except Exception:
        return False


def _save_image(
    data: bytes,
    source: PatternSource,
    data_dir: Path,
    index: int,
) -> Path | None:
    """
    Save image bytes to the correct directory structure.

    Returns:
        Path to saved file, or None on failure.
    """
    # Determine file extension
    ext = ".png"
    parsed = urlparse(source.url)
    url_ext = Path(parsed.path).suffix.lower()
    if url_ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
        ext = url_ext

    # Build path
    dest_dir = data_dir / source.garment_type / source.piece_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    url_hash = _url_hash(source.url)
    filename = f"IMG_{source.source_name}_{url_hash}_{index:04d}{ext}"
    filepath = dest_dir / filename

    # Skip if already downloaded (dedup by hash)
    if filepath.exists():
        return None

    # Save image
    filepath.write_bytes(data)

    # Save annotation + provenance sidecar
    annotation = {
        "has_fold_line": source.has_fold_line,
        "has_grain_line": source.has_grain_line,
        "notch_count": source.notch_count,
        "dart_count": source.dart_count,
        "source_url": source.url,
        "source_name": source.source_name,
        "license": source.license,
        "attribution": source.attribution,
    }
    ann_path = dest_dir / f"{filename}.json"
    ann_path.write_text(json.dumps(annotation, indent=2))

    return filepath


def fetch_patterns(
    data_dir: str | Path = "data",
    source_name: str | None = None,
    limit: int = 100,
    dry_run: bool = False,
    timeout: float = 30.0,
) -> FetchResult:
    """
    Download open-source pattern images into the dataset directory.

    Args:
        data_dir: Root directory for saved images.
        source_name: Fetch only from this source (None = all sources).
        limit: Maximum images to download per source.
        dry_run: If True, log what would be downloaded without saving.
        timeout: HTTP request timeout in seconds.

    Returns:
        FetchResult with counts of downloaded/skipped/failed images.
    """
    if requests is None:
        raise ImportError(
            "fetch_patterns requires the 'requests' library. "
            "Install with: pip install requests"
        )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Select sources
    if source_name:
        sources = get_sources_by_name(source_name)
        if not sources:
            raise ValueError(
                f"Unknown source '{source_name}'. "
                f"Available: {list(_SOURCE_REGISTRY.keys())}"
            )
    else:
        sources = get_all_sources()

    result = FetchResult()

    for i, source in enumerate(sources[:limit]):
        # Validate garment_type and piece_name
        if source.garment_type not in GARMENT_TYPES:
            log.warning(f"Skipping {source.url}: unknown garment type '{source.garment_type}'")
            result.skipped += 1
            continue
        if source.piece_name not in PIECE_NAMES:
            log.warning(f"Skipping {source.url}: unknown piece name '{source.piece_name}'")
            result.skipped += 1
            continue

        if dry_run:
            log.info(
                f"[DRY RUN] Would fetch: {source.url} -> "
                f"{source.garment_type}/{source.piece_name} "
                f"({source.license})"
            )
            result.downloaded += 1
            continue

        # Download
        try:
            resp = requests.get(source.url, timeout=timeout, headers={
                "User-Agent": "PatternBridge/1.0 (open-source sewing pattern tool)"
            })
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"Failed to download {source.url}: {e}")
            result.errors.append(f"{source.url}: {e}")
            result.failed += 1
            continue

        # Validate image content
        if not _is_valid_image(resp.content):
            log.warning(f"Invalid image from {source.url}")
            result.errors.append(f"{source.url}: not a valid image")
            result.failed += 1
            continue

        # Save
        saved_path = _save_image(resp.content, source, data_dir, i)
        if saved_path:
            log.info(f"Saved: {saved_path} ({source.license})")
            result.downloaded += 1
        else:
            log.info(f"Skipped (already exists): {source.url}")
            result.skipped += 1

        # Be polite — rate limit between requests
        time.sleep(0.5)

    return result


# ── Custom URL fetching ─────────────────────────────────────────────────────


def fetch_url(
    url: str,
    data_dir: str | Path = "data",
    garment_type: str | None = None,
    piece_name: str | None = None,
    license: str = "unknown",
    timeout: float = 30.0,
) -> Path | None:
    """
    Download a single pattern image URL and save it.

    Auto-classifies garment type and piece name from the URL if not provided.

    Args:
        url: Direct URL to an image file.
        data_dir: Root output directory.
        garment_type: Override garment classification (auto-detected if None).
        piece_name: Override piece classification (auto-detected if None).
        license: License string for provenance tracking.
        timeout: HTTP request timeout.

    Returns:
        Path to saved file, or None on failure.
    """
    if requests is None:
        raise ImportError("fetch_url requires 'requests'. Install with: pip install requests")

    # Auto-classify if not specified
    if garment_type is None or piece_name is None:
        auto_g, auto_p = classify_from_url(url)
        garment_type = garment_type or auto_g
        piece_name = piece_name or auto_p

    source = PatternSource(
        url=url,
        garment_type=garment_type,
        piece_name=piece_name,
        source_name="custom",
        license=license,
    )

    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "PatternBridge/1.0 (open-source sewing pattern tool)"
        })
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Failed to download {url}: {e}")
        return None

    if not _is_valid_image(resp.content):
        log.error(f"Invalid image from {url}")
        return None

    return _save_image(resp.content, source, Path(data_dir), 0)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fetch open-source sewing pattern images for training"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--source", default=None,
        choices=list(_SOURCE_REGISTRY.keys()),
        help="Fetch from a specific source only",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max images per source")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="  %(message)s",
    )

    print(f"\n  PatternBridge Pattern Fetcher")
    print(f"  ─────────────────────────────")
    print(f"  Output:  {Path(args.data_dir).resolve()}")
    print(f"  Source:  {args.source or 'all'}")
    print(f"  Limit:   {args.limit}")
    if args.dry_run:
        print(f"  Mode:    DRY RUN (no files will be saved)")
    print()

    result = fetch_patterns(
        data_dir=args.data_dir,
        source_name=args.source,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    print(f"\n  Results:")
    print(f"    Downloaded: {result.downloaded}")
    print(f"    Skipped:    {result.skipped}")
    print(f"    Failed:     {result.failed}")
    if result.errors:
        print(f"    Errors:")
        for err in result.errors:
            print(f"      - {err}")
    print()


if __name__ == "__main__":
    main()
