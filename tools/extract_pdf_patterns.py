"""
Extract labeled pattern piece images from sewing pattern PDFs.

Commercial and free sewing patterns ship as PDFs. This tool turns them into
the labeled image tree that ``pattern_vision.dataset.PatternDataset`` expects:

    data/<garment_type>/<piece_name>/<name>.png

Two PDF shapes are handled:

1. **Single-page pieces** — one page holds one or more complete pieces. A
   fractional ``crop`` selects the region, then the image is auto-trimmed to
   its ink bounding box.

2. **Tiled pieces** — a full-size pattern split across many letter-size pages
   that butt-join into one large sheet. ``TileLayout`` describes the grid and
   the trim box inside each page; pieces are then cropped from the assembled
   sheet.

Each saved image gets a sidecar ``.json`` with the annotation fields the
training pipeline reads plus provenance (source, license, attribution) — same
schema ``tools/fetch_patterns.py`` writes.

Licensing
---------
Pattern PDFs are copyrighted even when they are free downloads. Every registry
entry carries a ``redistributable`` flag. Entries with ``redistributable=False``
carry an explicit no-distribution notice from the PDF itself and are **skipped
by default**; they are only extracted when you pass ``--include-restricted``,
and they should be written somewhere untracked (see ``--data-dir``) so they are
never committed. See ``data/PROVENANCE.md``.

Usage:
    python tools/extract_pdf_patterns.py --list
    python tools/extract_pdf_patterns.py --pdf-dir ~/patterns            # all open entries
    python tools/extract_pdf_patterns.py --pdf-dir ~/patterns --dry-run
    python tools/extract_pdf_patterns.py --pdf-dir ~/patterns --pattern butterick_retro_wrap
    python tools/extract_pdf_patterns.py --pdf-dir ~/patterns \
        --include-restricted --data-dir data_local

Requires: pymupdf, Pillow
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - exercised via import guard tests
    fitz = None

try:
    from PIL import Image, ImageChops
except ImportError:  # pragma: no cover
    Image = None
    ImageChops = None


# ── Defaults ────────────────────────────────────────────────────────────────

# 200 dpi keeps thin pattern lines and notch marks legible while staying well
# under the classifier's appetite (images are resized to 224 px at train time).
DEFAULT_DPI = 200

# Pixels of white space left around a piece after auto-trimming.
TRIM_MARGIN_PX = 24

# Luminance below this counts as ink. Chosen to reject the pale blue
# enlargement grid Butterick prints behind its reduced-scale pieces
# (~215 luma) while keeping black pattern lines.
INK_THRESHOLD = 200


# ── Spec types ──────────────────────────────────────────────────────────────


@dataclass
class PieceSpec:
    """One pattern piece to cut out of a page or an assembled sheet."""

    name: str
    garment_type: str
    piece_name: str

    # Source location. Exactly one of these applies:
    #   page  — 1-indexed page number, for single-page PDFs
    #   sheet — True to crop from the assembled tile sheet
    page: int | None = None
    sheet: bool = False

    # Region to keep, as fractions (x0, y0, x1, y1) of the page or sheet.
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    # Annotation labels written to the sidecar JSON.
    has_fold_line: bool = False
    has_grain_line: bool = True
    notch_count: int = 0
    dart_count: int = 0


@dataclass
class TileLayout:
    """
    A full-size pattern split across pages that butt-join into one sheet.

    Pages are listed in row-major order. ``None`` marks a blank cell so a
    ragged grid still lines up.
    """

    pages: list[int | None]
    columns: int
    # Trim box inside each page, as fractions (x0, y0, x1, y1). Tiled PDFs
    # print registration marks at the edge of the printable area; the content
    # inside that box is what butt-joins.
    content_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    @property
    def rows(self) -> int:
        return (len(self.pages) + self.columns - 1) // self.columns


@dataclass
class PatternPDF:
    """A source PDF plus the pieces to pull out of it."""

    key: str
    filename: str
    title: str
    source_name: str
    license: str
    attribution: str
    redistributable: bool
    notice: str = ""
    tiles: TileLayout | None = None
    pieces: list[PieceSpec] = field(default_factory=list)


@dataclass
class ExtractResult:
    """Counts from an extraction run."""

    written: int = 0
    skipped: int = 0
    failed: int = 0
    paths: list[Path] = field(default_factory=list)


# ── Pattern registry ────────────────────────────────────────────────────────
#
# License notes are quoted from the copyright line printed in each PDF.
# None of these are public domain: US corporate works run 95 years from
# publication, so the oldest here (2007) is protected into the 2100s.

BUTTERICK_RETRO_WRAP = PatternPDF(
    key="butterick_retro_wrap",
    filename="butterickfreepatternretrowrap.pdf",
    title="Butterick Retro Wrap (free project)",
    source_name="butterick",
    license="(c)2008 Butterick, The McCall Pattern Company - free download",
    attribution="Butterick, The McCall Pattern Company",
    redistributable=True,
    notice="Free promotional download. Copyright retained by the publisher.",
    pieces=[
        PieceSpec(
            name="retro_wrap_front_back",
            garment_type="jacket",
            piece_name="front",
            page=4,
            crop=(0.03, 0.03, 0.512, 0.92),
            has_fold_line=True,
            notch_count=3,
        ),
        PieceSpec(
            name="retro_wrap_pocket",
            garment_type="jacket",
            piece_name="pocket",
            page=4,
            crop=(0.507, 0.04, 0.73, 0.26),
            has_fold_line=True,
            notch_count=2,
        ),
        PieceSpec(
            name="retro_wrap_collar",
            garment_type="jacket",
            piece_name="collar",
            page=4,
            crop=(0.718, 0.04, 0.93, 0.40),
            has_fold_line=True,
            notch_count=2,
        ),
    ],
)

MCCALLS_COSMETIC_BAG = PatternPDF(
    key="mccalls_cosmetic_bag",
    filename="mccallspatterncosmeticbag.pdf",
    title="McCall's Cosmetic Bag (free project)",
    source_name="mccalls",
    license="(c)2007 The McCall Pattern Company, All rights reserved - free download",
    attribution="The McCall Pattern Company",
    redistributable=True,
    notice="Free promotional download. Copyright retained by the publisher.",
    pieces=[
        PieceSpec(
            name="cosmetic_bag_front_back",
            garment_type="other",
            piece_name="front",
            page=4,
            dart_count=1,
            notch_count=3,
        ),
        PieceSpec(
            name="cosmetic_bag_bottom",
            garment_type="other",
            piece_name="other",
            page=5,
            crop=(0.08, 0.05, 0.53, 0.93),
            dart_count=2,
            notch_count=4,
        ),
        PieceSpec(
            name="cosmetic_bag_strap",
            garment_type="other",
            piece_name="other",
            page=5,
            crop=(0.55, 0.02, 0.88, 0.97),
            has_fold_line=True,
            notch_count=3,
        ),
        PieceSpec(
            name="cosmetic_bag_lining",
            garment_type="other",
            piece_name="facing",
            page=6,
            dart_count=2,
            notch_count=4,
        ),
    ],
)

# Stitch magazine prints its full-size insert as an 8-column x 5-row grid of
# letter pages: the tile label is <row><column>, e.g. "3e" is row 3, column e.
# Page 5 of the PDF is tile 1a, and pages run row-major from there. The trim
# box is the red dotted rectangle, 576 x 756 pt inset at (19.7, 17.6).
_AMELIA_CONTENT_BOX = (19.7 / 612, 17.62 / 792, (19.7 + 576) / 612, (17.62 + 756) / 792)

AMELIA_COAT = PatternPDF(
    key="amelia_coat",
    filename="Amelia_Coat.pdf",
    title="Amelia Coat by Katrin Vorbeck (Stitch Winter 2012)",
    source_name="interweave",
    license="(c) Interweave Press LLC - All rights reserved, not to be reprinted",
    attribution="Katrin Vorbeck / Interweave Press LLC",
    redistributable=False,
    notice=(
        "PDF states: 'Not to be reprinted. All rights reserved. Please respect "
        "the copyright by not forwarding or distributing this document.' "
        "Do not commit these images to a public repository."
    ),
    tiles=TileLayout(
        # Rows 1-4 are full 8-column rows; row 5 stops after column f.
        pages=list(range(5, 43)) + [None, None],
        columns=8,
        content_box=_AMELIA_CONTENT_BOX,
    ),
    # Crops measured off the assembled sheet. Pieces are nested closely, so a
    # few boxes clip a neighbour's corner; the dominant piece is still correct.
    pieces=[
        PieceSpec("amelia_side_back_bodice", "jacket", "back", sheet=True,
                  crop=(0.000, 0.014, 0.365, 0.203), notch_count=4),
        PieceSpec("amelia_lower_side_panel", "jacket", "side", sheet=True,
                  crop=(0.370, 0.010, 0.712, 0.393), notch_count=2),
        PieceSpec("amelia_collar", "jacket", "collar", sheet=True,
                  crop=(0.752, 0.070, 0.995, 0.660), notch_count=2),
        PieceSpec("amelia_center_back", "jacket", "back", sheet=True,
                  crop=(0.000, 0.203, 0.379, 0.371), has_fold_line=True, notch_count=4),
        PieceSpec("amelia_center_front", "jacket", "front", sheet=True,
                  crop=(0.000, 0.396, 0.372, 0.574), notch_count=4),
        PieceSpec("amelia_side_front", "jacket", "side", sheet=True,
                  crop=(0.374, 0.396, 0.736, 0.591), notch_count=4),
        PieceSpec("amelia_lower_back", "jacket", "back", sheet=True,
                  crop=(0.014, 0.616, 0.351, 0.708), has_fold_line=True, notch_count=2),
        PieceSpec("amelia_lower_sleeve", "jacket", "sleeve", sheet=True,
                  crop=(0.335, 0.612, 0.739, 0.789), notch_count=4),
        PieceSpec("amelia_sleeve", "jacket", "sleeve", sheet=True,
                  crop=(0.000, 0.720, 0.420, 0.957), notch_count=4),
        PieceSpec("amelia_lower_center_front", "jacket", "front", sheet=True,
                  crop=(0.409, 0.806, 0.732, 0.970), notch_count=2),
    ],
)

# Scanned pages with no printed registration marks. Adjacency was recovered by
# correlating ink along shared page edges and confirmed visually: pages
# 1,3 / 4,5 / 7,8 butt-join into one 2-column x 3-row sheet carrying the
# front/back and the sleeve nested together. Page 9 holds the collar complete.
# Page 6 does not join any edge of that sheet and is left unplaced.
#
# The front/back and the sleeve overlap in bounding box on the assembled
# sheet, so neither crop isolates its piece completely.
LUXURY_FUR_COAT = PatternPDF(
    key="luxury_fur_coat",
    filename="LuxuryFurCoatPattern.pdf",
    title="Luxury Fur Coat Pattern, sizes 2T-8",
    source_name="stefanie_knaus",
    license="Copyright 2015 Stefanie Knaus - for personal use only",
    attribution="Stefanie Knaus",
    redistributable=False,
    notice=(
        "PDF states: 'for personal use only'. Do not commit these images to a "
        "public repository."
    ),
    tiles=TileLayout(pages=[1, 3, 4, 5, 7, 8], columns=2),
    pieces=[
        PieceSpec("fur_coat_front_back", "jacket", "front", sheet=True,
                  crop=(0.00, 0.00, 0.80, 1.00), has_fold_line=True),
        PieceSpec("fur_coat_sleeve", "jacket", "sleeve", sheet=True,
                  crop=(0.52, 0.00, 1.00, 0.42), has_fold_line=True),
        PieceSpec("fur_coat_collar", "jacket", "collar", page=9,
                  has_fold_line=True),
    ],
)

PATTERN_PDFS: list[PatternPDF] = [
    BUTTERICK_RETRO_WRAP,
    MCCALLS_COSMETIC_BAG,
    AMELIA_COAT,
    LUXURY_FUR_COAT,
]


# ── Rendering ───────────────────────────────────────────────────────────────


def _require_deps() -> None:
    if fitz is None:
        raise ImportError(
            "extract_pdf_patterns requires PyMuPDF. Install with: pip install pymupdf"
        )
    if Image is None:
        raise ImportError(
            "extract_pdf_patterns requires Pillow. Install with: pip install Pillow"
        )


def _frac_rect(page_rect, box: tuple[float, float, float, float]):
    """Convert a fractional box into an absolute rect on ``page_rect``."""
    x0, y0, x1, y1 = box
    w, h = page_rect.width, page_rect.height
    return fitz.Rect(
        page_rect.x0 + x0 * w,
        page_rect.y0 + y0 * h,
        page_rect.x0 + x1 * w,
        page_rect.y0 + y1 * h,
    )


def render_page(
    doc,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
):
    """
    Render a 1-indexed PDF page to a PIL image.

    Args:
        doc: An open ``fitz.Document``.
        page_number: 1-indexed page number.
        dpi: Render resolution.
        box: Fractional region of the page to render.

    Returns:
        RGB PIL Image.
    """
    _require_deps()
    page = doc[page_number - 1]
    clip = _frac_rect(page.rect, box)
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def assemble_sheet(doc, layout: TileLayout, dpi: int = DEFAULT_DPI):
    """
    Butt-join a tiled pattern's pages into one large sheet image.

    Args:
        doc: An open ``fitz.Document``.
        layout: Grid description.
        dpi: Render resolution.

    Returns:
        RGB PIL Image of the assembled sheet.
    """
    _require_deps()

    # Every tile renders at the same size, so measure the first real page once.
    first = next(p for p in layout.pages if p is not None)
    probe = render_page(doc, first, dpi=dpi, box=layout.content_box)
    tw, th = probe.size

    sheet = Image.new("RGB", (tw * layout.columns, th * layout.rows), "white")
    for index, page_number in enumerate(layout.pages):
        if page_number is None:
            continue
        col, row = index % layout.columns, index // layout.columns
        tile = (
            probe
            if page_number == first and index == layout.pages.index(first)
            else render_page(doc, page_number, dpi=dpi, box=layout.content_box)
        )
        sheet.paste(tile, (col * tw, row * th))
    return sheet


def crop_fraction(img, box: tuple[float, float, float, float]):
    """Crop a PIL image to a fractional (x0, y0, x1, y1) box."""
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def autotrim(img, threshold: int = INK_THRESHOLD, margin: int = TRIM_MARGIN_PX):
    """
    Trim an image to the bounding box of its ink, then pad by ``margin``.

    Pixels darker than ``threshold`` count as ink, so pale scan backgrounds and
    printed enlargement grids do not defeat the trim. Returns the image
    unchanged when no ink is found.
    """
    _require_deps()
    mask = img.convert("L").point(lambda v: 255 if v < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    w, h = img.size
    return img.crop(
        (max(0, x0 - margin), max(0, y0 - margin),
         min(w, x1 + margin), min(h, y1 + margin))
    )


# ── Extraction ──────────────────────────────────────────────────────────────


def _sidecar(spec: PieceSpec, pattern: PatternPDF) -> dict:
    """Build the annotation + provenance dict saved beside each image."""
    return {
        "has_fold_line": spec.has_fold_line,
        "has_grain_line": spec.has_grain_line,
        "notch_count": spec.notch_count,
        "dart_count": spec.dart_count,
        "source_url": pattern.filename,
        "source_name": pattern.source_name,
        "license": pattern.license,
        "attribution": pattern.attribution,
        "redistributable": pattern.redistributable,
        "pattern_title": pattern.title,
    }


def extract_pattern(
    pdf_path: str | Path,
    pattern: PatternPDF,
    data_dir: str | Path = "data",
    dpi: int = DEFAULT_DPI,
    dry_run: bool = False,
    overwrite: bool = False,
) -> ExtractResult:
    """
    Extract every piece described by ``pattern`` out of ``pdf_path``.

    Args:
        pdf_path: Path to the source PDF.
        pattern: Registry entry describing the pieces.
        data_dir: Dataset root; images land in ``<root>/<garment>/<piece>/``.
        dpi: Render resolution.
        dry_run: Report what would be written without writing it.
        overwrite: Replace images that already exist.

    Returns:
        ExtractResult with counts and written paths.
    """
    _require_deps()
    result = ExtractResult()
    data_dir = Path(data_dir)
    doc = fitz.open(str(pdf_path))

    try:
        # Assemble the tile sheet once, but only if some piece needs it.
        sheet = None
        if pattern.tiles is not None and any(p.sheet for p in pattern.pieces):
            if not dry_run:
                sheet = assemble_sheet(doc, pattern.tiles, dpi=dpi)

        for spec in pattern.pieces:
            dest_dir = data_dir / spec.garment_type / spec.piece_name
            dest = dest_dir / f"{spec.name}.png"

            if dest.exists() and not overwrite:
                result.skipped += 1
                continue

            if dry_run:
                print(f"  would write {dest}")
                result.written += 1
                result.paths.append(dest)
                continue

            try:
                if spec.sheet:
                    if sheet is None:
                        raise ValueError(
                            f"{spec.name}: sheet=True but no tile layout defined"
                        )
                    img = crop_fraction(sheet, spec.crop)
                else:
                    if spec.page is None:
                        raise ValueError(f"{spec.name}: needs either page or sheet")
                    img = render_page(doc, spec.page, dpi=dpi, box=spec.crop)

                img = autotrim(img)
                dest_dir.mkdir(parents=True, exist_ok=True)
                img.save(dest, "PNG", optimize=True)
                dest.with_suffix(".png.json").write_text(
                    json.dumps(_sidecar(spec, pattern), indent=2)
                )
                result.written += 1
                result.paths.append(dest)
                print(f"  wrote {dest}  ({img.width}x{img.height})")
            except Exception as exc:  # keep going on a bad piece
                print(f"  FAILED {spec.name}: {exc}", file=sys.stderr)
                result.failed += 1
    finally:
        doc.close()

    return result


def extract_all(
    pdf_dir: str | Path,
    data_dir: str | Path = "data",
    key: str | None = None,
    dpi: int = DEFAULT_DPI,
    dry_run: bool = False,
    overwrite: bool = False,
    include_restricted: bool = False,
) -> ExtractResult:
    """
    Run every registry entry whose PDF is present in ``pdf_dir``.

    Restricted entries are skipped unless ``include_restricted`` is set.
    """
    total = ExtractResult()
    pdf_dir = Path(pdf_dir)

    for pattern in PATTERN_PDFS:
        if key and pattern.key != key:
            continue

        if not pattern.redistributable and not include_restricted:
            print(f"[skip] {pattern.key}: {pattern.notice}")
            total.skipped += len(pattern.pieces)
            continue

        pdf_path = pdf_dir / pattern.filename
        if not pdf_path.exists():
            print(f"[skip] {pattern.key}: {pdf_path} not found")
            total.skipped += len(pattern.pieces)
            continue

        print(f"[{pattern.key}] {pattern.title}")
        if not pattern.redistributable:
            print(f"  NOTE: {pattern.notice}")

        res = extract_pattern(
            pdf_path, pattern, data_dir=data_dir, dpi=dpi,
            dry_run=dry_run, overwrite=overwrite,
        )
        total.written += res.written
        total.skipped += res.skipped
        total.failed += res.failed
        total.paths.extend(res.paths)

    return total


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract labeled pattern piece images from sewing pattern PDFs."
    )
    parser.add_argument("--pdf-dir", default=".", help="Directory holding the source PDFs")
    parser.add_argument("--data-dir", default="data", help="Dataset root to write into")
    parser.add_argument("--pattern", help="Extract only this registry key")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Render resolution")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing images")
    parser.add_argument(
        "--include-restricted",
        action="store_true",
        help="Also extract entries whose PDF forbids redistribution "
             "(write these to an untracked --data-dir)",
    )
    parser.add_argument("--list", action="store_true", help="List registry entries and exit")
    args = parser.parse_args(argv)

    if args.list:
        for pattern in PATTERN_PDFS:
            flag = "open" if pattern.redistributable else "RESTRICTED"
            print(f"{pattern.key:24s} [{flag:10s}] {len(pattern.pieces):2d} pieces  {pattern.title}")
            print(f"{'':24s}  license: {pattern.license}")
        return 0

    result = extract_all(
        pdf_dir=args.pdf_dir,
        data_dir=args.data_dir,
        key=args.pattern,
        dpi=args.dpi,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        include_restricted=args.include_restricted,
    )
    print(
        f"\nwritten={result.written} skipped={result.skipped} failed={result.failed}"
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
