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
never committed. See ``data/PROVENANCE.md``, and ``docs/PATTERN_SOURCES.md``
for where to find patterns whose licensing allows reuse in the first place.

Usage:
    python tools/extract_pdf_patterns.py --check ~/Downloads   # vet before extracting
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
import hashlib
import json
import re
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

# Longest side a saved image may have. A full-size piece rendered at 200 dpi
# runs to five figures of pixels — ten times more than anything downstream
# reads, since the classifier resizes to 224 px. Capping here keeps the
# committed dataset small without touching legibility. 0 disables the cap.
MAX_DIMENSION = 2400

# Below this many characters of extractable text, a PDF is treated as scanned:
# --check cannot read terms printed inside artwork, so it says so instead of
# reporting a clean bill of health.
MIN_TEXT_FOR_VERDICT = 200


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
    """
    A source PDF plus the pieces to pull out of it.

    An entry with no ``pieces`` is one that has been assessed but has nothing
    mapped — either because the PDF holds no pattern pieces at all, or because
    it is restricted and mapping its tiles was not worth the effort. Recording
    it still means ``--check`` recognises the file instead of re-deriving it.
    """

    key: str
    filename: str
    title: str
    source_name: str
    license: str
    attribution: str
    redistributable: bool
    # Why a held-back entry is held. "terms" means the PDF itself forbids
    # sharing. "unknown-provenance" means it forbids nothing but nobody can
    # say where it came from, which is the common case for a pattern
    # downloaded years ago — free and paid look identical on disk. Empty for
    # anything publishable.
    hold_reason: str = ""
    notice: str = ""
    sha256: str = ""
    tiles: TileLayout | None = None
    pieces: list[PieceSpec] = field(default_factory=list)


@dataclass
class LicenseCheck:
    """What ``--check`` found in one PDF."""

    path: Path
    verdict: str          # restricted | no terms found | no text layer | unreadable
    evidence: list[str] = field(default_factory=list)
    known_key: str = ""   # registry entry this file matches, if any
    pages: int = 0
    # Anything the file says about itself. Not licensing — a search handle,
    # for when a PDF has been sitting on a phone long enough that where it
    # came from is no longer recoverable from memory.
    identity: list[str] = field(default_factory=list)


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
    sha256="7bed35396c43d29226ce08f1a099df90ce247e0468b7b77d5354a7537a0981b9",
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
    sha256="c4ef4cc729b484d92192f0702d3ac8b1fbaaad88900ebf52790d2c1e2f2836e3",
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

KWIKSEW_CLUTCH_PURSE = PatternPDF(
    key="kwiksew_clutch_purse",
    filename="clutch_purse.pdf",
    sha256="52bd864311567ea57ff0864bd753cd3ab65005f873a62f41707f53cbbd319dce",
    title="Kwik Sew 5001 Clutch Purse, sample pattern",
    source_name="kwiksew",
    license="(c)MMV (2005) Kwik Sew Pattern Co., Inc. - sample pattern, "
            "commercial/industrial use prohibited",
    attribution="Kerstin Martensson / Kwik Sew Pattern Co., Inc.",
    redistributable=True,
    notice="Free sample pattern. The notice bars commercial and industrial "
           "use; it does not bar redistribution.",
    pieces=[
        PieceSpec(
            name="clutch_purse_front",
            garment_type="other",
            piece_name="front",
            page=3,
            crop=(0.08, 0.20, 0.95, 0.94),
            has_fold_line=True,
            notch_count=2,
            dart_count=1,
        ),
        PieceSpec(
            name="clutch_purse_back",
            garment_type="other",
            piece_name="back",
            page=4,
            crop=(0.10, 0.02, 0.95, 0.96),
            has_fold_line=True,
            notch_count=2,
            dart_count=1,
        ),
    ],
)

# A 2012 tutorial carrying no copyright notice, no author and no URL. Absence
# of a notice is not a grant — copyright is automatic — but nothing in the
# document restricts use, which is why it is not flagged restricted here.
# Sock pieces map onto the classifier vocabulary as sole -> "back", upper ->
# "front"; the pattern's "direction of stretch" arrow is its grain marker.
FLEECE_SOCKS = PatternPDF(
    key="fleece_socks",
    filename="Fleece_Socks_Tutorial.pdf",
    sha256="ad2905d86545c63f2ee8411bc9676201c53a55f0e69af279a9bbf2c9c4b6e350",
    title="Fleece Socks Tutorial, women's 6-11",
    source_name="fleece_socks_tutorial",
    license="no copyright notice in PDF; rightsholder unidentified",
    attribution="unknown",
    redistributable=True,
    notice="No stated terms. Re-check before relying on this one.",
    pieces=[
        PieceSpec("fleece_sock_sole_upper", "sock", "back", page=5),
        PieceSpec("fleece_sock_sole_middle", "sock", "back", page=6,
                  has_fold_line=True),
        PieceSpec("fleece_sock_sole_lower", "sock", "back", page=7),
        PieceSpec("fleece_sock_upper_top", "sock", "front", page=8),
        PieceSpec("fleece_sock_upper_toe", "sock", "front", page=9),
    ],
)

# 25 numbered tiles butt-joining 5 columns x 5 rows (PDF page 1 is the print
# calibration square, so tiles start at page 2).
MOOD_LOTUS_LEGGING = PatternPDF(
    key="mood_lotus_legging",
    filename="MoodFabrics_MDF039_Lotus.pdf",
    sha256="c39717e7645be2b5b498eac67a4fb753689ac401c0ee9913de23114b58643dbe",
    title="Mood Fabrics MDF039 'The Lotus Legging', sizes 0-22",
    source_name="moodfabrics",
    license="no copyright notice or terms in PDF; free pattern from "
            "moodfabrics.com",
    attribution="Mood Fabrics",
    redistributable=True,
    notice="No stated terms.",
    tiles=TileLayout(pages=list(range(2, 27)), columns=5),
    pieces=[
        PieceSpec("lotus_legging_front_knee", "pants", "front", sheet=True,
                  crop=(0.39, 0.01, 0.72, 0.23)),
        PieceSpec("lotus_legging_front_thigh", "pants", "front", sheet=True,
                  crop=(0.70, 0.01, 1.00, 0.31)),
        PieceSpec("lotus_legging_back", "pants", "back", sheet=True,
                  crop=(0.00, 0.00, 0.42, 0.96)),
        PieceSpec("lotus_legging_lower_front", "pants", "front", sheet=True,
                  crop=(0.46, 0.24, 0.77, 0.60)),
        PieceSpec("lotus_legging_upper_front", "pants", "front", sheet=True,
                  crop=(0.38, 0.60, 0.79, 0.98)),
        PieceSpec("lotus_legging_waistband", "pants", "waistband", sheet=True,
                  crop=(0.76, 0.43, 1.00, 0.93), has_fold_line=True),
    ],
)

# Eight landscape tiles, 2 columns x 4 rows. These pages overlap by 88pt
# rather than butt-joining: the grey registration squares mark a 704 x 522.5pt
# content box, and aligning on those squares is what makes the curves meet.
# Assembled, the sheet is one symmetric pants pattern split down an optional
# side seam — BACK to the left of it, FRONT to the right.
ZUNES_KIDS_PANTS = PatternPDF(
    key="zunes_kids_pants",
    filename="KidsPantsPattern.pdf",
    sha256="5282e62cb945d16bc7321597ac4e7e2dafcd0b7fa3fe9572e9bef84adb689d01",
    title="Zune's Sewing Therapy Kids Pants, sizes 6M-5T",
    source_name="zunes_sewing_therapy",
    license="no copyright notice or terms in PDF",
    attribution="Zune's Sewing Therapy",
    redistributable=True,
    notice="No stated terms.",
    tiles=TileLayout(
        pages=list(range(1, 9)),
        columns=2,
        content_box=(42.5 / 792, 44.0 / 612, 746.5 / 792, 566.5 / 612),
    ),
    pieces=[
        PieceSpec("kids_pants_back", "pants", "back", sheet=True,
                  crop=(0.05, 0.03, 0.535, 0.99)),
        PieceSpec("kids_pants_front", "pants", "front", sheet=True,
                  crop=(0.528, 0.03, 0.95, 0.99)),
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
    sha256="2fcf1ea7bc557848fb739f7bd938dd71138763194fb2245cebdfdc67f8039ee8",
    title="Amelia Coat by Katrin Vorbeck (Stitch Winter 2012)",
    source_name="interweave",
    license="(c) Interweave Press LLC - All rights reserved, not to be reprinted",
    attribution="Katrin Vorbeck / Interweave Press LLC",
    redistributable=False,
    hold_reason="terms",
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
    sha256="c1057863b3bb3582944271d64144fac7d4e56742bff673f765baf9a6dc86515b",
    title="Luxury Fur Coat Pattern, sizes 2T-8",
    source_name="stefanie_knaus",
    license="Copyright 2015 Stefanie Knaus - for personal use only",
    attribution="Stefanie Knaus",
    redistributable=False,
    hold_reason="terms",
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

# Four A4 landscape tiles that butt-join 2x2: the title block and size chart
# sit top-left, FRONT top-right, BACK bottom-left, GUSSET bottom-right.
SOZO_UNDIES = PatternPDF(
    key="sozo_undies",
    filename="SoZoUndies_Pattern_A4_Version.pdf",
    sha256="de578f367776b28fd28e035e2c0dfc4a4194406f0e266795f59ab6ddbe4f4285",
    title="SoZo Undies, hip 32-50 inch",
    source_name="sozo",
    license="(c) Zoe Edwards 2021 - private home use only, may not be shared "
            "or re-distributed",
    attribution="Zoe Edwards (SoZoWhatDoYouKnow)",
    redistributable=False,
    hold_reason="terms",
    notice=(
        "PDF states: 'licensed for individual private home use only ... "
        "Pattern may not be shared, sold or re-distributed without owner's "
        "prior written consent.' Do not commit these images to a public "
        "repository."
    ),
    tiles=TileLayout(pages=[1, 2, 3, 4], columns=2),
    pieces=[
        PieceSpec("sozo_undies_front", "other", "front", sheet=True,
                  crop=(0.51, 0.01, 0.99, 0.48), has_fold_line=True),
        PieceSpec("sozo_undies_back", "other", "back", sheet=True,
                  crop=(0.03, 0.53, 0.49, 0.99), has_fold_line=True),
        # Cropped short of the test square that shares this quadrant.
        PieceSpec("sozo_undies_gusset", "other", "other", sheet=True,
                  crop=(0.55, 0.54, 0.99, 0.73), has_fold_line=True),
    ],
)

# Eight pages butt-joining 2 columns x 4 rows, one piece per row. The PDF also
# sets permission flags that deny content copying, so extracting from it goes
# against the publisher's stated intent as well as the personal-use line.
TILLY_SLIPPER_BOOTS = PatternPDF(
    key="tilly_slipper_boots",
    filename="SLIPPER_BOOTS_PDF.pdf",
    sha256="d1e3c50af46c877a4142fe34e140bf8654e76e11cacdbb80c2fb51bd77170983",
    title="Tilly and the Buttons Slipper Boots, sizes S/M/L",
    source_name="tilly_and_the_buttons",
    license="(c) Tilly and the Buttons - for personal use only; PDF denies "
            "copy permission",
    attribution="Tilly and the Buttons",
    redistributable=False,
    hold_reason="terms",
    notice=(
        "PDF states 'For personal use only' and sets permission flags that "
        "deny content copying. Do not commit these images to a public "
        "repository."
    ),
    tiles=TileLayout(pages=list(range(1, 9)), columns=2),
    pieces=[
        PieceSpec("slipper_boots_base_panel", "sock", "back", sheet=True,
                  crop=(0.25, 0.05, 1.00, 0.25)),
        PieceSpec("slipper_boots_front_panel", "sock", "front", sheet=True,
                  crop=(0.15, 0.25, 1.00, 0.50)),
        PieceSpec("slipper_boots_toe_panel", "sock", "side", sheet=True,
                  crop=(0.00, 0.50, 0.85, 0.74)),
        PieceSpec("slipper_boots_back_panel", "sock", "back", sheet=True,
                  crop=(0.01, 0.77, 0.99, 0.99)),
    ],
)

# Four unlabelled image pages. Page 3 carries two pieces; the rest carry one.
OLEDEMA_SOCKS = PatternPDF(
    key="oledema_socks",
    filename="pattern.pdf",
    sha256="fade39d72931f38b5dbb027dff0a58905b8267db12efc1f09f4a554e2704bcbd",
    title="OleDeMa sock pattern",
    source_name="oledema",
    license="OleDeMa - 'Only for personal using'",
    attribution="OleDeMa",
    redistributable=False,
    hold_reason="terms",
    notice=(
        "PDF states 'Only for personal using'. Do not commit these images to "
        "a public repository."
    ),
    pieces=[
        PieceSpec("oledema_sock_foot", "sock", "back", page=1),
        PieceSpec("oledema_sock_front_a", "sock", "front", page=2),
        PieceSpec("oledema_sock_back", "sock", "back", page=3,
                  crop=(0.06, 0.05, 0.80, 0.64)),
        PieceSpec("oledema_sock_front_b", "sock", "front", page=3,
                  crop=(0.14, 0.66, 0.80, 0.99)),
        PieceSpec("oledema_sock_top", "sock", "cuff", page=4,
                  crop=(0.28, 0.02, 0.72, 0.98), has_fold_line=True),
    ],
)

# FleeceFun's two patterns share one set of terms and both carry a repeated
# "www.FleeceFun.com" watermark across the pattern pages.
_FLEECEFUN_NOTICE = (
    "PDF states: 'You may not re-post the patterns or tutorials to the web or "
    "email them another person - if you want to share them just have them "
    "come to www.FleeceFun.com'. Pages are also watermarked. Do not commit "
    "these images to a public repository."
)

# Four tiles butt-joining 2x2 into one nested hat piece (0-3 months to adult XL).
FLEECEFUN_FLEECE_HAT = PatternPDF(
    key="fleecefun_fleece_hat",
    filename="Fleece_Fu_Basic_Fleece_Hat.pdf",
    sha256="fc2fb8c2da7820c354c7c80c6d3258b7c40f86941fb6562affe04c01e082311b",
    title="FleeceFun Basic Fleece Hat, 0-3 months to adult XL",
    source_name="fleecefun",
    license="FleeceFun.com - free pattern, may not be re-posted or emailed on",
    attribution="Angel Hickman Peterson / FleeceFun.com",
    redistributable=False,
    hold_reason="terms",
    notice=_FLEECEFUN_NOTICE,
    tiles=TileLayout(pages=[3, 4, 5, 6], columns=2),
    pieces=[
        PieceSpec("fleecefun_hat_panel", "hat", "side", sheet=True,
                  crop=(0.00, 0.00, 1.00, 0.72)),
    ],
)

# Nine tiles laid out 3 columns x 4 rows per the map on PDF page 2 (row 2 and
# row 3 have no third tile). Unlike the other tiled patterns here these pages
# carry untrimmed overlap margins rather than registration marks, so the
# butt-join is approximate and piece crops are correspondingly generous.
FLEECEFUN_PLEATED_SKIRT = PatternPDF(
    key="fleecefun_pleated_skirt",
    filename="child_pleated_skirt.pdf",
    sha256="09497300c6db5bce25f7c522847e2c0d4f457ba7b8b8d68be4bbf420f123d985",
    title="FleeceFun child's pleated skirt, size 7/8",
    source_name="fleecefun",
    license="FleeceFun.com - free pattern, may not be re-posted or emailed on",
    attribution="Angel Hickman Peterson / FleeceFun.com",
    redistributable=False,
    hold_reason="terms",
    notice=_FLEECEFUN_NOTICE,
    tiles=TileLayout(
        pages=[3, 4, 5, 6, 7, None, 8, 9, None, 10, 11, None],
        columns=3,
    ),
    pieces=[
        PieceSpec("pleated_skirt_top", "skirt", "waistband", sheet=True,
                  crop=(0.01, 0.00, 0.66, 0.21), has_fold_line=True),
        PieceSpec("pleated_skirt_panel_a", "skirt", "front", sheet=True,
                  crop=(0.01, 0.20, 0.33, 0.61)),
        PieceSpec("pleated_skirt_panel_b", "skirt", "back", sheet=True,
                  crop=(0.34, 0.20, 0.66, 0.61)),
        PieceSpec("pleated_mini_skirt_panel_a", "skirt", "front", sheet=True,
                  crop=(0.01, 0.58, 0.33, 0.90)),
        PieceSpec("pleated_mini_skirt_panel_b", "skirt", "back", sheet=True,
                  crop=(0.34, 0.58, 0.66, 0.90)),
    ],
)

# Three pages: instructions, terms, and one page of pattern pieces.
MAKEBRA_SOCK = PatternPDF(
    key="makebra_sock",
    filename="Sock_PatternMake_Bra1.pdf",
    sha256="e0b59d05fe952a5e290cc618ac4793319d51901b528cbd1bee94a71f5850d710",
    title="make Bra sock pattern, size 39-41 (UK 7-8)",
    source_name="makebra",
    license="Copyright (c) Annele Salonen Tmi - personal use only",
    attribution="Annele Salonen Tmi (make Bra)",
    redistributable=False,
    hold_reason="terms",
    notice=(
        "PDF states: 'This pattern is for your personal use only, any "
        "commercial use is prohibited.' Do not commit these images to a "
        "public repository."
    ),
    pieces=[
        PieceSpec("makebra_sock_pieces", "sock", "other", page=3, notch_count=2),
    ],
)

# Instructions only — step photos and text, no pattern pieces. Recorded so
# --check recognises it rather than re-deriving the same verdict.
TESSUTI_MONROE_TURTLENECK = PatternPDF(
    key="tessuti_monroe_turtleneck",
    filename="Monroe_Turtleneck.pdf",
    sha256="4dc18bb83078c1aef4d288d8bd2fb201a429b7423e396d6155c3286699948011",
    title="Tessuti Monroe Turtleneck - sewing instructions only",
    source_name="tessuti",
    license="(c)Tessuti Fabrics 2018 - personal use only",
    attribution="Tessuti Fabrics",
    redistributable=False,
    hold_reason="terms",
    notice=(
        "PDF states: 'Our patterns are for personal use only.' This file is "
        "the instruction booklet and holds no pattern pieces, so nothing is "
        "mapped."
    ),
)

# 33 tiles labelled <column><row> (1A-9G). Restricted, so the grid is left
# unmapped; the entry exists to record the verdict.
PEPPERMINT_PLAYSUIT = PatternPDF(
    key="peppermint_playsuit",
    filename="PLAYSUITA4.pdf",
    sha256="6465e65ea4f176f901eaf71b7b2defa1ebda0cf54e8e2bd903ac937aabadf560",
    title="Peppermint / In the Folds Playsuit",
    source_name="peppermint",
    license="Peppermint Magazine x In the Folds - for personal use only",
    attribution="In the Folds / Peppermint Magazine",
    redistributable=False,
    hold_reason="terms",
    notice=(
        "PDF states 'FOR PERSONAL USE ONLY'. Tile grid not mapped — the "
        "pattern is restricted, so its pieces are not extracted here."
    ),
)

# Two A0 sheets (2384 x 3370pt) holding ten numbered pieces nested across
# sizes XXS-5XL. Sheet 1: front, front facing, and two pocket versions.
# Sheet 2: back, sleeve, both collar parts, back facing, more pockets.
# Pieces are nested and rotated to save paper, so several crops clip a
# neighbour's edge.
LANTOKI_WORKER_JACKET = PatternPDF(
    key="lantoki_worker_jacket",
    filename="Patrones_A0.pdf",
    sha256="6ff4e5ef70511fe6c5feaaaadadb321b287e20d5127582823a7b60c9760d7322",
    title="Lantoki 'Chaqueta Worker Unisex', sizes XXS-5XL (A0 sheets)",
    source_name="lantoki",
    license="free download from the Domestika blog; no stated terms",
    attribution="Mónica Martín Rivas / Lantoki x Domestika",
    redistributable=True,
    notice=(
        "Free download published on the Domestika blog (16 Jan 2024). No "
        "terms are printed in the PDF or shown on the download page."
    ),
    pieces=[
        # ── Sheet 1 ──
        PieceSpec("lantoki_jacket_front", "jacket", "front", page=1,
                  crop=(0.00, 0.00, 0.53, 0.98), has_fold_line=True, notch_count=4),
        PieceSpec("lantoki_jacket_front_facing", "jacket", "facing", page=1,
                  crop=(0.55, 0.05, 0.71, 0.98), notch_count=2),
        PieceSpec("lantoki_jacket_upper_pocket", "jacket", "pocket", page=1,
                  crop=(0.80, 0.02, 0.98, 0.19)),
        PieceSpec("lantoki_jacket_pocket_c", "jacket", "pocket", page=1,
                  crop=(0.74, 0.20, 0.99, 0.44)),
        # ── Sheet 2 ──
        PieceSpec("lantoki_jacket_pocket_ab", "jacket", "pocket", page=2,
                  crop=(0.00, 0.01, 0.25, 0.23)),
        PieceSpec("lantoki_jacket_top_collar", "jacket", "collar", page=2,
                  crop=(0.24, 0.08, 0.60, 0.21), has_fold_line=True),
        PieceSpec("lantoki_jacket_back", "jacket", "back", page=2,
                  crop=(0.46, 0.00, 1.00, 0.98), notch_count=4),
        PieceSpec("lantoki_jacket_back_facing", "jacket", "facing", page=2,
                  crop=(0.22, 0.22, 0.47, 0.41), has_fold_line=True),
        PieceSpec("lantoki_jacket_collar", "jacket", "collar", page=2,
                  crop=(0.09, 0.29, 0.47, 0.52)),
        PieceSpec("lantoki_jacket_sleeve", "jacket", "sleeve", page=2,
                  crop=(0.00, 0.41, 0.46, 0.99), notch_count=2),
    ],
)

# Six scanned pages butt-joining 3 columns x 2 rows, holding two pieces. Carries no text at all —
# --check could only report "no text layer" — but looking at it shows the
# Patterns by Mood roundel, the same publisher as the Lotus Legging already in
# data/, so it lands in the same no-stated-terms tier.
MOOD_HOOD = PatternPDF(
    key="mood_hood",
    filename="HoodTemplate.pdf",
    sha256="701c8180e441890af90e916abfcf74c390a6bc098e50d946ad6301361ff1e38f",
    title="Patterns by Mood hood template",
    source_name="moodfabrics",
    license="no copyright notice or terms in PDF; free pattern from moodfabrics.com",
    attribution="Mood Fabrics",
    redistributable=True,
    notice="No stated terms. Publisher identified by the roundel printed on the pieces.",
    tiles=TileLayout(pages=list(range(1, 7)), columns=3),
    pieces=[
        PieceSpec("mood_hood_main", "hat", "side", sheet=True,
                  crop=(0.01, 0.00, 0.61, 0.75)),
        PieceSpec("mood_hood_band", "hat", "facing", sheet=True,
                  crop=(0.61, 0.05, 0.89, 0.63), has_fold_line=True),
    ],
)

# Held on provenance, not terms: nothing in these three forbids anything, but
# where they came from cannot be established, and a paid pattern looks exactly
# like a free one on disk. Extract them with --data-dir data_local.

# Two halves of one mitt piece, joined along a dashed edge marked "Join the two
# pattern pieces here". Registered as halves rather than assembled, because the
# pages carry margins and the join would be approximate.
BOMBAZINE_OVEN_MITT = PatternPDF(
    key="bombazine_oven_mitt",
    filename="bombazine_mitt1.pdf",
    sha256="a9bb4558443992099116a31fe31d41aeb9ea9a6c86a87b14794884bbd4694c76",
    title="Bombazine oven mitt",
    source_name="bombazine",
    license="no copyright notice or terms in PDF",
    attribution="Bombazine",
    redistributable=False,
    hold_reason="unknown-provenance",
    notice="No stated terms, and no way to establish whether this was free.",
    pieces=[
        PieceSpec("bombazine_mitt_part1", "other", "other", page=3),
        PieceSpec("bombazine_mitt_part2", "other", "other", page=4),
    ],
)

# Twenty tiles butt-joining 5 columns x 4 rows (PDF pages 6-25; pages 1-5 are
# the cover, cutting layouts and instructions).
FRENCHNAVY_ORLA_DRESS = PatternPDF(
    key="frenchnavy_orla_dress",
    filename="orlausletterpaper.pdf",
    sha256="545a68f0647576582b997ca5a2ca1ee345a623cf1878a216a393cdb6c2fbde17",
    title="French Navy 'The Orla Dress', XS-XL",
    source_name="frenchnavy",
    license="no copyright notice or terms in PDF; frenchnavy.co.za",
    attribution="French Navy",
    redistributable=False,
    hold_reason="unknown-provenance",
    notice=(
        "No stated terms. French Navy sells patterns as well as giving some "
        "away, so this could be either; nothing in the file settles it."
    ),
    tiles=TileLayout(pages=list(range(6, 26)), columns=5),
    pieces=[
        PieceSpec("orla_back_skirt", "dress", "back", sheet=True,
                  crop=(0.00, 0.00, 0.72, 0.36)),
        PieceSpec("orla_sleeve", "dress", "sleeve", sheet=True,
                  crop=(0.67, 0.01, 1.00, 0.37)),
        PieceSpec("orla_front_skirt", "dress", "front", sheet=True,
                  crop=(0.00, 0.35, 0.71, 0.71), has_fold_line=True),
        PieceSpec("orla_front_bodice", "dress", "front", sheet=True,
                  crop=(0.65, 0.50, 1.00, 0.97), has_fold_line=True, dart_count=1),
        PieceSpec("orla_back_bodice", "dress", "back", sheet=True,
                  crop=(0.05, 0.70, 0.57, 1.00), dart_count=1),
    ],
)

# Two pieces, one per page.
THREADSMONTHLY_HEADBAND = PatternPDF(
    key="threadsmonthly_headband",
    filename="widestretchheadbandUSlettersize.pdf",
    sha256="07e00e7fa2ca0d81d255f4810154f609aebab5f1fe2261acfc7c15b144f08e2d",
    title="Wide stretch headband for knit fabric, sizes 17-21 inch",
    source_name="threadsmonthly",
    license="no copyright notice or terms in PDF; ThreadsMonthly.com tutorial",
    attribution="ThreadsMonthly.com",
    redistributable=False,
    hold_reason="unknown-provenance",
    notice="No stated terms. Names a tutorial site, which does not say it was free.",
    pieces=[
        PieceSpec("headband_front", "other", "front", page=3),
        PieceSpec("headband_elastic_casing", "other", "other", page=4),
    ],
)

PATTERN_PDFS: list[PatternPDF] = [
    BUTTERICK_RETRO_WRAP,
    MCCALLS_COSMETIC_BAG,
    KWIKSEW_CLUTCH_PURSE,
    FLEECE_SOCKS,
    MOOD_LOTUS_LEGGING,
    ZUNES_KIDS_PANTS,
    AMELIA_COAT,
    LUXURY_FUR_COAT,
    SOZO_UNDIES,
    TILLY_SLIPPER_BOOTS,
    OLEDEMA_SOCKS,
    FLEECEFUN_FLEECE_HAT,
    FLEECEFUN_PLEATED_SKIRT,
    MAKEBRA_SOCK,
    TESSUTI_MONROE_TURTLENECK,
    PEPPERMINT_PLAYSUIT,
    LANTOKI_WORKER_JACKET,
    MOOD_HOOD,
    BOMBAZINE_OVEN_MITT,
    FRENCHNAVY_ORLA_DRESS,
    THREADSMONTHLY_HEADBAND,
]


# ── Licence triage ──────────────────────────────────────────────────────────

# Phrases that mean the rightsholder has restricted what you may do with the
# file itself. Ordered most to least specific; the reason is what gets shown.
#
# Deliberately narrow: a bare "all rights reserved" is NOT here, because on
# free promotional patterns it appears alongside permission to download and
# share, and treating it as a block would reject most usable patterns. What
# these look for is a restriction on *distribution* or a limit to *personal
# use*, which is the line data/PROVENANCE.md draws.
RESTRICTION_SIGNALS: list[tuple[str, str]] = [
    (r"not\s+to\s+be\s+reprinted", "forbids reprinting"),
    (r"may\s+not\s+be\s+(?:shared|re-?distributed|re-?posted|copied)",
     "forbids sharing/redistribution"),
    (r"may\s+not\s+re-?post", "forbids reposting"),
    (r"not\s+for\s+(?:reproduction|redistribution)", "forbids reproduction"),
    (r"do\s+not\s+(?:share|distribute|forward)", "forbids sharing"),
    (r"not\s+forwarding\s+or\s+distributing", "forbids forwarding"),
    (r"for\s+(?:your\s+)?personal\s+use\s+only", "personal use only"),
    (r"personal\s+use\s+only", "personal use only"),
    (r"only\s+for\s+personal\s+us(?:e|ing)", "personal use only"),
    (r"private\s+home\s+use\s+only", "private home use only"),
    (r"for\s+individual\s+use\s+only", "individual use only"),
]

_RESTRICTION_RE = [(re.compile(p, re.I), reason) for p, reason in RESTRICTION_SIGNALS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_registry_entry(path: Path) -> PatternPDF | None:
    """Match a file against the registry by content hash, then by filename."""
    try:
        digest = _sha256(path)
    except OSError:
        digest = ""
    if digest:
        for pattern in PATTERN_PDFS:
            if pattern.sha256 and pattern.sha256 == digest:
                return pattern
    for pattern in PATTERN_PDFS:
        if pattern.filename.lower() == path.name.lower():
            return pattern
    return None


# Lines naming a publisher, designer, or pattern number are the useful
# handles for looking a forgotten pattern back up.
_IDENTITY_RE = [
    re.compile(r"https?://[^\s)>\]]+", re.I),
    re.compile(r"(?:www\.)[\w.-]+\.[a-z]{2,}", re.I),
    # Both the glyph and the ASCII fallback some PDFs use.
    re.compile(r"(?:©|\(c\))\s*\S.{0,70}", re.I),
    re.compile(r"copyright\s+\S.{0,70}", re.I),
    re.compile(r"\b(?:pattern|patrón|modelo)\s*(?:no\.?|#|number)?\s*\d{3,6}\b", re.I),
]


def identify_pdf(doc, text: str) -> list[str]:
    """
    Collect whatever a PDF says about its own origin.

    This is deliberately not a licensing signal. It exists so a file whose
    provenance has been forgotten still yields something searchable — a
    publisher, a designer, a URL, a pattern number.
    """
    found: list[str] = []
    seen: set[str] = set()

    meta = getattr(doc, "metadata", None) or {}

    def add(entry: str) -> None:
        entry = " ".join(entry.split())[:90]
        if len(entry) >= 4 and entry.lower() not in seen:
            seen.add(entry.lower())
            found.append(entry)

    # Title and author name the thing; a publisher line or URL says who sold
    # it; the date is the weakest handle, so it goes last.
    for key in ("title", "author"):
        if (meta.get(key) or "").strip():
            add(f"{key}: {meta[key].strip()}")

    for regex in _IDENTITY_RE:
        for match in regex.findall(text):
            add(str(match))
            if len(found) >= 6:
                break

    created = (meta.get("creationDate") or "").strip()
    if created.startswith("D:") and len(created) >= 10:
        add(f"created: {created[2:6]}-{created[6:8]}-{created[8:10]}")

    return found[:6]


def check_pdf(path: str | Path) -> LicenseCheck:
    """
    Read a PDF's text and report whether it restricts redistribution.

    A "no terms found" verdict means nothing in the document restricts use.
    That is not the same as permission — copyright applies without a notice —
    but it does mean there is no stated restriction to honour.
    """
    _require_deps()
    path = Path(path)
    known = find_registry_entry(path)

    try:
        doc = fitz.open(str(path))
    except Exception:
        return LicenseCheck(path, "unreadable", known_key=known.key if known else "")

    try:
        pages = len(doc)
        text = "\n".join(page.get_text() for page in doc)
        identity = identify_pdf(doc, text)
    finally:
        doc.close()

    # Collapse the soft hyphens and line breaks that split phrases across lines.
    flat = re.sub(r"[-­]?\s*\n\s*", " ", text)
    flat = re.sub(r"\s+", " ", flat)

    evidence: list[str] = []
    seen: set[str] = set()
    for regex, reason in _RESTRICTION_RE:
        match = regex.search(flat)
        if not match or reason in seen:
            continue
        seen.add(reason)
        start = max(0, match.start() - 60)
        quote = flat[start:match.end() + 80].strip()
        evidence.append(f"{reason}: “…{quote}…”")

    if evidence:
        verdict = "restricted"
    elif len(flat.strip()) < MIN_TEXT_FOR_VERDICT:
        # Scanned or image-only PDF: the terms may be printed in the artwork
        # where no text scan can reach them. Saying "no terms" here would be a
        # false all-clear, which is the one mistake this check must not make.
        verdict = "no text layer"
    else:
        verdict = "no terms found"

    return LicenseCheck(
        path=path,
        verdict=verdict,
        evidence=evidence,
        known_key=known.key if known else "",
        pages=pages,
        identity=identity,
    )


def check_paths(targets: list[str | Path]) -> list[LicenseCheck]:
    """Check every PDF in the given files and/or directories."""
    files: list[Path] = []
    for target in targets:
        target = Path(target)
        if target.is_dir():
            files.extend(sorted(target.rglob("*.pdf")))
        elif target.suffix.lower() == ".pdf":
            files.append(target)
    return [check_pdf(f) for f in files]


def print_checks(checks: list[LicenseCheck]) -> None:
    """Print a phone-readable triage table."""
    if not checks:
        print("no PDFs found")
        return

    for check in checks:
        if check.known_key:
            registered = next(p for p in PATTERN_PDFS if p.key == check.known_key)
            flag = "ALREADY HAVE"
            note = f"registered as {registered.key}"
            hint = ""
        elif check.verdict == "restricted":
            flag = "DO NOT USE"
            note = check.evidence[0]
            hint = ""
        elif check.verdict == "no text layer":
            flag = "CHECK BY EYE"
            note = (f"{check.pages} scanned page(s), no searchable text — "
                    "terms may be printed in the artwork")
            hint = "keep local until you have looked at it"
        elif check.verdict == "unreadable":
            flag = "UNREADABLE"
            note = "could not open this file"
            hint = ""
        else:
            flag = "NO TERMS"
            note = f"nothing restricts sharing in {check.pages} pages"
            hint = "not a licence — keep local unless you know it was free"

        print(f"{flag:13s} {check.path.name}")
        print(f"{'':13s} {note}")
        if check.identity:
            print(f"{'':13s} says of itself: {' | '.join(check.identity[:4])}")
        if hint:
            print(f"{'':13s} -> {hint}")

    usable = sum(1 for c in checks if c.verdict == "no terms found" and not c.known_key)
    blocked = sum(1 for c in checks if c.verdict == "restricted" and not c.known_key)
    manual = sum(1 for c in checks if c.verdict == "no text layer" and not c.known_key)
    known = sum(1 for c in checks if c.known_key)
    print(f"\nno-terms={usable} restricted={blocked} needs-eyeball={manual} "
          f"already-registered={known}")
    print("NO TERMS means the file does not say you cannot share it. It is not "
          "a licence, and it cannot see where the file came from — a pattern "
          "you paid for often prints no terms at all. When you cannot place a "
          "file, extract it with --data-dir data_local and keep it out of the "
          "repo; training reads either tree. See data/PROVENANCE.md.")


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


def downscale(img, max_dim: int = MAX_DIMENSION):
    """
    Shrink an image so its longest side is at most ``max_dim``, keeping aspect.

    Returns the image untouched when it already fits, or when ``max_dim`` is 0.
    """
    _require_deps()
    if max_dim <= 0:
        return img
    longest = max(img.size)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.LANCZOS)


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
    max_dimension: int = MAX_DIMENSION,
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
        max_dimension: Cap on the longest side of a saved image; 0 disables.

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

                img = downscale(autotrim(img), max_dimension)
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
    max_dimension: int = MAX_DIMENSION,
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
            max_dimension=max_dimension,
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
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=MAX_DIMENSION,
        help=f"Cap the longest side of saved images (default {MAX_DIMENSION}, 0 disables)",
    )
    parser.add_argument(
        "--check",
        nargs="+",
        metavar="PATH",
        help="Report whether each PDF (or every PDF in a directory) restricts "
             "redistribution, then exit. Does not extract anything.",
    )
    parser.add_argument("--list", action="store_true", help="List registry entries and exit")
    args = parser.parse_args(argv)

    if args.check:
        print_checks(check_paths(args.check))
        return 0

    if args.list:
        for pattern in PATTERN_PDFS:
            if pattern.redistributable:
                flag = "open"
            elif pattern.hold_reason == "unknown-provenance":
                flag = "LOCAL ONLY"
            else:
                flag = "RESTRICTED"
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
        max_dimension=args.max_dimension,
    )
    print(
        f"\nwritten={result.written} skipped={result.skipped} failed={result.failed}"
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
