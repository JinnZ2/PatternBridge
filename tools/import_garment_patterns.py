"""
Import pattern geometry from Garment-Pattern-Generator templates.

Garment-Pattern-Generator (MIT, © 2021 Maria Korosteleva) ships a library of
parametric garment templates as JSON. Each template holds one or more panels,
and each panel is exact geometry — a vertex list plus an edge loop, where a
curved edge carries a quadratic Bézier control point:

    {"endpoints": [1, 0], "curvature": [0.5, 0.1]}

This is the cleanest pattern source the project has. It is openly licensed,
machine-readable, and already *geometry* — so it needs no vision layer, no
tile assembly, and none of the copyright caution that reproducing a printed
pattern page demands (see ``data/PROVENANCE.md``). Imported panels drop
straight into the rest of the pipeline:

    pieces = template_to_pieces("skirt_4_panels.json")
    PatternEncoder().encode(pieces[0])
    SVGWriter().save(pieces[0], "skirt_front.svg")

Curvature convention is taken from the generator's own
``pattern/core.py::_control_to_abs_coord``: the pair is a fraction *along* the
edge and a fraction *perpendicular* to it, so control points scale with the
edge rather than being fixed in space.

Usage:
    python tools/import_garment_patterns.py --src path/to/Garment-Pattern-Generator
    python tools/import_garment_patterns.py --src <dir> --out data_geometry
    python tools/import_garment_patterns.py --src <dir> --list

Requires: nothing beyond the standard library and pattern_geometry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from pattern_geometry.piece import GrainLine, PatternPiece, Point

# Templates are authored in centimetres; PatternBridge works in inches.
CM_PER_INCH = 2.54

# Points sampled along each curved edge. Twelve keeps a skirt hem visually
# smooth while leaving the boundary short enough for the encoder to stay
# sparse on straight runs.
DEFAULT_CURVE_SAMPLES = 12

# Confidence stamped on imported pieces so they clear PatternPiece's
# is_encodable gate. It is a provenance statement, not a rubric score: the
# boundary is read exactly from the template rather than inferred from a
# photograph. Templates carry no notch, dart, or seam-allowance data, so the
# band label says what the number does and does not cover.
IMPORTED_GEOMETRY_SCORE = 100.0
IMPORTED_GEOMETRY_BAND = (
    "Imported geometry - exact boundary from template; "
    "no notch, dart, or seam allowance data"
)

# Where a template lives tells us the garment. "combos" is resolved per file
# because it mixes dresses and jumpsuits.
GARMENT_TYPE_BY_DIR = {
    "skirts": "skirt",
    "pants": "pants",
    "basic tee": "top",
    "jackets": "jacket",
}

# Panel name -> classifier piece name, first match wins. Order matters:
# "lfsleeve" must reach the sleeve rule before the front rule sees "f".
PIECE_NAME_RULES: list[tuple[str, str]] = [
    ("sleeve", "sleeve"),
    ("hood", "other"),
    ("wb", "waistband"),
    ("waistband", "waistband"),
    ("collar", "collar"),
    ("cuff", "cuff"),
    ("pocket", "pocket"),
    ("front", "front"),
    ("back", "back"),
    ("left", "side"),
    ("right", "side"),
]


@dataclass
class ImportResult:
    """Counts from an import run."""

    templates: int = 0
    pieces: int = 0
    failed: int = 0
    paths: list[Path] = field(default_factory=list)


# ── Geometry ────────────────────────────────────────────────────────────────


def control_point(start: Point, end: Point, curvature: list[float]) -> Point:
    """
    Resolve a relative Bézier control point into absolute coordinates.

    ``curvature`` is (along, perpendicular) as fractions of the edge vector,
    matching Garment-Pattern-Generator's ``_control_to_abs_coord``.
    """
    ex, ey = end[0] - start[0], end[1] - start[1]
    # Perpendicular of (ex, ey), same handedness the generator uses.
    px, py = -ey, ex
    along, perp = curvature[0], curvature[1]
    return (
        start[0] + along * ex + perp * px,
        start[1] + along * ey + perp * py,
    )


def sample_quadratic(start: Point, control: Point, end: Point, samples: int) -> list[Point]:
    """
    Sample a quadratic Bézier, returning ``samples`` points from start
    (inclusive) up to but excluding ``end``.

    The end point is left off so consecutive edges can be concatenated without
    duplicating shared vertices.
    """
    points: list[Point] = []
    for i in range(samples):
        t = i / samples
        u = 1.0 - t
        points.append((
            u * u * start[0] + 2 * u * t * control[0] + t * t * end[0],
            u * u * start[1] + 2 * u * t * control[1] + t * t * end[1],
        ))
    return points


def order_edges(edges: list[dict]) -> list[tuple[int, int, list[float] | None]]:
    """
    Walk an edge list into a single closed loop of (start, end, curvature).

    Templates declare ``normalized_edge_loops``, so the given order is usually
    already a cycle; this still chains by endpoint so a template that is not
    normalised — or that lists an edge reversed — comes out as one continuous
    boundary rather than a scribble.
    """
    if not edges:
        return []

    remaining = list(edges)
    first = remaining.pop(0)
    start, end = first["endpoints"]
    loop: list[tuple[int, int, list[float] | None]] = [
        (start, end, first.get("curvature"))
    ]

    while remaining:
        cursor = loop[-1][1]
        for index, edge in enumerate(remaining):
            a, b = edge["endpoints"]
            if a == cursor:
                loop.append((a, b, edge.get("curvature")))
            elif b == cursor:
                # Traversed backwards: the control point is a fixed location,
                # so flipping direction only swaps which vertex we start from.
                loop.append((b, a, _reverse_curvature(edge.get("curvature"))))
            else:
                continue
            remaining.pop(index)
            break
        else:
            # Nothing chains onto the cursor: the panel is not a single loop.
            # Append what is left in declared order rather than dropping it.
            for edge in remaining:
                a, b = edge["endpoints"]
                loop.append((a, b, edge.get("curvature")))
            break

    return loop


def _reverse_curvature(curvature: list[float] | None) -> list[float] | None:
    """
    Express a curvature pair relative to the reversed edge.

    Along the edge the fraction measures from the other end (1 - t), and the
    perpendicular flips sign because the edge vector flipped.
    """
    if curvature is None:
        return None
    return [1.0 - curvature[0], -curvature[1]]


def panel_boundary(
    panel: dict,
    samples: int = DEFAULT_CURVE_SAMPLES,
    units_in_meter: float = 100.0,
) -> list[Point]:
    """
    Build a closed boundary for one panel, in inches.

    Straight edges contribute their start vertex; curved edges are sampled.
    """
    vertices = panel.get("vertices", [])
    if not vertices:
        return []

    # units_in_meter is how many template units make a metre: 100 means cm.
    to_inches = (100.0 / units_in_meter) / CM_PER_INCH

    boundary: list[Point] = []
    for start_index, end_index, curvature in order_edges(panel.get("edges", [])):
        try:
            start = tuple(vertices[start_index])
            end = tuple(vertices[end_index])
        except IndexError:
            continue
        if curvature:
            control = control_point(start, end, curvature)
            boundary.extend(sample_quadratic(start, control, end, samples))
        else:
            boundary.append(start)

    # The generator uses a y-up frame (a skirt's waist sits at +y); this
    # project and its SVG/PDF output are y-down. Negating y keeps pieces the
    # right way up — waist above hem, neckline above shoulder — and everything
    # downstream re-origins to the bounding box anyway.
    return [(x * to_inches, -y * to_inches) for x, y in boundary]


# ── Labelling ───────────────────────────────────────────────────────────────


def piece_name_for(panel_name: str) -> str:
    """Map a template panel name onto the classifier's piece vocabulary."""
    lowered = panel_name.lower()
    for needle, piece_name in PIECE_NAME_RULES:
        if needle in lowered:
            return piece_name
    return "other"


def garment_type_for(template_path: Path) -> str:
    """
    Infer garment type from the template's filename, then its directory.

    Filename wins because the generator files jackets under "basic tee"
    alongside the tees, and a jacket is not a top for the classifier.
    """
    stem = template_path.stem.lower()
    if "jacket" in stem:
        return "jacket"

    directory = template_path.parent.name.lower()
    if directory in GARMENT_TYPE_BY_DIR:
        return GARMENT_TYPE_BY_DIR[directory]

    if "dress" in stem:
        return "dress"
    if "skirt" in stem:
        return "skirt"
    if "pants" in stem or "jumpsuit" in stem:
        return "pants" if "pants" in stem else "other"
    if "jacket" in stem:
        return "jacket"
    if "tee" in stem:
        return "top"
    return "other"


# ── Import ──────────────────────────────────────────────────────────────────


def template_to_pieces(
    template_path: str | Path,
    samples: int = DEFAULT_CURVE_SAMPLES,
) -> list[PatternPiece]:
    """
    Convert one template file into PatternPiece objects, one per panel.

    Args:
        template_path: Path to a Garment-Pattern-Generator template JSON.
        samples: Points sampled along each curved edge.

    Returns:
        A PatternPiece per panel, with boundary_points filled in inches.
    """
    template_path = Path(template_path)
    data = json.loads(template_path.read_text())

    pattern = data.get("pattern", {})
    panels = pattern.get("panels", {})
    properties = data.get("properties", {}) or pattern.get("properties", {})
    units = float(properties.get("units_in_meter", 100.0))

    garment_type = garment_type_for(template_path)
    pieces: list[PatternPiece] = []

    for panel_name, panel in panels.items():
        boundary = panel_boundary(panel, samples=samples, units_in_meter=units)
        if len(boundary) < 3:
            continue

        piece = PatternPiece(
            name=panel_name.upper(),
            piece_number=None,
            garment_type=garment_type,
            pattern_brand="garment-pattern-generator",
            pattern_id=template_path.stem,
            boundary_points=boundary,
            # No image was involved; record where the geometry came from.
            image_source=f"garment-pattern-generator/{template_path.name}#{panel_name}",
            # Points are placed by this importer, not by the adaptive grid.
            encoding_resolution="manual",
            # PatternPiece.is_encodable gates on total_vision_score, because a
            # piece normally arrives from a photograph and a low score means
            # the boundary cannot be trusted. These boundaries are exact — read
            # from the template, never estimated — so the gate would otherwise
            # reject the most reliable geometry in the project. The band label
            # keeps this from being mistaken for an actual rubric result.
            total_vision_score=IMPORTED_GEOMETRY_SCORE,
            band_label=IMPORTED_GEOMETRY_BAND,
        )
        piece.grain_line = _default_grain_line(boundary)
        pieces.append(piece)

    return pieces


def _default_grain_line(boundary: list[Point]) -> GrainLine:
    """
    Give a panel a vertical grain line down the middle of its bounding box.

    Templates carry no grain marking — the generator only needs the outline —
    so this is a stated default, not extracted data.
    """
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    mid_x = (min(xs) + max(xs)) / 2.0
    span = max(ys) - min(ys)
    inset = span * 0.1
    return GrainLine(
        start=(mid_x, min(ys) + inset),
        end=(mid_x, max(ys) - inset),
        angle_degrees=90.0,
    )


def find_templates(src: str | Path) -> list[Path]:
    """
    Locate template JSON files inside a Garment-Pattern-Generator checkout.

    Accepts the repository root, the ``data_generation`` directory, or the
    ``Patterns`` directory itself.
    """
    src = Path(src)
    for candidate in (
        src / "data_generation" / "Patterns",
        src / "Patterns",
        src,
    ):
        if candidate.is_dir():
            found = sorted(
                path for path in candidate.rglob("*.json")
                if _looks_like_template(path)
            )
            if found:
                return found
    return []


def _looks_like_template(path: Path) -> bool:
    """Cheap structural check so config and spec files are not imported."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    panels = data.get("pattern", {}).get("panels")
    return isinstance(panels, dict) and bool(panels)


def import_templates(
    src: str | Path,
    out_dir: str | Path | None = None,
    samples: int = DEFAULT_CURVE_SAMPLES,
    dry_run: bool = False,
) -> ImportResult:
    """
    Convert every template under ``src`` and optionally write piece JSON.

    Args:
        src: Garment-Pattern-Generator checkout (or its Patterns directory).
        out_dir: Where to write ``<garment_type>/<template>/`` piece files.
            None means convert only, write nothing.
        samples: Points sampled along each curved edge.
        dry_run: Report what would be written without writing it.

    Returns:
        ImportResult with counts and written paths.
    """
    from pattern_output.data_export import save_pattern_set

    result = ImportResult()
    templates = find_templates(src)

    for template in templates:
        try:
            pieces = template_to_pieces(template, samples=samples)
        except Exception as exc:
            print(f"  FAILED {template.name}: {exc}")
            result.failed += 1
            continue

        if not pieces:
            continue

        result.templates += 1
        result.pieces += len(pieces)
        garment_type = pieces[0].garment_type
        print(f"  {template.stem:38s} {garment_type:7s} {len(pieces):2d} panels")

        if out_dir is None or dry_run:
            continue

        destination = Path(out_dir) / garment_type / template.stem
        written = save_pattern_set(
            pieces, destination, pattern_name=template.stem, include_tokens=False
        )
        result.paths.extend(written.values())

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import pattern geometry from Garment-Pattern-Generator templates."
    )
    parser.add_argument(
        "--src", required=True,
        help="Garment-Pattern-Generator checkout, or its Patterns directory",
    )
    parser.add_argument(
        "--out", default=None,
        help="Directory to write piece JSON into (omit to convert only)",
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_CURVE_SAMPLES,
        help=f"Points sampled per curved edge (default {DEFAULT_CURVE_SAMPLES})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    parser.add_argument(
        "--list", action="store_true",
        help="List the templates found and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        templates = find_templates(args.src)
        for template in templates:
            print(template)
        print(f"\n{len(templates)} templates")
        return 0

    result = import_templates(
        args.src, out_dir=args.out, samples=args.samples, dry_run=args.dry_run
    )
    print(
        f"\ntemplates={result.templates} pieces={result.pieces} "
        f"failed={result.failed} files={len(result.paths)}"
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
