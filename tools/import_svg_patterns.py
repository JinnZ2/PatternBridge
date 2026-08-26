"""
Import pattern geometry from SVG.

Anything that draws a pattern as vectors can feed this: FreeSewing's exported
SVG, an Inkscape or Illustrator tracing, a digitiser's output, or a file this
project wrote itself. Each closed path becomes a ``PatternPiece`` boundary in
inches, ready to encode, grade and re-render.

    pieces = svg_to_pieces("aaron.svg")
    PatternEncoder().encode(pieces[0])
    SVGWriter().save(pieces[0], "aaron_front.svg")

Why SVG matters here: it is already geometry. A pattern that arrives this way
skips the vision layer, the tile assembly and the copyright caution that
reproducing a printed page demands — the same reasons ``data_geometry/`` is
the cleanest data in the project (see ``data/PROVENANCE.md``).

Scale
-----
SVG carries real-world size in its ``width``/``height`` attributes, optionally
combined with a ``viewBox``. A pattern authored in millimetres and one authored
at 96 px/inch both land at the right size in inches, provided the file declares
its units. A file with no units at all is assumed to be 96 px/inch, matching
``pattern_output.svg_writer``; pass ``units_per_inch`` to override.

Usage:
    python -m tools.import_svg_patterns pattern.svg --list
    python -m tools.import_svg_patterns pattern.svg --out data_geometry/svg
    python -m tools.import_svg_patterns pattern.svg --min-area 2.0

Requires: nothing beyond the standard library and pattern_geometry.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from pattern_geometry.piece import GrainLine, PatternPiece, Point

SVG_NS = "http://www.w3.org/2000/svg"

# CSS absolute units, expressed as inches per unit.
UNIT_TO_INCH: dict[str, float] = {
    "in": 1.0,
    "px": 1.0 / 96.0,
    "pt": 1.0 / 72.0,
    "pc": 1.0 / 6.0,
    "mm": 1.0 / 25.4,
    "cm": 1.0 / 2.54,
    "q": 1.0 / 101.6,
    "": 1.0 / 96.0,        # unitless: CSS says px
}

# Points sampled along each curve segment. Pattern curves are shallow, so this
# is plenty; the encoder thins straight runs again afterwards.
DEFAULT_CURVE_SAMPLES = 12

# Closed paths smaller than this (square inches) are treated as marks —
# notches, grain arrows, logos — rather than pattern pieces.
DEFAULT_MIN_AREA = 1.0

_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_COMMAND = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])")
_LENGTH = re.compile(r"^\s*([-+]?[\d.eE+-]+)\s*([a-zA-Z%]*)\s*$")

# Group names that describe a layer rather than a piece. Renderers — including
# this project's own SVGWriter — wrap the outline in <g id="boundary"> inside
# <g id="piece_0_FRONT">, so the innermost label is the least informative one.
# Elements whose children define things rather than draw them. A clipPath
# rectangle or a glyph outline in <defs> is a perfectly good closed path, and
# treating it as a pattern piece produces confident nonsense — a PDF-derived
# SVG can hide dozens of them.
NON_RENDERED_TAGS = {
    "defs", "clippath", "mask", "symbol", "marker", "pattern",
    "lineargradient", "radialgradient", "filter", "metadata",
}

GENERIC_LABELS = {
    "boundary", "outline", "cutline", "cutting-line", "seam", "seam-line",
    "seamline", "grain", "grain-line", "grainline", "fold", "fold-line",
    "foldline", "notches", "notch", "darts", "dart", "labels", "label",
    "text", "marks", "markings", "layer1", "g", "path", "svg", "fabric",
    "lining", "interfacing", "canvas", "pattern", "piece",
}


@dataclass
class SvgImportResult:
    """Counts from an SVG import."""

    pieces: int = 0
    skipped_small: int = 0
    paths: list[Path] = field(default_factory=list)


# ── Units and transforms ────────────────────────────────────────────────────


def parse_length(text: str | None) -> tuple[float, str] | None:
    """Split an SVG length like ``"210mm"`` into (210.0, "mm")."""
    if not text:
        return None
    match = _LENGTH.match(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value, match.group(2).lower()


def looks_like_freesewing(root: ET.Element) -> bool:
    """
    Recognise FreeSewing output by its group ids.

    FreeSewing names groups "<prefix>stack-<id>[-part-<name>]", and its user
    units are millimetres (packages/core/src/svg.mjs writes width in mm and a
    viewBox with the same numbers).
    """
    for element in root.iter():
        identifier = element.get("id") or ""
        if _PART_ID.match(identifier) or _STACK_ID.match(identifier):
            return True
    return False


def units_per_inch(root: ET.Element, override: float | None = None) -> float:
    """
    Work out how many SVG user units make one inch.

    With a ``viewBox``, the declared physical width maps onto the viewBox
    width, so the two together give the scale. Without one, user units are
    whatever the width unit says. A ``%`` width carries no physical size and is
    ignored.
    """
    if override:
        return override

    declared = parse_length(root.get("width"))
    view_box = root.get("viewBox")

    if declared and declared[1] != "%":
        value, unit = declared
        inches = value * UNIT_TO_INCH.get(unit, UNIT_TO_INCH[""])
        if inches > 0:
            if view_box:
                numbers = [float(n) for n in _NUMBER.findall(view_box)]
                if len(numbers) == 4 and numbers[2] > 0:
                    return numbers[2] / inches
            return value / inches

    # No physical width. FreeSewing drops width/height when a pattern is
    # embedded, leaving a viewBox whose units are millimetres — assuming CSS
    # pixels there would shrink the pattern by a factor of 3.8.
    if view_box and looks_like_freesewing(root):
        return 1.0 / UNIT_TO_INCH["mm"]

    return 96.0


def parse_transform(text: str | None) -> tuple[float, float, float, float, float, float]:
    """
    Reduce an SVG ``transform`` attribute to a single (a, b, c, d, e, f) matrix.

    Supports matrix, translate, scale, rotate, skewX and skewY, applied left to
    right as SVG specifies.
    """
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not text:
        return matrix

    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text):
        values = [float(n) for n in _NUMBER.findall(args)]
        name = name.lower()

        if name == "matrix" and len(values) == 6:
            step = tuple(values)
        elif name == "translate":
            tx = values[0] if values else 0.0
            ty = values[1] if len(values) > 1 else 0.0
            step = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = values[0] if values else 1.0
            sy = values[1] if len(values) > 1 else sx
            step = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            cos, sin = math.cos(angle), math.sin(angle)
            step = (cos, sin, -sin, cos, 0.0, 0.0)
            if len(values) >= 3:
                cx, cy = values[1], values[2]
                step = _multiply(_multiply((1, 0, 0, 1, cx, cy), step),
                                 (1, 0, 0, 1, -cx, -cy))
        elif name == "skewx" and values:
            step = (1.0, 0.0, math.tan(math.radians(values[0])), 1.0, 0.0, 0.0)
        elif name == "skewy" and values:
            step = (1.0, math.tan(math.radians(values[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            continue

        matrix = _multiply(matrix, step)

    return matrix


def _multiply(m: tuple, n: tuple) -> tuple:
    """Compose two SVG matrices: apply ``n`` first, then ``m``."""
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def apply_matrix(matrix: tuple, point: Point) -> Point:
    a, b, c, d, e, f = matrix
    x, y = point
    return (a * x + c * y + e, b * x + d * y + f)


# ── Path parsing ────────────────────────────────────────────────────────────


def _bezier3(p0, p1, p2, p3, samples):
    out = []
    for i in range(1, samples + 1):
        t = i / samples
        u = 1 - t
        out.append((
            u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
        ))
    return out


def _bezier2(p0, p1, p2, samples):
    out = []
    for i in range(1, samples + 1):
        t = i / samples
        u = 1 - t
        out.append((
            u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
        ))
    return out


def _arc(start, rx, ry, rotation, large_arc, sweep, end, samples):
    """
    Sample an SVG elliptical arc, converting endpoint to centre parameterisation
    as the SVG spec's implementation notes describe.
    """
    if start == end:
        return []
    if rx == 0 or ry == 0:
        return [end]

    rx, ry = abs(rx), abs(ry)
    phi = math.radians(rotation)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    dx2 = (start[0] - end[0]) / 2.0
    dy2 = (start[1] - end[1]) / 2.0
    x1 = cos_p * dx2 + sin_p * dy2
    y1 = -sin_p * dx2 + cos_p * dy2

    # Scale the radii up if they are too small to span the endpoints.
    lam = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if lam > 1:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale

    denominator = rx * rx * y1 * y1 + ry * ry * x1 * x1
    numerator = rx * rx * ry * ry - denominator
    factor = 0.0 if numerator <= 0 or denominator == 0 else math.sqrt(numerator / denominator)
    if large_arc == sweep:
        factor = -factor

    cx1 = factor * rx * y1 / ry
    cy1 = -factor * ry * x1 / rx
    cx = cos_p * cx1 - sin_p * cy1 + (start[0] + end[0]) / 2.0
    cy = sin_p * cx1 + cos_p * cy1 + (start[1] + end[1]) / 2.0

    def angle_of(ux, uy):
        return math.atan2(uy, ux)

    theta1 = angle_of((x1 - cx1) / rx, (y1 - cy1) / ry)
    theta2 = angle_of((-x1 - cx1) / rx, (-y1 - cy1) / ry)
    delta = theta2 - theta1
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    out = []
    steps = max(2, samples)
    for i in range(1, steps + 1):
        theta = theta1 + delta * (i / steps)
        px = rx * math.cos(theta)
        py = ry * math.sin(theta)
        out.append((cos_p * px - sin_p * py + cx, sin_p * px + cos_p * py + cy))
    return out


def parse_path(d: str, samples: int = DEFAULT_CURVE_SAMPLES) -> list[dict]:
    """
    Parse an SVG path into flattened subpaths.

    Returns a list of ``{"points": [...], "closed": bool}``. Curves are
    sampled; arcs are converted exactly and then sampled.
    """
    if not d:
        return []

    tokens = [tok for tok in _COMMAND.split(d) if tok.strip()]
    subpaths: list[dict] = []
    points: list[Point] = []
    current: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    last_control: Point | None = None
    previous_command = ""

    def flush(closed: bool) -> None:
        if len(points) >= 2:
            subpaths.append({"points": list(points), "closed": closed})

    index = 0
    while index < len(tokens):
        command = tokens[index]
        if not _COMMAND.fullmatch(command):
            index += 1
            continue
        args = [float(n) for n in _NUMBER.findall(tokens[index + 1])] \
            if index + 1 < len(tokens) and not _COMMAND.fullmatch(tokens[index + 1]) else []
        index += 2 if args else 1

        upper = command.upper()
        relative = command.islower()
        cursor = 0

        if upper == "Z":
            if points:
                flush(True)
                points = []
                current = start
            previous_command = upper
            continue

        while True:
            if upper == "M":
                need = 2
            elif upper in ("L", "T"):
                need = 2
            elif upper in ("H", "V"):
                need = 1
            elif upper in ("S", "Q"):
                need = 4
            elif upper == "C":
                need = 6
            elif upper == "A":
                need = 7
            else:
                break
            if cursor + need > len(args):
                break
            chunk = args[cursor:cursor + need]
            cursor += need

            if upper == "M":
                if points:
                    flush(False)
                    points = []
                point = (chunk[0], chunk[1])
                if relative:
                    point = (current[0] + point[0], current[1] + point[1])
                current = start = point
                points = [current]
                # Subsequent pairs after an M are implicit L.
                upper = "L"
                last_control = None
            elif upper == "L":
                point = (chunk[0], chunk[1])
                if relative:
                    point = (current[0] + point[0], current[1] + point[1])
                current = point
                points.append(current)
                last_control = None
            elif upper == "H":
                x = chunk[0] + (current[0] if relative else 0.0)
                current = (x, current[1])
                points.append(current)
                last_control = None
            elif upper == "V":
                y = chunk[0] + (current[1] if relative else 0.0)
                current = (current[0], y)
                points.append(current)
                last_control = None
            elif upper == "C":
                c1 = (chunk[0], chunk[1])
                c2 = (chunk[2], chunk[3])
                end = (chunk[4], chunk[5])
                if relative:
                    c1 = (current[0] + c1[0], current[1] + c1[1])
                    c2 = (current[0] + c2[0], current[1] + c2[1])
                    end = (current[0] + end[0], current[1] + end[1])
                points.extend(_bezier3(current, c1, c2, end, samples))
                last_control, current = c2, end
            elif upper == "S":
                c2 = (chunk[0], chunk[1])
                end = (chunk[2], chunk[3])
                if relative:
                    c2 = (current[0] + c2[0], current[1] + c2[1])
                    end = (current[0] + end[0], current[1] + end[1])
                if previous_command in ("C", "S") and last_control:
                    c1 = (2 * current[0] - last_control[0], 2 * current[1] - last_control[1])
                else:
                    c1 = current
                points.extend(_bezier3(current, c1, c2, end, samples))
                last_control, current = c2, end
            elif upper == "Q":
                c1 = (chunk[0], chunk[1])
                end = (chunk[2], chunk[3])
                if relative:
                    c1 = (current[0] + c1[0], current[1] + c1[1])
                    end = (current[0] + end[0], current[1] + end[1])
                points.extend(_bezier2(current, c1, end, samples))
                last_control, current = c1, end
            elif upper == "T":
                end = (chunk[0], chunk[1])
                if relative:
                    end = (current[0] + end[0], current[1] + end[1])
                if previous_command in ("Q", "T") and last_control:
                    c1 = (2 * current[0] - last_control[0], 2 * current[1] - last_control[1])
                else:
                    c1 = current
                points.extend(_bezier2(current, c1, end, samples))
                last_control, current = c1, end
            elif upper == "A":
                end = (chunk[5], chunk[6])
                if relative:
                    end = (current[0] + end[0], current[1] + end[1])
                points.extend(_arc(current, chunk[0], chunk[1], chunk[2],
                                   bool(chunk[3]), bool(chunk[4]), end, samples))
                current = end
                last_control = None

            previous_command = upper
            if cursor >= len(args):
                break

    flush(False)
    return subpaths


# ── Geometry helpers ────────────────────────────────────────────────────────


def polygon_area(points: list[Point]) -> float:
    """Absolute area of a polygon by the shoelace formula."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


# FreeSewing names its groups "<prefix>stack-<stackId>-part-<partName>", with
# the prefix configurable and defaulting to "fs-" (packages/core/src/svg.mjs,
# __renderPart). Taking everything after the last "-part-" recovers the part
# name whatever the prefix or stack happens to be, and the "stack-" form is the
# fallback for a group that never got down to a part.
_PART_ID = re.compile(r".*-part-(?P<name>.+)$", re.I)
_STACK_ID = re.compile(r".*stack-(?P<name>.+)$", re.I)


def _endpoints_meet(points: list[Point], tolerance: float = 1e-6) -> bool:
    """True if a subpath ends where it began, so it encloses an area."""
    (x0, y0), (x1, y1) = points[0], points[-1]
    span = max(
        max(p[0] for p in points) - min(p[0] for p in points),
        max(p[1] for p in points) - min(p[1] for p in points),
    )
    return math.hypot(x1 - x0, y1 - y0) <= max(tolerance, span * 1e-3)


def _clean(name: str) -> str:
    """Turn an SVG id or class into a piece name."""
    name = name.strip()
    match = _PART_ID.match(name) or _STACK_ID.match(name)
    if match:
        name = match.group("name")
    name = re.sub(r"^piece[_-]?\d*[_-]?", "", name)
    name = re.sub(r"[_-]+", " ", name)
    return name.strip().upper() or "PIECE"


# ── Import ──────────────────────────────────────────────────────────────────


def _walk(element: ET.Element, matrix: tuple, label: str, out: list) -> None:
    """Depth-first walk, accumulating transforms and the nearest group label."""
    tag = element.tag.split("}")[-1]
    if tag.lower() in NON_RENDERED_TAGS:
        return

    matrix = _multiply(matrix, parse_transform(element.get("transform")))

    own_label = element.get("id") or element.get("class") or ""
    if own_label.strip().lower() in GENERIC_LABELS:
        own_label = ""
    if tag == "g" and own_label:
        label = own_label

    # An enclosing group names the piece; a leaf shape's own id is usually
    # machine-generated (FreeSewing numbers every path "fs-1", "fs-2", ...),
    # so it is only a fallback for a shape sitting outside any named group.
    if tag == "path":
        out.append((element.get("d") or "", matrix, label or own_label))
    elif tag in ("polygon", "polyline"):
        numbers = [float(n) for n in _NUMBER.findall(element.get("points") or "")]
        pairs = list(zip(numbers[0::2], numbers[1::2]))
        if pairs:
            d = "M " + " L ".join(f"{x},{y}" for x, y in pairs)
            out.append((d + (" Z" if tag == "polygon" else ""), matrix, label or own_label))
    elif tag == "rect":
        try:
            x = float(element.get("x", 0)); y = float(element.get("y", 0))
            w = float(element.get("width", 0)); h = float(element.get("height", 0))
        except ValueError:
            w = h = 0.0
        if w > 0 and h > 0:
            out.append((f"M {x},{y} H {x+w} V {y+h} H {x} Z", matrix, label or own_label))

    for child in element:
        _walk(child, matrix, label, out)


def svg_to_pieces(
    svg_path: str | Path,
    samples: int = DEFAULT_CURVE_SAMPLES,
    min_area: float = DEFAULT_MIN_AREA,
    units_per_inch_override: float | None = None,
    garment_type: str = "unknown",
) -> list[PatternPiece]:
    """
    Read an SVG and return one PatternPiece per closed path large enough to be
    a pattern piece.

    Args:
        svg_path: Path to the SVG file.
        samples: Points sampled along each curve segment.
        min_area: Square inches below which a closed path counts as a mark.
        units_per_inch_override: Force a scale instead of reading the header.
        garment_type: Stamped on every piece produced.

    Returns:
        PatternPiece objects with boundary_points in inches, largest first.
    """
    svg_path = Path(svg_path)
    root = ET.parse(svg_path).getroot()
    scale = units_per_inch(root, units_per_inch_override)

    collected: list[tuple[str, tuple, str]] = []
    _walk(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), "", collected)

    candidates: list[tuple[float, str, list[Point]]] = []
    for d, matrix, label in collected:
        for sub in parse_path(d, samples=samples):
            if len(sub["points"]) < 3:
                continue
            # An outline that returns to its start is closed whether or not the
            # author wrote Z. Real files often omit it, and skipping those
            # would silently drop the only path that mattered.
            if not sub["closed"] and not _endpoints_meet(sub["points"]):
                continue
            points = [apply_matrix(matrix, p) for p in sub["points"]]
            inches = [(x / scale, y / scale) for x, y in points]
            area = polygon_area(inches)
            if area < min_area:
                continue
            candidates.append((area, label, inches))

    candidates.sort(key=lambda item: -item[0])

    pieces: list[PatternPiece] = []
    for index, (area, label, boundary) in enumerate(candidates):
        piece = PatternPiece(
            name=_clean(label) if label else f"PIECE {index + 1}",
            piece_number=index + 1,
            garment_type=garment_type,
            pattern_brand="svg-import",
            pattern_id=svg_path.stem,
            boundary_points=boundary,
            image_source=f"{svg_path.name}#{label}" if label else svg_path.name,
            encoding_resolution="manual",
            # Same reasoning as the Garment-Pattern-Generator importer: the
            # boundary is read exactly from vectors, never inferred from a
            # photograph, so the vision-score gate on is_encodable must not
            # reject it. The band label keeps that from reading as a rubric.
            total_vision_score=100.0,
            band_label="Imported geometry - exact boundary from SVG",
        )
        piece.grain_line = _grain_line(boundary)
        pieces.append(piece)

    return pieces


def _grain_line(boundary: list[Point]) -> GrainLine:
    """Vertical grain line down the middle of the piece's bounding box."""
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    mid_x = (min(xs) + max(xs)) / 2.0
    inset = (max(ys) - min(ys)) * 0.1
    return GrainLine(
        start=(mid_x, min(ys) + inset),
        end=(mid_x, max(ys) - inset),
        angle_degrees=90.0,
    )


def import_svg(
    svg_path: str | Path,
    out_dir: str | Path | None = None,
    samples: int = DEFAULT_CURVE_SAMPLES,
    min_area: float = DEFAULT_MIN_AREA,
    garment_type: str = "unknown",
) -> SvgImportResult:
    """Convert one SVG and optionally write piece JSON."""
    from pattern_output.data_export import save_pattern_set

    svg_path = Path(svg_path)
    pieces = svg_to_pieces(
        svg_path, samples=samples, min_area=min_area, garment_type=garment_type
    )
    result = SvgImportResult(pieces=len(pieces))

    for piece in pieces:
        xs = [p[0] for p in piece.boundary_points]
        ys = [p[1] for p in piece.boundary_points]
        print(f"  {piece.name:20s} {len(piece.boundary_points):4d} pts  "
              f"{max(xs) - min(xs):6.2f} x {max(ys) - min(ys):6.2f} in")

    if out_dir and pieces:
        written = save_pattern_set(
            pieces, Path(out_dir) / svg_path.stem,
            pattern_name=svg_path.stem, include_tokens=False,
        )
        result.paths.extend(written.values())

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import pattern geometry from an SVG file."
    )
    parser.add_argument("svg", help="SVG file to import")
    parser.add_argument("--out", default=None, help="Directory to write piece JSON into")
    parser.add_argument("--samples", type=int, default=DEFAULT_CURVE_SAMPLES,
                        help=f"Points per curve segment (default {DEFAULT_CURVE_SAMPLES})")
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA,
                        help=f"Square inches below which a path is a mark, not a piece "
                             f"(default {DEFAULT_MIN_AREA})")
    parser.add_argument("--garment-type", default="unknown", help="Label pieces with this")
    parser.add_argument("--list", action="store_true", help="List pieces without writing")
    args = parser.parse_args(argv)

    result = import_svg(
        args.svg,
        out_dir=None if args.list else args.out,
        samples=args.samples,
        min_area=args.min_area,
        garment_type=args.garment_type,
    )
    print(f"\npieces={result.pieces} files={len(result.paths)}")
    return 0 if result.pieces else 1


if __name__ == "__main__":
    raise SystemExit(main())
