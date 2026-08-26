"""
SVGWriter: Converts a PatternPiece to a printable SVG file.

Output is at real-world scale — 1 inch = 96px (SVG standard).
Pattern pieces render with semantic layers:
    - boundary (outer seam line)
    - grain line
    - fold line
    - notches
    - darts
    - lengthen/shorten lines
    - labels and annotations

SVG is structured for both screen viewing and print-ready output.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

try:
    import svgwrite
    from svgwrite import Drawing
    from svgwrite.container import Group
except ImportError:
    raise ImportError(
        "svgwrite is required for SVGWriter. "
        'Install with: pip install "patternbridge[svg]"'
    )

from pattern_geometry.piece import PatternPiece, Point, Notch, Dart


# ── Constants ─────────────────────────────────────────────────────────────────

PX_PER_INCH = 96.0          # SVG standard
MM_PER_INCH = 25.4
CM_PER_INCH = 2.54

# Visual style constants
BOUNDARY_STROKE        = "#000000"
BOUNDARY_WIDTH         = 1.5        # px
SEAM_LINE_STROKE       = "#333333"
SEAM_LINE_WIDTH        = 0.75
GRAIN_LINE_STROKE      = "#0055AA"
GRAIN_LINE_WIDTH       = 1.0
FOLD_LINE_STROKE       = "#AA0055"
FOLD_LINE_WIDTH        = 1.5
FOLD_LINE_DASH         = "8,4"
NOTCH_STROKE           = "#000000"
NOTCH_FILL             = "#000000"
NOTCH_SIZE_PX          = 6.0
DART_STROKE            = "#555555"
DART_WIDTH             = 0.75
DART_DASH              = "4,3"
LS_LINE_STROKE         = "#007700"   # lengthen/shorten = green
LS_LINE_WIDTH          = 0.75
LS_LINE_DASH           = "6,3"
LABEL_FONT             = "Arial, Helvetica, sans-serif"
LABEL_COLOR            = "#000000"
MARGIN_INCHES          = 0.5        # page margin around each piece


# ── SVGWriter ─────────────────────────────────────────────────────────────────

class SVGWriter:
    """
    Render a PatternPiece (or list of pieces) to SVG at real-world scale.

    Args:
        px_per_inch: Scale factor. Default 96px = 1 inch (SVG standard).
                     Use 72 for print-focused output matching points.
        include_seam_line: Draw inner seam line offset from boundary.
        include_grain_line: Draw grain line arrow.
        include_fold_line: Draw fold line indicator.
        include_notches: Draw notch marks.
        include_darts: Draw dart lines.
        include_labels: Draw piece name and metadata text.
        include_margin: Add margin around piece boundary.

    Usage:
        writer = SVGWriter()
        writer.save(piece, "output/pants_front.svg")

        # Multiple pieces on one sheet
        writer.save_sheet([front, back], "output/pants_all.svg")
    """

    def __init__(
        self,
        px_per_inch: float = PX_PER_INCH,
        include_seam_line: bool = True,
        include_grain_line: bool = True,
        include_fold_line: bool = True,
        include_notches: bool = True,
        include_darts: bool = True,
        include_labels: bool = True,
        include_margin: bool = True,
    ):
        self.ppi = px_per_inch
        self.include_seam_line = include_seam_line
        self.include_grain_line = include_grain_line
        self.include_fold_line = include_fold_line
        self.include_notches = include_notches
        self.include_darts = include_darts
        self.include_labels = include_labels
        self.margin_px = MARGIN_INCHES * px_per_inch if include_margin else 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, piece: PatternPiece, output_path: str | Path) -> Path:
        """
        Render a single PatternPiece to SVG and save to disk.

        Args:
            piece: PatternPiece with boundary_points populated.
            output_path: Where to write the .svg file.

        Returns:
            Path to written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dwg = self._render_piece(piece)
        dwg.saveas(str(output_path))
        return output_path

    def save_sheet(
        self,
        pieces: list[PatternPiece],
        output_path: str | Path,
        columns: int = 2,
    ) -> Path:
        """
        Render multiple PatternPieces onto one SVG sheet, arranged in columns.

        Args:
            pieces: List of PatternPieces to render.
            output_path: Where to write the .svg file.
            columns: Number of columns to arrange pieces in.

        Returns:
            Path to written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not pieces:
            raise ValueError("No pieces to render.")

        # Calculate individual piece dimensions
        piece_sizes = [self._piece_dimensions(p) for p in pieces]

        # Arrange in a grid
        rows = math.ceil(len(pieces) / columns)
        col_widths = [0.0] * columns
        row_heights = [0.0] * rows

        for i, (pw, ph) in enumerate(piece_sizes):
            col = i % columns
            row = i // columns
            col_widths[col] = max(col_widths[col], pw)
            row_heights[row] = max(row_heights[row], ph)

        total_width = sum(col_widths) + self.margin_px * (columns + 1)
        total_height = sum(row_heights) + self.margin_px * (rows + 1)

        dwg = svgwrite.Drawing(
            str(output_path),
            size=(f"{total_width:.1f}px", f"{total_height:.1f}px"),
            profile="full",
        )
        self._add_metadata(dwg, pieces[0] if pieces else None)

        # Render each piece into its grid position
        x_offset = self.margin_px
        y_offset = self.margin_px

        for i, piece in enumerate(pieces):
            col = i % columns
            row = i // columns

            if col == 0 and i > 0:
                y_offset += row_heights[row - 1] + self.margin_px
                x_offset = self.margin_px

            piece_group = self._render_piece_group(dwg, piece, x_offset, y_offset)
            dwg.add(piece_group)

            x_offset += col_widths[col] + self.margin_px

        dwg.save()
        return output_path

    def to_string(self, piece: PatternPiece) -> str:
        """Render a piece to SVG string without saving."""
        dwg = self._render_piece(piece)
        return dwg.tostring()

    # ── Core rendering ────────────────────────────────────────────────────────

    def _render_piece(self, piece: PatternPiece) -> Drawing:
        """Create a complete SVG Drawing for a single piece."""
        width_px, height_px = self._piece_dimensions(piece)
        canvas_w = width_px + self.margin_px * 2
        canvas_h = height_px + self.margin_px * 2

        dwg = svgwrite.Drawing(
            size=(f"{canvas_w:.1f}px", f"{canvas_h:.1f}px"),
            profile="full",
        )
        self._add_metadata(dwg, piece)
        self._add_styles(dwg)

        group = self._render_piece_group(dwg, piece, self.margin_px, self.margin_px)
        dwg.add(group)

        return dwg

    def _render_piece_group(
        self,
        dwg: Drawing,
        piece: PatternPiece,
        offset_x: float,
        offset_y: float,
    ) -> Group:
        """Render all elements of a piece into a positioned group."""
        group = dwg.g(id=f"piece_{piece.piece_number or 0}_{piece.name}")

        if not piece.boundary_points:
            return group

        # Compute transform — maps pattern inches to px with offset
        transform = self._build_transform(piece, offset_x, offset_y)

        # Layer order: seam line, boundary, grain, fold, darts, notches, labels
        if self.include_seam_line and piece.seam_allowance:
            group.add(self._draw_seam_line(dwg, piece, transform))

        group.add(self._draw_boundary(dwg, piece, transform))

        if self.include_grain_line and piece.grain_line:
            group.add(self._draw_grain_line(dwg, piece, transform))

        if self.include_fold_line and piece.fold_line:
            group.add(self._draw_fold_line(dwg, piece, transform))

        if self.include_darts and piece.darts:
            group.add(self._draw_darts(dwg, piece, transform))

        if self.include_notches and piece.notches:
            group.add(self._draw_notches(dwg, piece, transform))

        if self.include_labels:
            group.add(self._draw_labels(dwg, piece, transform))

        return group

    # ── Individual element renderers ──────────────────────────────────────────

    def _draw_boundary(
        self, dwg: Drawing, piece: PatternPiece, transform: "Transform"
    ) -> Group:
        """Draw the outer boundary of the pattern piece."""
        g = dwg.g(id="boundary", class_="boundary")

        pts = [transform.apply(p) for p in piece.boundary_points]
        path_data = self._points_to_path(pts, closed=True)

        g.add(dwg.path(
            d=path_data,
            fill="none",
            stroke=BOUNDARY_STROKE,
            stroke_width=BOUNDARY_WIDTH,
            stroke_linejoin="round",
            stroke_linecap="round",
        ))

        return g

    def _draw_seam_line(
        self, dwg: Drawing, piece: PatternPiece, transform: "Transform"
    ) -> Group:
        """
        Draw inner seam line — boundary inset by seam allowance.
        This is the actual sewing line.
        """
        g = dwg.g(id="seam_line", class_="seam-line")

        sa = piece.seam_allowance.global_inches if piece.seam_allowance else 0.625
        inset_pts = self._inset_polygon(piece.boundary_points, sa)

        if len(inset_pts) >= 3:
            pts = [transform.apply(p) for p in inset_pts]
            path_data = self._points_to_path(pts, closed=True)

            g.add(dwg.path(
                d=path_data,
                fill="none",
                stroke=SEAM_LINE_STROKE,
                stroke_width=SEAM_LINE_WIDTH,
                stroke_dasharray="5,3",
                stroke_linejoin="round",
            ))

        return g

    def _draw_grain_line(
        self, dwg: Drawing, piece: PatternPiece, transform: "Transform"
    ) -> Group:
        """Draw grain line with double-headed arrow."""
        g = dwg.g(id="grain_line", class_="grain-line")
        gl = piece.grain_line

        start = transform.apply(gl.start)
        end = transform.apply(gl.end)

        # Main line
        g.add(dwg.line(
            start=start,
            end=end,
            stroke=GRAIN_LINE_STROKE,
            stroke_width=GRAIN_LINE_WIDTH,
        ))

        # Arrowheads at both ends
        g.add(self._arrowhead(dwg, end, start, GRAIN_LINE_STROKE))     # start arrow
        g.add(self._arrowhead(dwg, start, end, GRAIN_LINE_STROKE))     # end arrow

        # Label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        g.add(dwg.text(
            "STRAIGHT GRAIN",
            insert=(mid_x, mid_y - 8),
            font_family=LABEL_FONT,
            font_size="8px",
            fill=GRAIN_LINE_STROKE,
            text_anchor="middle",
            letter_spacing="1",
        ))

        return g

    def _draw_fold_line(
        self, dwg: Drawing, piece: PatternPiece, transform: "Transform"
    ) -> Group:
        """Draw fold line — dashed, with FOLD LINE label."""
        g = dwg.g(id="fold_line", class_="fold-line")
        fl = piece.fold_line

        start = transform.apply(fl.start)
        end = transform.apply(fl.end)

        g.add(dwg.line(
            start=start,
            end=end,
            stroke=FOLD_LINE_STROKE,
            stroke_width=FOLD_LINE_WIDTH,
            stroke_dasharray=FOLD_LINE_DASH,
        ))

        # FOLD LINE label rotated along the line
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        angle = math.degrees(math.atan2(
            end[1] - start[1], end[0] - start[0]
        ))

        label = dwg.text(
            "FOLD LINE",
            insert=(mid_x, mid_y),
            font_family=LABEL_FONT,
            font_size="9px",
            fill=FOLD_LINE_STROKE,
            text_anchor="middle",
            transform=f"rotate({angle:.1f},{mid_x:.1f},{mid_y:.1f})",
        )
        label.update({"dy": "-4"})
        g.add(label)

        return g

    def _draw_notches(
        self, dwg: Drawing, piece: PatternPiece, transform: "Transform"
    ) -> Group:
        """Draw notch marks — small triangles pointing outward from boundary."""
        g = dwg.g(id="notches", class_="notches")

        for notch in piece.notches:
            px, py = transform.apply(notch.position)

            # Find boundary direction at this notch for orientation
            idx = notch.boundary_index
            pts = piece.boundary_points
            n = len(pts)

            if n >= 2:
                prev_pt = pts[(idx - 1) % n]
                next_pt = pts[(idx + 1) % n]
                # Normal vector pointing outward
                dx = next_pt[0] - prev_pt[0]
                dy = next_pt[1] - prev_pt[1]
                length = math.sqrt(dx * dx + dy * dy) + 1e-10
                # Perpendicular (outward normal)
                nx = -dy / length
                ny = dx / length
            else:
                nx, ny = 0, -1  # default: point up

            # Triangle notch
            tip_x = px + nx * NOTCH_SIZE_PX
            tip_y = py + ny * NOTCH_SIZE_PX
            left_x = px - ny * NOTCH_SIZE_PX * 0.4
            left_y = py + nx * NOTCH_SIZE_PX * 0.4
            right_x = px + ny * NOTCH_SIZE_PX * 0.4
            right_y = py - nx * NOTCH_SIZE_PX * 0.4

            notch_pts = [
                (tip_x, tip_y),
                (left_x, left_y),
                (right_x, right_y),
            ]

            if notch.notch_type == "double":
                # Two notches side by side
                offset = NOTCH_SIZE_PX * 0.6
                for sign in [-1, 1]:
                    ox = px + sign * ny * offset
                    oy = py - sign * nx * offset
                    t_x = ox + nx * NOTCH_SIZE_PX
                    t_y = oy + ny * NOTCH_SIZE_PX
                    double_pts = [
                        (t_x, t_y),
                        (ox - ny * NOTCH_SIZE_PX * 0.3, oy + nx * NOTCH_SIZE_PX * 0.3),
                        (ox + ny * NOTCH_SIZE_PX * 0.3, oy - nx * NOTCH_SIZE_PX * 0.3),
                    ]
                    g.add(dwg.polygon(
                        points=double_pts,
                        fill=NOTCH_FILL,
                        stroke=NOTCH_STROKE,
                        stroke_width=0.5,
                    ))
            else:
                g.add(dwg.polygon(
                    points=notch_pts,
                    fill=NOTCH_FILL,
                    stroke=NOTCH_STROKE,
                    stroke_width=0.5,
                ))

        return g

    def _draw_darts(
        self, dwg: Drawing, piece: PatternPiece, transform: "Transform"
    ) -> Group:
        """Draw dart lines — two dashed legs meeting at apex."""
        g = dwg.g(id="darts", class_="darts")

        for i, dart in enumerate(piece.darts):
            apex = transform.apply(dart.apex)
            leg1 = transform.apply(dart.leg_start)
            leg2 = transform.apply(dart.leg_end)

            # Leg 1
            g.add(dwg.line(
                start=leg1,
                end=apex,
                stroke=DART_STROKE,
                stroke_width=DART_WIDTH,
                stroke_dasharray=DART_DASH,
            ))

            # Leg 2
            g.add(dwg.line(
                start=leg2,
                end=apex,
                stroke=DART_STROKE,
                stroke_width=DART_WIDTH,
                stroke_dasharray=DART_DASH,
            ))

            # Apex dot
            g.add(dwg.circle(
                center=apex,
                r=2,
                fill=DART_STROKE,
            ))

        return g

    def _draw_labels(
        self, dwg: Drawing, piece: PatternPiece, transform: "Transform"
    ) -> Group:
        """Draw piece name, number, cut quantity, size, and seam allowance."""
        g = dwg.g(id="labels", class_="labels")

        # Find label position — centroid of boundary
        if not piece.boundary_points:
            return g

        cx = sum(p[0] for p in piece.boundary_points) / len(piece.boundary_points)
        cy = sum(p[1] for p in piece.boundary_points) / len(piece.boundary_points)
        label_x, label_y = transform.apply((cx, cy))

        line_height = 16
        y = label_y - line_height * 2

        # Pattern brand and ID
        if piece.pattern_brand and piece.pattern_brand != "unknown":
            g.add(self._label_text(dwg, piece.pattern_brand, label_x, y, size="10px"))
            y += line_height * 0.8

        # Piece name — large
        g.add(self._label_text(
            dwg, piece.name, label_x, y, size="14px", bold=True
        ))
        y += line_height

        # Piece number and cut quantity
        if piece.piece_number is not None:
            cut_text = f"Cut {piece.cut_quantity}"
            if piece.is_cut_on_fold:
                cut_text += " on fold"
            g.add(self._label_text(
                dwg, f"#{piece.piece_number} — {cut_text}", label_x, y, size="10px"
            ))
            y += line_height * 0.9

        # Size label
        if piece.size_label:
            g.add(self._label_text(dwg, piece.size_label, label_x, y, size="10px"))
            y += line_height * 0.9

        # Seam allowance
        if piece.seam_allowance:
            sa = piece.seam_allowance.global_inches
            sa_text = f"Seam allowance: {sa}\""
            g.add(self._label_text(
                dwg, sa_text, label_x, y, size="9px", color="#555555"
            ))

        return g

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _build_transform(
        self,
        piece: PatternPiece,
        offset_x: float,
        offset_y: float,
    ) -> "Transform":
        """
        Build a Transform that maps pattern inches to SVG pixels,
        with the piece positioned at (offset_x, offset_y).
        """
        if not piece.boundary_points:
            return Transform(self.ppi, 0, 0)

        xs = [p[0] for p in piece.boundary_points]
        ys = [p[1] for p in piece.boundary_points]
        min_x, min_y = min(xs), min(ys)

        # Translate so piece starts at offset
        tx = offset_x - min_x * self.ppi
        ty = offset_y - min_y * self.ppi

        return Transform(self.ppi, tx, ty)

    def _piece_dimensions(self, piece: PatternPiece) -> tuple[float, float]:
        """Return (width_px, height_px) for a piece's bounding box."""
        if not piece.boundary_points:
            return (200.0, 200.0)

        xs = [p[0] for p in piece.boundary_points]
        ys = [p[1] for p in piece.boundary_points]

        width = (max(xs) - min(xs)) * self.ppi
        height = (max(ys) - min(ys)) * self.ppi
        return (width, height)

    def _points_to_path(self, pts: list[tuple], closed: bool = True) -> str:
        """Convert list of (x, y) px points to SVG path data string."""
        if not pts:
            return ""
        parts = [f"M {pts[0][0]:.2f},{pts[0][1]:.2f}"]
        for x, y in pts[1:]:
            parts.append(f"L {x:.2f},{y:.2f}")
        if closed:
            parts.append("Z")
        return " ".join(parts)

    def _inset_polygon(
        self, points: list[Point], inset_inches: float
    ) -> list[Point]:
        """
        Inset a polygon by inset_inches (for seam line).
        Uses simple normal-based inset — works well for convex shapes.
        """
        import numpy as np

        if len(points) < 3:
            return points

        pts = np.array(points)
        n = len(pts)
        inset_px = inset_inches  # stays in inches — transform applied later

        inset_pts = []
        for i in range(n):
            prev = pts[(i - 1) % n]
            curr = pts[i]
            next_pt = pts[(i + 1) % n]

            # Edge vectors
            v1 = curr - prev
            v2 = next_pt - curr

            # Normals (inward pointing)
            def inward_normal(v):
                length = np.linalg.norm(v)
                if length < 1e-10:
                    return np.array([0.0, 0.0])
                return np.array([v[1], -v[0]]) / length

            n1 = inward_normal(v1)
            n2 = inward_normal(v2)
            avg_normal = n1 + n2
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-10:
                avg_normal /= norm

            inset_pt = curr + avg_normal * inset_px
            inset_pts.append((float(inset_pt[0]), float(inset_pt[1])))

        return inset_pts

    def _arrowhead(
        self,
        dwg: Drawing,
        from_pt: tuple,
        to_pt: tuple,
        color: str,
        size: float = 8.0,
    ) -> svgwrite.base.BaseElement:
        """Draw a small arrowhead at to_pt pointing away from from_pt."""
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        length = math.sqrt(dx * dx + dy * dy) + 1e-10
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular

        tip = to_pt
        left = (
            to_pt[0] - ux * size - px * size * 0.4,
            to_pt[1] - uy * size - py * size * 0.4,
        )
        right = (
            to_pt[0] - ux * size + px * size * 0.4,
            to_pt[1] - uy * size + py * size * 0.4,
        )

        return dwg.polygon(points=[tip, left, right], fill=color)

    def _label_text(
        self,
        dwg: Drawing,
        text: str,
        x: float,
        y: float,
        size: str = "12px",
        bold: bool = False,
        color: str = LABEL_COLOR,
    ) -> svgwrite.text.Text:
        weight = "bold" if bold else "normal"
        return dwg.text(
            text,
            insert=(x, y),
            font_family=LABEL_FONT,
            font_size=size,
            font_weight=weight,
            fill=color,
            text_anchor="middle",
        )

    # ── SVG metadata ──────────────────────────────────────────────────────────

    def _add_metadata(
        self, dwg: Drawing, piece: Optional[PatternPiece]
    ) -> None:
        """Add SVG title and description metadata."""
        if piece:
            dwg.set_desc(
                title=f"PatternBridge — {piece.name}",
                desc=(
                    f"Pattern: {piece.pattern_brand} {piece.pattern_id} | "
                    f"Piece: {piece.name} #{piece.piece_number} | "
                    f"Size: {piece.size_label} | "
                    f"Generated by PatternBridge"
                ),
            )

    def _add_styles(self, dwg: Drawing) -> None:
        """Add CSS styles for print media."""
        dwg.defs.add(dwg.style("""
            @media print {
                .boundary { stroke-width: 1pt; }
                .seam-line { stroke-width: 0.5pt; }
                .grain-line { stroke-width: 0.75pt; }
                .fold-line { stroke-width: 1pt; }
                .labels text { font-size: 10pt; }
            }
            @media screen {
                svg { background: white; }
            }
        """))


# ── Transform helper ──────────────────────────────────────────────────────────

class Transform:
    """
    Maps pattern coordinates (inches) to SVG pixels.
    Applies scale and translation.
    """

    def __init__(self, ppi: float, tx: float, ty: float):
        self.ppi = ppi
        self.tx = tx
        self.ty = ty

    def apply(self, point: Point) -> tuple[float, float]:
        """Transform (x_inches, y_inches) → (x_px, y_px)."""
        return (
            point[0] * self.ppi + self.tx,
            point[1] * self.ppi + self.ty,
        )

    def apply_scalar(self, inches: float) -> float:
        """Scale a length in inches to pixels."""
        return inches * self.ppi
