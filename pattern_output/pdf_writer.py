"""
PDFWriter: Converts PatternPiece(s) to tiled, printable PDF.

Tiles large pattern pieces across multiple letter or A4 pages
with registration marks and overlap zones so you can print at home,
cut the pages, and tape them together into a full-size pattern.

Uses reportlab for PDF generation.

Tiling strategy:
    - Each page has a printable area (page size minus margins)
    - Pattern is divided into a grid of tiles
    - Each tile overlaps neighbors by OVERLAP_INCHES for alignment
    - Registration marks (crosshairs) appear in each corner
    - Page coordinates printed on each tile (e.g. "Row 2, Col 3")
    - A cover page shows the full assembly diagram
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

try:
    from reportlab.lib.pagesizes import LETTER, A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import simpleSplit
except ImportError:
    raise ImportError(
        "reportlab is required for PDFWriter. "
        "Install with: pip install reportlab"
    )

from ..pattern_geometry.piece import PatternPiece, Point


# ── Constants ─────────────────────────────────────────────────────────────────

# Page sizes (width, height) in points — reportlab uses points (1pt = 1/72 inch)
PAGE_LETTER = LETTER          # 612 x 792 pt  = 8.5 x 11 inch
PAGE_A4     = A4              # 595 x 842 pt  = ~8.27 x 11.69 inch

# Page margins in inches
MARGIN_TOP    = 0.5
MARGIN_BOTTOM = 0.5
MARGIN_LEFT   = 0.5
MARGIN_RIGHT  = 0.5

# Tile overlap in inches — how much adjacent pages overlap for alignment
OVERLAP_INCHES = 0.5

# Registration mark size in points
REG_MARK_SIZE = 12.0

# Scale: 1 inch of pattern = 1 inch on paper (full scale)
PATTERN_SCALE = 1.0

# Font
FONT_BODY  = "Helvetica"
FONT_BOLD  = "Helvetica-Bold"
FONT_SMALL = 7
FONT_MED   = 9
FONT_LARGE = 12


# ── PDFWriter ─────────────────────────────────────────────────────────────────

class PDFWriter:
    """
    Render a PatternPiece (or list of pieces) to a tiled PDF
    suitable for home printing and assembly.

    Args:
        page_size: PAGE_LETTER or PAGE_A4 (or custom tuple in points).
        include_cover: Add an assembly diagram cover page.
        include_seam_line: Draw inner seam/sewing line.
        include_grain_line: Draw grain line.
        include_fold_line: Draw fold line.
        include_notches: Draw notch marks.
        include_labels: Draw piece labels.
        overlap_inches: How much adjacent tiles overlap.

    Usage:
        writer = PDFWriter()
        writer.save(piece, "output/pants_front.pdf")
        writer.save_all([front, back], "output/pants_complete.pdf")
    """

    def __init__(
        self,
        page_size: tuple = PAGE_LETTER,
        include_cover: bool = True,
        include_seam_line: bool = True,
        include_grain_line: bool = True,
        include_fold_line: bool = True,
        include_notches: bool = True,
        include_labels: bool = True,
        overlap_inches: float = OVERLAP_INCHES,
    ):
        self.page_size = page_size
        self.include_cover = include_cover
        self.include_seam_line = include_seam_line
        self.include_grain_line = include_grain_line
        self.include_fold_line = include_fold_line
        self.include_notches = include_notches
        self.include_labels = include_labels
        self.overlap = overlap_inches

        # Compute printable area in inches
        page_w_pt, page_h_pt = page_size
        self.page_w_in = page_w_pt / 72.0
        self.page_h_in = page_h_pt / 72.0
        self.print_w_in = self.page_w_in - MARGIN_LEFT - MARGIN_RIGHT
        self.print_h_in = self.page_h_in - MARGIN_TOP  - MARGIN_BOTTOM

        # Effective tile area (minus overlap on right/bottom shared with neighbor)
        self.tile_w_in = self.print_w_in - self.overlap
        self.tile_h_in = self.print_h_in - self.overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, piece: PatternPiece, output_path: str | Path) -> Path:
        """
        Render a single PatternPiece to a tiled PDF.

        Args:
            piece: PatternPiece with boundary_points populated.
            output_path: Where to write the .pdf file.

        Returns:
            Path to written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        canvas = Canvas(str(output_path), pagesize=self.page_size)
        self._render_piece_to_canvas(canvas, piece)
        canvas.save()
        return output_path

    def save_all(
        self,
        pieces: list[PatternPiece],
        output_path: str | Path,
    ) -> Path:
        """
        Render multiple PatternPieces into a single PDF.
        Each piece gets its own tiled section, with an optional cover page
        showing all pieces and their tile coordinates.

        Args:
            pieces: List of PatternPieces to render.
            output_path: Where to write the .pdf file.

        Returns:
            Path to written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        canvas = Canvas(str(output_path), pagesize=self.page_size)

        if self.include_cover:
            self._render_cover_page(canvas, pieces)

        for piece in pieces:
            self._render_piece_to_canvas(canvas, piece)

        canvas.save()
        return output_path

    # ── Piece rendering ───────────────────────────────────────────────────────

    def _render_piece_to_canvas(
        self, canvas: Canvas, piece: PatternPiece
    ) -> None:
        """
        Tile a single piece across as many pages as needed.
        Each page gets registration marks, tile coordinates, and
        assembly instructions.
        """
        if not piece.boundary_points:
            return

        # Piece bounding box in inches
        bounds = self._piece_bounds(piece)
        piece_w = bounds["max_x"] - bounds["min_x"]
        piece_h = bounds["max_y"] - bounds["min_y"]

        # How many tiles needed
        cols = max(1, math.ceil(piece_w / self.tile_w_in))
        rows = max(1, math.ceil(piece_h / self.tile_h_in))

        # Render each tile
        for row in range(rows):
            for col in range(cols):
                self._render_tile(canvas, piece, bounds, row, col, rows, cols)
                canvas.showPage()

    def _render_tile(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        bounds: dict,
        row: int,
        col: int,
        total_rows: int,
        total_cols: int,
    ) -> None:
        """Render one tile of a pattern piece onto the current canvas page."""

        # Origin of this tile in pattern space (inches)
        tile_origin_x = bounds["min_x"] + col * self.tile_w_in
        tile_origin_y = bounds["min_y"] + row * self.tile_h_in

        # Save canvas state
        canvas.saveState()

        # Set clip region to printable area
        print_x = MARGIN_LEFT * inch
        print_y = MARGIN_BOTTOM * inch
        print_w = self.print_w_in * inch
        print_h = self.print_h_in * inch
        canvas.clipPath(
            canvas.beginPath(),
            stroke=0, fill=0
        )

        # ── Draw registration marks ────────────────────────────────────────
        self._draw_registration_marks(canvas)

        # ── Draw tile boundary guide (light dotted box) ────────────────────
        canvas.setStrokeColor(colors.Color(0.8, 0.8, 0.8))
        canvas.setLineWidth(0.5)
        canvas.setDash(3, 3)
        canvas.rect(
            MARGIN_LEFT * inch,
            MARGIN_BOTTOM * inch,
            self.print_w_in * inch,
            self.print_h_in * inch,
        )
        canvas.setDash()  # reset dash

        # ── Draw overlap zone indicators ───────────────────────────────────
        self._draw_overlap_indicators(canvas, row, col, total_rows, total_cols)

        # ── Transform: pattern inches → page points ────────────────────────
        # Translate so tile_origin maps to page margin
        offset_x = (MARGIN_LEFT - tile_origin_x) * inch
        offset_y_base = MARGIN_BOTTOM * inch
        # PDF y-axis is bottom-up; pattern y-axis is top-down
        # Flip y: page_y = page_h - pattern_y * inch + offset
        # We'll handle this in coordinate conversion

        def to_page(pt: Point) -> tuple[float, float]:
            """Convert pattern (x, y) inches to page (x, y) points."""
            px = (pt[0] - tile_origin_x + MARGIN_LEFT) * inch
            # Flip y: pattern top = page top
            py = (self.page_h_in - MARGIN_TOP - (pt[1] - tile_origin_y)) * inch
            return px, py

        # ── Draw pattern boundary ──────────────────────────────────────────
        self._draw_boundary_pdf(canvas, piece, to_page)

        # ── Draw seam line ─────────────────────────────────────────────────
        if self.include_seam_line and piece.seam_allowance:
            self._draw_seam_line_pdf(canvas, piece, to_page)

        # ── Draw grain line ────────────────────────────────────────────────
        if self.include_grain_line and piece.grain_line:
            self._draw_grain_line_pdf(canvas, piece, to_page)

        # ── Draw fold line ─────────────────────────────────────────────────
        if self.include_fold_line and piece.fold_line:
            self._draw_fold_line_pdf(canvas, piece, to_page)

        # ── Draw notches ───────────────────────────────────────────────────
        if self.include_notches and piece.notches:
            self._draw_notches_pdf(canvas, piece, to_page)

        # ── Draw labels (only on first tile) ──────────────────────────────
        if self.include_labels and row == 0 and col == 0:
            self._draw_labels_pdf(canvas, piece, to_page)

        # ── Draw tile info header ──────────────────────────────────────────
        self._draw_tile_header(
            canvas, piece, row, col, total_rows, total_cols
        )

        canvas.restoreState()

    # ── PDF element drawing ───────────────────────────────────────────────────

    def _draw_boundary_pdf(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        to_page,
    ) -> None:
        """Draw outer boundary of pattern piece."""
        if len(piece.boundary_points) < 2:
            return

        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1.5)
        canvas.setLineCap(1)   # round
        canvas.setLineJoin(1)  # round

        path = canvas.beginPath()
        pts = [to_page(p) for p in piece.boundary_points]
        path.moveTo(*pts[0])
        for pt in pts[1:]:
            path.lineTo(*pt)
        path.close()
        canvas.drawPath(path, stroke=1, fill=0)

    def _draw_seam_line_pdf(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        to_page,
    ) -> None:
        """Draw inner sewing line (dashed)."""
        sa = piece.seam_allowance.global_inches if piece.seam_allowance else 0.625
        inset_pts = self._inset_polygon(piece.boundary_points, sa)

        if len(inset_pts) < 3:
            return

        canvas.setStrokeColor(colors.Color(0.3, 0.3, 0.3))
        canvas.setLineWidth(0.75)
        canvas.setDash(5, 3)

        path = canvas.beginPath()
        pts = [to_page(p) for p in inset_pts]
        path.moveTo(*pts[0])
        for pt in pts[1:]:
            path.lineTo(*pt)
        path.close()
        canvas.drawPath(path, stroke=1, fill=0)
        canvas.setDash()

    def _draw_grain_line_pdf(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        to_page,
    ) -> None:
        """Draw grain line with arrows."""
        gl = piece.grain_line
        start = to_page(gl.start)
        end = to_page(gl.end)

        canvas.setStrokeColor(colors.Color(0, 0.33, 0.67))
        canvas.setLineWidth(1.0)
        canvas.line(*start, *end)

        # Arrowheads
        self._draw_arrow_pdf(canvas, end, start, colors.Color(0, 0.33, 0.67))
        self._draw_arrow_pdf(canvas, start, end, colors.Color(0, 0.33, 0.67))

        # Label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        canvas.setFillColor(colors.Color(0, 0.33, 0.67))
        canvas.setFont(FONT_BODY, FONT_SMALL)
        canvas.drawCentredString(mid_x, mid_y + 6, "STRAIGHT GRAIN")

    def _draw_fold_line_pdf(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        to_page,
    ) -> None:
        """Draw fold line (dashed, colored)."""
        fl = piece.fold_line
        start = to_page(fl.start)
        end = to_page(fl.end)

        canvas.setStrokeColor(colors.Color(0.67, 0, 0.33))
        canvas.setLineWidth(1.5)
        canvas.setDash(8, 4)
        canvas.line(*start, *end)
        canvas.setDash()

        # Label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        canvas.setFillColor(colors.Color(0.67, 0, 0.33))
        canvas.setFont(FONT_BOLD, FONT_SMALL)
        canvas.drawCentredString(mid_x, mid_y + 6, "FOLD LINE")

    def _draw_notches_pdf(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        to_page,
    ) -> None:
        """Draw notch triangles."""
        pts = piece.boundary_points
        n = len(pts)
        size_pt = 6.0  # points

        for notch in piece.notches:
            px, py = to_page(notch.position)
            idx = notch.boundary_index

            if n >= 2:
                prev_pt = pts[(idx - 1) % n]
                next_pt = pts[(idx + 1) % n]
                dx = next_pt[0] - prev_pt[0]
                dy = next_pt[1] - prev_pt[1]
                length = math.sqrt(dx * dx + dy * dy) + 1e-10
                # Outward normal (flipped y for PDF)
                nx = dy / length
                ny = -dx / length
            else:
                nx, ny = 0, 1

            # Triangle
            tip_x = px + nx * size_pt
            tip_y = py + ny * size_pt
            left_x = px - ny * size_pt * 0.4
            left_y = py + nx * size_pt * 0.4
            right_x = px + ny * size_pt * 0.4
            right_y = py - nx * size_pt * 0.4

            canvas.setFillColor(colors.black)
            canvas.setStrokeColor(colors.black)
            path = canvas.beginPath()
            path.moveTo(tip_x, tip_y)
            path.lineTo(left_x, left_y)
            path.lineTo(right_x, right_y)
            path.close()
            canvas.drawPath(path, stroke=1, fill=1)

    def _draw_labels_pdf(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        to_page,
    ) -> None:
        """Draw piece name, number, cut quantity, seam allowance."""
        if not piece.boundary_points:
            return

        # Centroid
        cx = sum(p[0] for p in piece.boundary_points) / len(piece.boundary_points)
        cy = sum(p[1] for p in piece.boundary_points) / len(piece.boundary_points)
        lx, ly = to_page((cx, cy))

        canvas.setFillColor(colors.black)
        line_h = 14

        y = ly + line_h * 1.5

        # Brand
        if piece.pattern_brand and piece.pattern_brand != "unknown":
            canvas.setFont(FONT_BODY, FONT_SMALL)
            canvas.drawCentredString(lx, y, piece.pattern_brand)
            y -= line_h * 0.7

        # Piece name
        canvas.setFont(FONT_BOLD, FONT_LARGE)
        canvas.drawCentredString(lx, y, piece.name)
        y -= line_h

        # Number + cut
        if piece.piece_number is not None:
            cut_str = f"Cut {piece.cut_quantity}"
            if piece.is_cut_on_fold:
                cut_str += " on fold"
            canvas.setFont(FONT_BODY, FONT_MED)
            canvas.drawCentredString(
                lx, y, f"#{piece.piece_number}  —  {cut_str}"
            )
            y -= line_h * 0.9

        # Size
        if piece.size_label:
            canvas.setFont(FONT_BODY, FONT_MED)
            canvas.drawCentredString(lx, y, piece.size_label)
            y -= line_h * 0.9

        # Seam allowance
        if piece.seam_allowance:
            canvas.setFont(FONT_BODY, FONT_SMALL)
            canvas.setFillColor(colors.Color(0.3, 0.3, 0.3))
            canvas.drawCentredString(
                lx, y,
                f"Seam allowance: {piece.seam_allowance.global_inches}\""
            )

    # ── Registration marks ────────────────────────────────────────────────────

    def _draw_registration_marks(self, canvas: Canvas) -> None:
        """
        Draw crosshair registration marks in all four corners.
        These align pages during assembly.
        """
        s = REG_MARK_SIZE
        margin_x = MARGIN_LEFT * inch
        margin_y = MARGIN_BOTTOM * inch
        w = self.page_size[0]
        h = self.page_size[1]

        corners = [
            (margin_x,       margin_y),
            (w - margin_x,   margin_y),
            (margin_x,       h - margin_y),
            (w - margin_x,   h - margin_y),
        ]

        canvas.setStrokeColor(colors.Color(0.5, 0.5, 0.5))
        canvas.setLineWidth(0.5)

        for cx, cy in corners:
            # Horizontal bar
            canvas.line(cx - s, cy, cx + s, cy)
            # Vertical bar
            canvas.line(cx, cy - s, cx, cy + s)
            # Circle
            canvas.circle(cx, cy, s * 0.4, stroke=1, fill=0)

    def _draw_overlap_indicators(
        self,
        canvas: Canvas,
        row: int,
        col: int,
        total_rows: int,
        total_cols: int,
    ) -> None:
        """
        Draw overlap zone shading on the right and bottom edges.
        Indicates where this page overlaps with the next page
        during assembly.
        """
        overlap_pt = self.overlap * inch
        w = self.page_size[0]
        h = self.page_size[1]

        canvas.setFillColor(colors.Color(0.95, 0.95, 1.0, 0.5))
        canvas.setStrokeColor(colors.Color(0.7, 0.7, 0.9))
        canvas.setLineWidth(0.5)
        canvas.setDash(4, 2)

        # Right overlap zone (if not last column)
        if col < total_cols - 1:
            right_x = w - MARGIN_RIGHT * inch - overlap_pt
            canvas.rect(
                right_x,
                MARGIN_BOTTOM * inch,
                overlap_pt,
                self.print_h_in * inch,
                stroke=1, fill=1,
            )

        # Bottom overlap zone in PDF = top in pattern space
        # (if not last row)
        if row < total_rows - 1:
            bottom_y = MARGIN_BOTTOM * inch
            canvas.rect(
                MARGIN_LEFT * inch,
                bottom_y,
                self.print_w_in * inch,
                overlap_pt,
                stroke=1, fill=1,
            )

        canvas.setDash()

    def _draw_tile_header(
        self,
        canvas: Canvas,
        piece: PatternPiece,
        row: int,
        col: int,
        total_rows: int,
        total_cols: int,
    ) -> None:
        """
        Draw tile coordinate and assembly instructions at top of page.
        e.g. "FRONT #1  |  Page 3 of 6  |  Row 1, Col 3  |  Align ▶ with next page"
        """
        page_num = row * total_cols + col + 1
        total_pages = total_rows * total_cols

        header_y = self.page_size[1] - (MARGIN_TOP * inch * 0.6)

        canvas.setFillColor(colors.black)
        canvas.setFont(FONT_BOLD, FONT_SMALL)

        left_text = f"{piece.name}"
        if piece.piece_number:
            left_text += f" #{piece.piece_number}"
        canvas.drawString(MARGIN_LEFT * inch, header_y, left_text)

        center_text = f"Page {page_num} of {total_pages}  |  Row {row+1}, Col {col+1}"
        canvas.setFont(FONT_BODY, FONT_SMALL)
        canvas.drawCentredString(self.page_size[0] / 2, header_y, center_text)

        # Assembly hint
        hints = []
        if col < total_cols - 1:
            hints.append("align right edge ▶")
        if row < total_rows - 1:
            hints.append("align bottom edge ▼")

        if hints:
            canvas.setFont(FONT_BODY, FONT_SMALL)
            canvas.setFillColor(colors.Color(0.4, 0.4, 0.4))
            canvas.drawRightString(
                self.page_size[0] - MARGIN_RIGHT * inch,
                header_y,
                "  |  ".join(hints),
            )

        # Thin rule under header
        canvas.setStrokeColor(colors.Color(0.7, 0.7, 0.7))
        canvas.setLineWidth(0.5)
        rule_y = header_y - 4
        canvas.line(
            MARGIN_LEFT * inch, rule_y,
            self.page_size[0] - MARGIN_RIGHT * inch, rule_y,
        )

    # ── Cover page ────────────────────────────────────────────────────────────

    def _render_cover_page(
        self, canvas: Canvas, pieces: list[PatternPiece]
    ) -> None:
        """
        Render a cover page showing:
        - Project title
        - List of pieces with tile counts
        - Assembly overview
        - Measurement profile if available
        """
        w, h = self.page_size
        y = h - 1.5 * inch

        # Title
        canvas.setFont(FONT_BOLD, 18)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(w / 2, y, "PatternBridge")
        y -= 0.35 * inch

        canvas.setFont(FONT_BODY, FONT_MED)
        canvas.setFillColor(colors.Color(0.3, 0.3, 0.3))
        canvas.drawCentredString(
            w / 2, y, "Custom Pattern — Print and Assemble"
        )
        y -= 0.5 * inch

        # Divider
        canvas.setStrokeColor(colors.Color(0.8, 0.8, 0.8))
        canvas.setLineWidth(1)
        canvas.line(MARGIN_LEFT * inch, y, w - MARGIN_RIGHT * inch, y)
        y -= 0.35 * inch

        # Pieces table
        canvas.setFont(FONT_BOLD, FONT_MED)
        canvas.setFillColor(colors.black)
        canvas.drawString(MARGIN_LEFT * inch, y, "Pattern Pieces:")
        y -= 0.3 * inch

        for piece in pieces:
            bounds = self._piece_bounds(piece)
            pw = bounds["max_x"] - bounds["min_x"]
            ph = bounds["max_y"] - bounds["min_y"]
            cols = max(1, math.ceil(pw / self.tile_w_in))
            rows = max(1, math.ceil(ph / self.tile_h_in))
            pages = rows * cols

            fold_note = " (cut on fold)" if piece.is_cut_on_fold else ""
            line = (
                f"  {piece.name}"
                f"  —  Cut {piece.cut_quantity}{fold_note}"
                f"  —  {pw:.1f}\" × {ph:.1f}\""
                f"  —  {pages} page{'s' if pages > 1 else ''}"
                f"  ({rows} row × {cols} col)"
            )

            canvas.setFont(FONT_BODY, FONT_MED)
            canvas.setFillColor(colors.black)
            canvas.drawString(MARGIN_LEFT * inch, y, line)
            y -= 0.28 * inch

            # Seam allowance note
            if piece.seam_allowance:
                sa_note = (
                    f"      Seam allowance: {piece.seam_allowance.global_inches}\""
                )
                canvas.setFont(FONT_BODY, FONT_SMALL)
                canvas.setFillColor(colors.Color(0.4, 0.4, 0.4))
                canvas.drawString(MARGIN_LEFT * inch, y, sa_note)
                y -= 0.22 * inch

            y -= 0.08 * inch

        # Divider
        y -= 0.2 * inch
        canvas.setStrokeColor(colors.Color(0.8, 0.8, 0.8))
        canvas.line(MARGIN_LEFT * inch, y, w - MARGIN_RIGHT * inch, y)
        y -= 0.35 * inch

        # Assembly instructions
        canvas.setFont(FONT_BOLD, FONT_MED)
        canvas.setFillColor(colors.black)
        canvas.drawString(MARGIN_LEFT * inch, y, "Assembly Instructions:")
        y -= 0.3 * inch

        instructions = [
            "1. Print all pages at 100% scale (do NOT scale to fit page).",
            "2. Verify scale: measure the 1\" reference square on page 1.",
            f"3. Each page overlaps {self.overlap}\" with adjacent pages.",
            "4. Trim the right and bottom overlap zones from each page.",
            "5. Align registration marks (crosshairs in corners) when joining pages.",
            "6. Tape or glue pages together before cutting fabric.",
            "7. The dashed inner line is the sewing line.",
            "8. The solid outer line is the cutting line (includes seam allowance).",
            "9. Cut on fold pieces: place fold line edge on folded fabric edge.",
        ]

        canvas.setFont(FONT_BODY, FONT_SMALL)
        canvas.setFillColor(colors.black)
        for instruction in instructions:
            canvas.drawString(MARGIN_LEFT * inch, y, instruction)
            y -= 0.22 * inch

        # Measurement profile if available
        if pieces and pieces[0].target_measurements:
            y -= 0.3 * inch
            canvas.setStrokeColor(colors.Color(0.8, 0.8, 0.8))
            canvas.line(MARGIN_LEFT * inch, y, w - MARGIN_RIGHT * inch, y)
            y -= 0.35 * inch

            canvas.setFont(FONT_BOLD, FONT_MED)
            canvas.setFillColor(colors.black)
            canvas.drawString(MARGIN_LEFT * inch, y, "Target Measurements:")
            y -= 0.3 * inch

            meas = pieces[0].target_measurements
            ease = pieces[0].ease_allowances

            for key, val in meas.items():
                ease_val = ease.get(key, 0.0)
                ease_note = f" + {ease_val}\" ease" if ease_val > 0 else ""
                line = f"  {key.replace('_', ' ').title()}: {val}\"  {ease_note}"
                canvas.setFont(FONT_BODY, FONT_SMALL)
                canvas.setFillColor(colors.black)
                canvas.drawString(MARGIN_LEFT * inch, y, line)
                y -= 0.22 * inch

        # PatternBridge footer
        canvas.setFont(FONT_BODY, FONT_SMALL)
        canvas.setFillColor(colors.Color(0.6, 0.6, 0.6))
        canvas.drawCentredString(
            w / 2,
            MARGIN_BOTTOM * inch,
            "Generated by PatternBridge — github.com/JinnZ2/PatternBridge",
        )

        canvas.showPage()

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _piece_bounds(self, piece: PatternPiece) -> dict:
        """Return bounding box of piece in inches."""
        if not piece.boundary_points:
            return {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10}
        xs = [p[0] for p in piece.boundary_points]
        ys = [p[1] for p in piece.boundary_points]
        return {
            "min_x": min(xs), "max_x": max(xs),
            "min_y": min(ys), "max_y": max(ys),
        }

    def _inset_polygon(
        self, points: list[Point], inset_inches: float
    ) -> list[Point]:
        """Inset polygon by inset_inches for seam line."""
        import numpy as np

        if len(points) < 3:
            return points

        pts = np.array(points)
        n = len(pts)
        inset_pts = []

        for i in range(n):
            prev = pts[(i - 1) % n]
            curr = pts[i]
            nxt  = pts[(i + 1) % n]

            v1 = curr - prev
            v2 = nxt  - curr

            def inward_normal(v):
                l = np.linalg.norm(v)
                if l < 1e-10:
                    return np.zeros(2)
                return np.array([v[1], -v[0]]) / l

            n1 = inward_normal(v1)
            n2 = inward_normal(v2)
            avg = n1 + n2
            norm = np.linalg.norm(avg)
            if norm > 1e-10:
                avg /= norm

            inset_pts.append(tuple((curr + avg * inset_inches).tolist()))

        return inset_pts

    def _draw_arrow_pdf(
        self,
        canvas: Canvas,
        from_pt: tuple,
        to_pt: tuple,
        color,
        size: float = 6.0,
    ) -> None:
        """Draw arrowhead at to_pt."""
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        length = math.sqrt(dx * dx + dy * dy) + 1e-10
        ux, uy = dx / length, dy / length
        px, py = -uy, ux

        tip   = to_pt
        left  = (to_pt[0] - ux*size - px*size*0.4,
                 to_pt[1] - uy*size - py*size*0.4)
        right = (to_pt[0] - ux*size + px*size*0.4,
                 to_pt[1] - uy*size + py*size*0.4)

        canvas.setFillColor(color)
        path = canvas.beginPath()
        path.moveTo(*tip)
        path.lineTo(*left)
        path.lineTo(*right)
        path.close()
        canvas.drawPath(path, stroke=0, fill=1)
