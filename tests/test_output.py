"""Tests for the pattern_output layer: SVG and PDF writers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from pattern_geometry.piece import PatternPiece
from pattern_output.svg_writer import SVGWriter, PX_PER_INCH, MM_PER_INCH, CM_PER_INCH
from pattern_output.pdf_writer import PDFWriter, PAGE_LETTER, PAGE_A4


# ── SVGWriter ────────────────────────────────────────────────────────────────


class TestSVGWriter:
    def test_create_default(self):
        writer = SVGWriter()
        assert writer is not None

    def test_create_custom_ppi(self):
        writer = SVGWriter(px_per_inch=72.0)
        assert writer is not None

    def test_to_string_minimal(self, minimal_piece):
        writer = SVGWriter()
        svg = writer.to_string(minimal_piece)
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_to_string_full_piece(self, pants_front):
        writer = SVGWriter()
        svg = writer.to_string(pants_front)
        assert "<svg" in svg
        # Should contain piece name in a label
        assert "FRONT" in svg

    def test_to_string_fold_piece(self, fold_piece):
        writer = SVGWriter()
        svg = writer.to_string(fold_piece)
        assert "<svg" in svg

    def test_save_to_file(self, pants_front):
        writer = SVGWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.save(pants_front, os.path.join(tmpdir, "test.svg"))
            assert path.exists()
            content = path.read_text()
            assert "<svg" in content

    def test_save_sheet(self, pants_front, fold_piece):
        writer = SVGWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.save_sheet(
                [pants_front, fold_piece],
                os.path.join(tmpdir, "sheet.svg"),
                columns=2,
            )
            assert path.exists()
            content = path.read_text()
            assert "<svg" in content

    def test_feature_toggles(self, pants_front):
        writer = SVGWriter(
            include_seam_line=False,
            include_grain_line=False,
            include_fold_line=False,
            include_notches=False,
            include_darts=False,
            include_labels=False,
        )
        svg = writer.to_string(pants_front)
        assert "<svg" in svg

    def test_constants(self):
        assert PX_PER_INCH == 96.0
        assert MM_PER_INCH == 25.4
        assert CM_PER_INCH == 2.54


# ── PDFWriter ────────────────────────────────────────────────────────────────


class TestPDFWriter:
    def test_create_default(self):
        writer = PDFWriter()
        assert writer is not None

    def test_create_a4(self):
        writer = PDFWriter(page_size=PAGE_A4)
        assert writer is not None

    def test_save_single_piece(self, pants_front):
        writer = PDFWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.save(pants_front, os.path.join(tmpdir, "test.pdf"))
            assert path.exists()
            assert path.stat().st_size > 0

    def test_save_all_pieces(self, pants_front, fold_piece):
        writer = PDFWriter(include_cover=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.save_all(
                [pants_front, fold_piece],
                os.path.join(tmpdir, "all.pdf"),
            )
            assert path.exists()
            assert path.stat().st_size > 0

    def test_save_no_cover(self, pants_front):
        writer = PDFWriter(include_cover=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.save(pants_front, os.path.join(tmpdir, "nocover.pdf"))
            assert path.exists()

    def test_feature_toggles(self, pants_front):
        writer = PDFWriter(
            include_seam_line=False,
            include_grain_line=False,
            include_fold_line=False,
            include_notches=False,
            include_labels=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.save(pants_front, os.path.join(tmpdir, "minimal.pdf"))
            assert path.exists()

    def test_page_constants(self):
        # Letter: 8.5 x 11 in = 612 x 792 pt
        assert PAGE_LETTER == (612.0, 792.0)
        # A4: 595.276 x 841.89 pt
        assert abs(PAGE_A4[0] - 595.276) < 1.0
        assert abs(PAGE_A4[1] - 841.89) < 1.0
