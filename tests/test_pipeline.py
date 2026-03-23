"""End-to-end pipeline tests: vision result → piece → encode → scale → output."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from pattern_geometry.piece import PatternPiece
from pattern_geometry.encoder import PatternEncoder
from pattern_geometry.scaler import PatternScaler, PANTS_GRADE_RULES
from pattern_output.svg_writer import SVGWriter
from pattern_output.pdf_writer import PDFWriter
from bridge.pattern_bridge import PatternBridge, PipelineResult, PROFILES


# ── PipelineResult ───────────────────────────────────────────────────────────


class TestPipelineResult:
    def test_create_empty(self):
        result = PipelineResult()
        assert result.raw_pieces == []
        assert result.warnings == []
        assert "svg" in result.output_files

    def test_summary_empty(self):
        result = PipelineResult()
        summary = result.summary()
        assert isinstance(summary, str)


# ── PatternBridge construction ───────────────────────────────────────────────


class TestPatternBridgeInit:
    def test_create_default(self):
        bridge = PatternBridge(api_key="test-key")
        assert bridge.provider == "anthropic"
        assert bridge.encode is True

    def test_create_no_encode(self):
        bridge = PatternBridge(api_key="test-key", encode=False)
        assert bridge.encoder is None

    def test_profiles_available(self):
        assert "zero_muscular" in PROFILES
        assert "tall_36_36" in PROFILES


# ── End-to-end pipeline (synthetic data, no LLM) ────────────────────────────


class TestEndToEndPipeline:
    """Test the full pipeline using synthetic vision results (no API calls)."""

    # Synthetic boundary points to simulate geometry layer output
    SYNTHETIC_BOUNDARY = [(0, 0), (12, 0), (12, 18), (6, 20), (0, 18)]

    def _piece_with_geometry(self, vision_result):
        """Create a piece from vision result and add synthetic boundary points."""
        piece = PatternPiece.from_vision_result(
            vision_result, image_source="synthetic"
        )
        piece.boundary_points = self.SYNTHETIC_BOUNDARY
        return piece

    def test_vision_to_piece(self, sample_vision_result):
        """Stage 1→2: vision result → PatternPiece."""
        piece = PatternPiece.from_vision_result(
            sample_vision_result, image_source="synthetic"
        )
        assert piece.name == "FRONT"
        # from_vision_result does NOT populate boundary_points (geometry layer does)
        assert piece.boundary_points == []
        assert piece.total_vision_score == 85.0

    def test_piece_to_encoded(self, sample_vision_result):
        """Stage 2→3: PatternPiece → encoded tokens."""
        piece = self._piece_with_geometry(sample_vision_result)
        encoder = PatternEncoder()
        encoder.encode(piece)
        assert len(piece.encoded_tokens) > 0

    def test_encoded_to_scaled(self, sample_vision_result):
        """Stage 3→4: encoded piece → scaled piece."""
        piece = self._piece_with_geometry(sample_vision_result)
        encoder = PatternEncoder()
        encoder.encode(piece)

        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(piece)
        assert result.scaled_piece is not None
        assert len(result.scaled_piece.boundary_points) > 0

    def test_scaled_to_svg(self, sample_vision_result):
        """Stage 4→5: scaled piece → SVG output."""
        piece = self._piece_with_geometry(sample_vision_result)

        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(piece)

        writer = SVGWriter()
        svg = writer.to_string(result.scaled_piece)
        assert "<svg" in svg

    def test_scaled_to_pdf(self, sample_vision_result):
        """Stage 4→5: scaled piece → PDF output."""
        piece = self._piece_with_geometry(sample_vision_result)

        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(piece)

        writer = PDFWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = writer.save(result.scaled_piece, os.path.join(tmpdir, "out.pdf"))
            assert path.exists()
            assert path.stat().st_size > 0

    def test_scaled_to_json(self, sample_vision_result):
        """Stage 4→5: scaled piece → JSON output."""
        piece = self._piece_with_geometry(sample_vision_result)

        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(piece)

        j = result.scaled_piece.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "FRONT"

    def test_full_pipeline_synthetic(self, sample_vision_result):
        """Full pipeline: vision result → piece → encode → scale → SVG + PDF + JSON."""
        # Vision → Piece (with synthetic geometry)
        piece = self._piece_with_geometry(sample_vision_result)

        # Encode
        encoder = PatternEncoder()
        encoder.encode(piece)
        assert len(piece.encoded_tokens) > 0

        # Scale
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(piece, grade_rules=PANTS_GRADE_RULES)
        scaled = result.scaled_piece

        # SVG
        svg_writer = SVGWriter()
        svg = svg_writer.to_string(scaled)
        assert "<svg" in svg

        # PDF
        pdf_writer = PDFWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = pdf_writer.save(scaled, os.path.join(tmpdir, "out.pdf"))
            assert pdf_path.exists()

            # SVG file
            svg_path = svg_writer.save(scaled, os.path.join(tmpdir, "out.svg"))
            assert svg_path.exists()

        # JSON
        j = scaled.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "FRONT"
        # encoded_tokens are not carried over by scaler (would need re-encoding)
        assert len(parsed["boundary_points"]) > 0

    def test_multi_piece_pipeline(self, multi_piece_set):
        """Pipeline with multiple pieces at once."""
        encoder = PatternEncoder()
        encoded = encoder.encode_batch(multi_piece_set)
        assert len(encoded) == 3

        scaler = PatternScaler.for_zero_muscular()
        results = scaler.scale_batch(encoded)
        assert len(results) == 3

        svg_writer = SVGWriter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = svg_writer.save_sheet(
                [r.scaled_piece for r in results],
                os.path.join(tmpdir, "sheet.svg"),
            )
            assert path.exists()
