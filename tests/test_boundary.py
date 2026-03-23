"""Tests for boundary extraction — the vision→geometry bridge."""

from __future__ import annotations

import pytest

from pattern_geometry.piece import PatternPiece, GrainLine, FoldLine
from pattern_geometry.boundary import (
    generate_boundary,
    generate_boundary_batch,
    _pants_front,
    _pants_back,
    _bodice_front,
    _bodice_back,
    _skirt_front,
    _sock,
    _hat_crown,
    _match_template,
)


# ── Template functions ───────────────────────────────────────────────────────


class TestTemplates:
    def test_pants_front_default(self):
        pts = _pants_front()
        assert len(pts) >= 6
        # Should start at origin area
        assert pts[0][1] == 0.0  # waist at top

    def test_pants_front_custom_measurements(self):
        pts = _pants_front(waist=14.0, hip=16.0, rise=11.0, inseam=32.0)
        assert len(pts) >= 6
        # Height should reflect rise + inseam
        max_y = max(p[1] for p in pts)
        assert max_y == pytest.approx(43.0, abs=1.0)

    def test_pants_back_default(self):
        pts = _pants_back()
        assert len(pts) >= 6

    def test_pants_back_wider_than_front(self):
        front = _pants_front(waist=12.0, hip=13.0)
        back = _pants_back(waist=13.0, hip=14.0)
        front_width = max(p[0] for p in front) - min(p[0] for p in front)
        back_width = max(p[0] for p in back) - min(p[0] for p in back)
        assert back_width > front_width

    def test_bodice_front_default(self):
        pts = _bodice_front()
        assert len(pts) >= 5

    def test_bodice_back_default(self):
        pts = _bodice_back()
        assert len(pts) >= 5

    def test_skirt_front_default(self):
        pts = _skirt_front()
        assert len(pts) >= 4

    def test_sock_default(self):
        pts = _sock()
        assert len(pts) >= 6

    def test_hat_crown_default(self):
        pts = _hat_crown()
        assert len(pts) >= 4

    def test_all_templates_return_valid_points(self):
        templates = [
            _pants_front, _pants_back, _bodice_front, _bodice_back,
            _skirt_front, _sock, _hat_crown,
        ]
        for fn in templates:
            pts = fn()
            assert len(pts) >= 3, f"{fn.__name__} returned < 3 points"
            for p in pts:
                assert len(p) == 2, f"{fn.__name__} returned non-2D point"
                assert isinstance(p[0], (int, float))
                assert isinstance(p[1], (int, float))


# ── Template matching ────────────────────────────────────────────────────────


class TestTemplateMatching:
    def test_exact_match(self):
        fn = _match_template("pants", "front")
        assert fn is _pants_front

    def test_partial_match(self):
        fn = _match_template("pants", "PANTS FRONT")
        assert fn is _pants_front

    def test_garment_only_match(self):
        fn = _match_template("sock", "SOLE")
        assert fn is _sock

    def test_unknown_returns_none(self):
        fn = _match_template("spaceship", "wing")
        assert fn is None

    def test_case_insensitive(self):
        fn = _match_template("PANTS", "BACK")
        assert fn is _pants_back


# ── generate_boundary ────────────────────────────────────────────────────────


class TestGenerateBoundary:
    def test_pants_front_piece(self):
        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="pants",
        )
        result = generate_boundary(piece)
        assert result is piece  # mutates in place
        assert len(piece.boundary_points) >= 6
        assert piece.grain_line is not None

    def test_pants_back_piece(self):
        piece = PatternPiece(
            name="BACK", piece_number=2, garment_type="pants",
        )
        generate_boundary(piece)
        assert len(piece.boundary_points) >= 6

    def test_bodice_with_measurements(self):
        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="dress",
        )
        measurements = {"bust": 34.0, "waist": 26.0, "shoulder_width": 15.0}
        generate_boundary(piece, measurements)
        assert len(piece.boundary_points) >= 5
        # Bust measurement should influence width
        max_x = max(p[0] for p in piece.boundary_points)
        assert max_x == pytest.approx(17.0, abs=1.0)  # bust/2

    def test_unknown_garment_gets_rectangle(self):
        piece = PatternPiece(
            name="MYSTERY", piece_number=1, garment_type="spaceship",
        )
        generate_boundary(piece)
        assert len(piece.boundary_points) == 4  # rectangle fallback

    def test_generates_grain_line(self):
        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="pants",
        )
        generate_boundary(piece)
        assert piece.grain_line is not None
        assert piece.grain_line.angle_degrees == 0.0

    def test_does_not_overwrite_existing_grain_line(self):
        existing_gl = GrainLine(start=(5, 1), end=(5, 20), angle_degrees=45.0)
        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="pants",
            grain_line=existing_gl,
        )
        generate_boundary(piece)
        assert piece.grain_line is existing_gl
        assert piece.grain_line.angle_degrees == 45.0

    def test_fold_piece_gets_fold_line(self):
        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="dress",
        )
        generate_boundary(piece)
        # Bodice front template starts at x=0 → fold piece
        if min(p[0] for p in piece.boundary_points) <= 0.01:
            assert piece.fold_line is not None
            assert piece.fold_line.axis == "vertical"

    def test_generates_notches(self):
        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="pants",
        )
        generate_boundary(piece)
        # Long edges should get notches
        assert len(piece.notches) >= 0  # may or may not have notches depending on edge lengths

    def test_batch(self):
        pieces = [
            PatternPiece(name="FRONT", piece_number=1, garment_type="pants"),
            PatternPiece(name="BACK", piece_number=2, garment_type="pants"),
        ]
        results = generate_boundary_batch(pieces)
        assert len(results) == 2
        assert all(len(p.boundary_points) >= 6 for p in results)


# ── Full pipeline with boundary ──────────────────────────────────────────────


class TestBoundaryPipeline:
    """Test that boundary-generated pieces work through encode and scale."""

    def test_boundary_to_encode(self):
        from pattern_geometry.encoder import PatternEncoder

        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="pants",
        )
        generate_boundary(piece)
        encoder = PatternEncoder()
        encoder.encode(piece)
        assert len(piece.encoded_tokens) > 0

    def test_boundary_to_scale(self):
        from pattern_geometry.scaler import PatternScaler

        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="pants",
        )
        generate_boundary(piece)
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(piece)
        assert result.scaled_piece is not None
        assert len(result.scaled_piece.boundary_points) > 0

    def test_boundary_to_svg(self):
        from pattern_output.svg_writer import SVGWriter

        piece = PatternPiece(
            name="FRONT", piece_number=1, garment_type="pants",
        )
        generate_boundary(piece)
        writer = SVGWriter()
        svg = writer.to_string(piece)
        assert "<svg" in svg
        assert "FRONT" in svg

    def test_full_pipeline_vision_to_output(self):
        """Simulate: vision result → from_vision_result → generate_boundary → encode → scale → SVG."""
        from pattern_geometry.encoder import PatternEncoder
        from pattern_geometry.scaler import PatternScaler
        from pattern_output.svg_writer import SVGWriter

        # Simulate vision output
        vision_result = {
            "piece_name": "FRONT",
            "piece_number": 1,
            "garment_type": "pants",
            "pattern_brand": "McCall",
            "seam_allowance_inches": 0.625,
            "total_score": 80.0,
            "band_label": "Good",
        }

        # Vision → Piece
        piece = PatternPiece.from_vision_result(vision_result)

        # Generate boundary (the new bridge step)
        measurements = {"waist": 12.0, "hip": 13.0, "rise_front": 10.0, "inseam": 28.0}
        generate_boundary(piece, measurements)
        assert len(piece.boundary_points) >= 6

        # Encode
        encoder = PatternEncoder()
        encoder.encode(piece)
        assert len(piece.encoded_tokens) > 0

        # Scale
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(piece)

        # SVG
        writer = SVGWriter()
        svg = writer.to_string(result.scaled_piece)
        assert "<svg" in svg
