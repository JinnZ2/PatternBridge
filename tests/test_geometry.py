"""Tests for the pattern_geometry layer: piece, encoder, and scaler."""

from __future__ import annotations

import json
import pytest

from pattern_geometry.piece import (
    PatternPiece,
    GrainLine,
    FoldLine,
    Notch,
    Dart,
    SeamAllowance,
    LengthenShortenLine,
    Point,
)
from pattern_geometry.encoder import PatternEncoder
from pattern_geometry.scaler import (
    PatternScaler,
    ScaleResult,
    GradePoint,
    Measure,
    PROFILE_ZERO_MUSCULAR,
    PROFILE_TALL_36_36,
    PANTS_GRADE_RULES,
    BODICE_GRADE_RULES,
)


# ── PatternPiece ─────────────────────────────────────────────────────────────


class TestPatternPiece:
    def test_create_minimal(self, minimal_piece):
        assert minimal_piece.name == "MINIMAL"
        assert minimal_piece.piece_number == 1

    def test_boundary_point_count(self, pants_front):
        assert pants_front.boundary_point_count == 4

    def test_is_encodable_with_enough_points(self, pants_front):
        # pants_front has 4 points and score 85 > 51
        assert pants_front.is_encodable is True

    def test_is_not_encodable_with_two_points(self):
        piece = PatternPiece(
            name="LINE", piece_number=1,
            boundary_points=[(0, 0), (10, 0)],
            total_vision_score=90.0,
        )
        assert piece.is_encodable is False

    def test_is_cut_on_fold(self, fold_piece):
        assert fold_piece.is_cut_on_fold is True

    def test_not_cut_on_fold(self, pants_front):
        assert pants_front.is_cut_on_fold is False

    def test_full_cut_quantity_fold(self, fold_piece):
        # fold pieces double the cut quantity
        assert fold_piece.full_cut_quantity == 2

    def test_has_darts(self, pants_front):
        assert pants_front.has_darts is True

    def test_no_darts(self, minimal_piece):
        assert minimal_piece.has_darts is False

    def test_needs_better_image_low_score(self):
        piece = PatternPiece(
            name="BAD", piece_number=1,
            total_vision_score=30.0,
        )
        assert piece.needs_better_image is True

    def test_doesnt_need_better_image_high_score(self, pants_front):
        assert pants_front.needs_better_image is False

    def test_seam_allowance_for_edge(self, pants_front):
        sa = pants_front.seam_allowance
        assert sa.for_edge("waist") == 1.0
        assert sa.for_edge("side") == 0.625  # falls back to global

    def test_effective_measurement_with_ease(self):
        piece = PatternPiece(
            name="TEST", piece_number=1,
            target_measurements={"bust": 34.0},
            ease_allowances={"bust": 2.0},
        )
        assert piece.effective_measurement("bust") == 36.0

    def test_effective_measurement_without_ease(self):
        piece = PatternPiece(
            name="TEST", piece_number=1,
            target_measurements={"bust": 34.0},
        )
        assert piece.effective_measurement("bust") == 34.0

    def test_effective_measurement_missing_key(self, minimal_piece):
        assert minimal_piece.effective_measurement("bust") is None


class TestPatternPieceSerialization:
    def test_to_dict(self, pants_front):
        d = pants_front.to_dict()
        assert d["name"] == "FRONT"
        assert d["piece_number"] == 1
        assert d["garment_type"] == "pants"
        assert len(d["boundary_points"]) == 4
        assert d["grain_line"] is not None

    def test_to_json(self, pants_front):
        j = pants_front.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "FRONT"
        assert isinstance(parsed["boundary_points"], list)

    def test_to_dict_roundtrip_keys(self, pants_front):
        d = pants_front.to_dict()
        expected_keys = {
            "name", "piece_number", "cut_quantity", "garment_type",
            "pattern_brand", "pattern_id", "size_label", "units",
            "boundary_points", "grain_line", "fold_line", "notches",
            "darts", "seam_allowance", "encoded_tokens",
        }
        # All expected keys should be present
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"


class TestPatternPieceFactory:
    def test_from_vision_result(self, sample_vision_result):
        piece = PatternPiece.from_vision_result(
            sample_vision_result, image_source="test.jpg"
        )
        assert piece.name == "FRONT"
        assert piece.piece_number == 1
        assert piece.garment_type == "pants"
        assert piece.image_source == "test.jpg"
        # boundary_points are empty — filled by geometry layer later
        assert piece.boundary_points == []

    def test_from_vision_result_seam_allowance(self, sample_vision_result):
        piece = PatternPiece.from_vision_result(sample_vision_result)
        assert piece.seam_allowance is not None
        assert piece.seam_allowance.global_inches == 0.625

    def test_from_vision_result_no_geometry(self, sample_vision_result):
        """from_vision_result does not populate geometry fields."""
        piece = PatternPiece.from_vision_result(sample_vision_result)
        assert piece.grain_line is None
        assert piece.fold_line is None
        assert piece.notches == []

    def test_from_vision_result_scores(self, sample_vision_result):
        piece = PatternPiece.from_vision_result(sample_vision_result)
        assert piece.total_vision_score == 85.0
        assert piece.vision_scores is not None
        assert len(piece.vision_scores) == 7

    def test_from_vision_result_cut_quantity(self, sample_vision_result):
        piece = PatternPiece.from_vision_result(sample_vision_result)
        assert piece.cut_quantity == 2


# ── PatternEncoder ───────────────────────────────────────────────────────────


class TestPatternEncoder:
    def test_create_encoder(self):
        encoder = PatternEncoder()
        assert encoder is not None

    def test_create_encoder_no_symmetry(self):
        encoder = PatternEncoder(detect_symmetry=False)
        assert encoder.symmetry_detector is None

    def test_encode_minimal(self, minimal_piece):
        encoder = PatternEncoder()
        result = encoder.encode(minimal_piece)
        assert result is minimal_piece  # mutates in place
        assert len(minimal_piece.encoded_tokens) > 0

    def test_encode_pants_front(self, pants_front):
        encoder = PatternEncoder()
        encoder.encode(pants_front)
        assert len(pants_front.encoded_tokens) > 0
        # Tokens should be strings
        for token in pants_front.encoded_tokens:
            assert isinstance(token, str)
            assert len(token) >= 5  # e.g. "001|O"

    def test_encode_rejects_too_few_points(self):
        piece = PatternPiece(
            name="LINE", piece_number=1,
            boundary_points=[(0, 0), (10, 0)],
        )
        encoder = PatternEncoder()
        with pytest.raises(ValueError, match="at least 3"):
            encoder.encode(piece)

    def test_encode_batch(self, pants_front, minimal_piece):
        encoder = PatternEncoder()
        results = encoder.encode_batch([pants_front, minimal_piece])
        assert len(results) == 2
        assert all(len(p.encoded_tokens) > 0 for p in results)

    def test_decode_returns_points(self, pants_front):
        encoder = PatternEncoder()
        encoder.encode(pants_front)
        points = encoder.decode(pants_front)
        assert isinstance(points, list)
        assert len(points) > 0
        for pt in points:
            assert len(pt) == 2

    def test_decode_empty_tokens(self, minimal_piece):
        encoder = PatternEncoder()
        # Don't encode — tokens empty
        points = encoder.decode(minimal_piece)
        assert points == []

    def test_token_summary(self, pants_front):
        encoder = PatternEncoder()
        encoder.encode(pants_front)
        summary = encoder.token_summary(pants_front)
        assert "piece" in summary
        assert summary["piece"] == "FRONT"
        assert "total_tokens" in summary
        assert summary["total_tokens"] > 0
        assert "operators" in summary
        assert "symbols" in summary

    def test_token_summary_no_tokens(self, minimal_piece):
        encoder = PatternEncoder()
        summary = encoder.token_summary(minimal_piece)
        assert "error" in summary

    def test_fold_piece_symmetry(self, fold_piece):
        encoder = PatternEncoder(detect_symmetry=True)
        encoder.encode(fold_piece)
        assert len(fold_piece.encoded_tokens) > 0
        # symmetry_detected should have been set (True or False depending on detector)
        assert isinstance(fold_piece.symmetry_detected, bool)


# ── PatternScaler ────────────────────────────────────────────────────────────


class TestPatternScaler:
    def test_create_for_zero_muscular(self):
        scaler = PatternScaler.for_zero_muscular()
        assert scaler is not None

    def test_create_for_tall_36_36(self):
        scaler = PatternScaler.for_tall_36_36()
        assert scaler is not None

    def test_create_custom(self):
        scaler = PatternScaler(
            source_measurements={"bust": 34, "waist": 26, "hip": 36},
            target_measurements={"bust": 38, "waist": 30, "hip": 40},
        )
        assert scaler is not None

    def test_scale_pants_front(self, pants_front):
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(pants_front)
        assert isinstance(result, ScaleResult)
        assert result.scaled_piece is not None
        assert result.scaled_piece.name == "FRONT"
        assert len(result.scaled_piece.boundary_points) > 0

    def test_scale_preserves_name(self, pants_front):
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(pants_front)
        assert result.scaled_piece.name == pants_front.name

    def test_scale_result_has_deltas(self, pants_front):
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(pants_front)
        assert isinstance(result.measurement_deltas, dict)

    def test_scale_result_summary(self, pants_front):
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(pants_front)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_scale_batch(self, pants_front, fold_piece):
        scaler = PatternScaler.for_zero_muscular()
        results = scaler.scale_batch([pants_front, fold_piece])
        assert len(results) == 2
        assert all(isinstance(r, ScaleResult) for r in results)

    def test_scale_with_explicit_rules(self, pants_front):
        scaler = PatternScaler.for_zero_muscular()
        result = scaler.scale(pants_front, grade_rules=PANTS_GRADE_RULES)
        assert result.scaled_piece is not None

    def test_profiles_have_expected_keys(self):
        for profile in [PROFILE_ZERO_MUSCULAR, PROFILE_TALL_36_36]:
            assert Measure.BUST in profile
            assert Measure.WAIST in profile
            assert Measure.HIP in profile
            assert Measure.INSEAM in profile

    def test_measure_constants(self):
        assert Measure.BUST == "bust"
        assert Measure.WAIST == "waist"
        assert Measure.HIP == "hip"


# ── GradePoint ───────────────────────────────────────────────────────────────


class TestGradePoint:
    def test_pants_grade_rules_exist(self):
        assert len(PANTS_GRADE_RULES) > 0

    def test_bodice_grade_rules_exist(self):
        assert len(BODICE_GRADE_RULES) > 0

    def test_grade_point_fields(self):
        gp = PANTS_GRADE_RULES[0]
        assert isinstance(gp.landmark_name, str)
        assert isinstance(gp.measurement_key, str)
        assert isinstance(gp.x_rate, (int, float))
        assert isinstance(gp.y_rate, (int, float))
