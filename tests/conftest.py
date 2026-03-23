"""Shared test fixtures for PatternBridge tests."""

from __future__ import annotations

import pytest

from pattern_geometry.piece import (
    PatternPiece,
    GrainLine,
    FoldLine,
    Notch,
    Dart,
    SeamAllowance,
    LengthenShortenLine,
)


# ── Simple shapes ────────────────────────────────────────────────────────────


@pytest.fixture
def rectangle_points():
    """A 10x15 inch rectangle (pants front-like)."""
    return [(0, 0), (10, 0), (10, 15), (0, 15)]


@pytest.fixture
def pentagon_points():
    """A 5-sided shape (bodice-like)."""
    return [(0, 0), (10, 0), (10, 12), (5, 16), (0, 12)]


@pytest.fixture
def triangle_points():
    """Minimal encodable shape."""
    return [(0, 0), (5, 0), (2.5, 8)]


# ── PatternPiece fixtures ────────────────────────────────────────────────────


@pytest.fixture
def minimal_piece(triangle_points):
    """Bare-minimum piece with 3 boundary points."""
    return PatternPiece(
        name="MINIMAL",
        piece_number=1,
        boundary_points=triangle_points,
    )


@pytest.fixture
def pants_front(rectangle_points):
    """A pants front piece with typical features."""
    return PatternPiece(
        name="FRONT",
        piece_number=1,
        garment_type="pants",
        pattern_brand="McCall",
        pattern_id="M0001",
        size_label="Size 2",
        boundary_points=rectangle_points,
        grain_line=GrainLine(
            start=(5, 1), end=(5, 14), angle_degrees=0.0
        ),
        notches=[
            Notch(position=(0, 7.5), boundary_index=3, notch_type="single"),
            Notch(position=(10, 7.5), boundary_index=1, notch_type="single"),
        ],
        darts=[
            Dart(
                apex=(5, 3),
                leg_start=(4, 0),
                leg_end=(6, 0),
                intake_inches=1.0,
                depth_inches=3.0,
            ),
        ],
        seam_allowance=SeamAllowance(
            global_inches=0.625,
            edge_overrides={"waist": 1.0},
        ),
        total_vision_score=85.0,
        band_label="Good",
    )


@pytest.fixture
def fold_piece():
    """A piece meant to be cut on fold (symmetric about x=5)."""
    return PatternPiece(
        name="BACK",
        piece_number=2,
        garment_type="dress",
        boundary_points=[
            (0, 0), (5, 0), (5, 15), (0, 15),
        ],
        fold_line=FoldLine(
            start=(5, 0), end=(5, 15), axis="vertical", position="right"
        ),
        grain_line=GrainLine(start=(2.5, 1), end=(2.5, 14), angle_degrees=0.0),
        seam_allowance=SeamAllowance(global_inches=0.625),
    )


@pytest.fixture
def multi_piece_set(pants_front, fold_piece, minimal_piece):
    """A set of three diverse pieces."""
    return [pants_front, fold_piece, minimal_piece]


# ── Vision result fixtures ───────────────────────────────────────────────────


@pytest.fixture
def sample_vision_result():
    """A mock vision LLM result dict matching from_vision_result() expected format."""
    return {
        "piece_name": "FRONT",
        "piece_number": 1,
        "cut_quantity": 2,
        "garment_type": "pants",
        "pattern_brand": "Simplicity",
        "seam_allowance_inches": 0.625,
        "score_piece_identification": 18,
        "score_grain_line": 14,
        "score_fold_line": 15,
        "score_notch_positions": 12,
        "score_dart_definitions": 10,
        "score_seam_allowance": 8,
        "score_boundary_traceability": 8,
        "total_score": 85.0,
        "band_label": "Good",
        "image_quality_notes": "",
    }
