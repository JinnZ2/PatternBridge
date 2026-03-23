"""
Boundary extraction: generates boundary points for a PatternPiece from
vision metadata and garment type.

This bridges the gap between the vision layer (which extracts features
and scores but not coordinates) and the geometry layer (which needs
boundary_points to encode and scale).

Two approaches:
  1. Template-based — pick a standard shape template for the garment type
     and piece name, then adjust dimensions from measurements.
  2. From contour — given an image and a binary mask, extract boundary
     points via contour detection (requires OpenCV, optional).

This module implements approach 1. Approach 2 can be added when
OpenCV-based preprocessing is integrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .piece import (
    PatternPiece,
    Point,
    GrainLine,
    FoldLine,
    Notch,
    Dart,
    SeamAllowance,
)


# ── Template shapes ──────────────────────────────────────────────────────────

# Each template is a function that returns (boundary_points, features) for
# a given set of measurements. Points are in inches, origin at top-left.


def _rectangle(width: float, height: float) -> list[Point]:
    """Simple rectangle — base for many pieces."""
    return [
        (0.0, 0.0), (width, 0.0),
        (width, height), (0.0, height),
    ]


def _pants_front(
    waist: float = 12.0,
    hip: float = 13.0,
    rise: float = 10.0,
    inseam: float = 28.0,
    hem: float = 9.0,
) -> list[Point]:
    """
    Pants front piece template.

    Shape: waistband at top, widens to hip, crotch curve,
    narrows to knee and hem.
    """
    total_height = rise + inseam
    hip_y = 7.0  # hip drops ~7" from waist
    knee_y = rise + inseam * 0.55
    crotch_ext = hip * 0.15  # crotch extension

    return [
        # Waist (top edge)
        (0.0, 0.0),
        (waist / 2, 0.0),
        # Side seam down to hip
        (hip / 2, hip_y),
        # Side seam to knee
        (hip / 2 * 0.85, knee_y),
        # Side seam to hem
        (hem / 2, total_height),
        # Hem across
        (-(hem / 2 - hem / 2), total_height),  # center hem
        # Inseam up from hem to crotch
        (-(hem / 2 * 0.1), knee_y),
        (-crotch_ext, rise),
        # Crotch curve back to center waist
        (-crotch_ext * 0.3, rise * 0.4),
    ]


def _pants_back(
    waist: float = 13.0,
    hip: float = 14.0,
    rise: float = 12.0,
    inseam: float = 28.0,
    hem: float = 9.5,
) -> list[Point]:
    """Pants back piece template — wider than front, deeper crotch curve."""
    total_height = rise + inseam
    hip_y = 7.0
    knee_y = rise + inseam * 0.55
    crotch_ext = hip * 0.2

    return [
        (0.0, 0.0),
        (waist / 2, 0.0),
        (hip / 2, hip_y),
        (hip / 2 * 0.85, knee_y),
        (hem / 2, total_height),
        (0.0, total_height),
        (-(hem / 2 * 0.1), knee_y),
        (-crotch_ext, rise),
        (-crotch_ext * 0.5, rise * 0.3),
    ]


def _bodice_front(
    bust: float = 16.0,
    waist: float = 13.0,
    shoulder: float = 14.5,
    length: float = 16.0,
) -> list[Point]:
    """Bodice front template — shoulder to waist with bust dart."""
    neck_width = shoulder * 0.35
    neck_drop = 3.0
    armhole_depth = length * 0.45

    return [
        # Center front neck
        (0.0, neck_drop),
        # Shoulder
        (neck_width, 0.0),
        (shoulder / 2, 0.5),
        # Armhole curve
        (bust / 2, armhole_depth),
        # Side seam
        (bust / 2, length * 0.7),
        (waist / 2, length),
        # Waist to center
        (0.0, length),
    ]


def _bodice_back(
    bust: float = 16.0,
    waist: float = 13.0,
    shoulder: float = 14.5,
    length: float = 17.0,
) -> list[Point]:
    """Bodice back template — similar to front, shallower neck."""
    neck_width = shoulder * 0.35
    neck_drop = 1.5
    armhole_depth = length * 0.42

    return [
        (0.0, neck_drop),
        (neck_width, 0.0),
        (shoulder / 2, 0.5),
        (bust / 2, armhole_depth),
        (bust / 2, length * 0.7),
        (waist / 2, length),
        (0.0, length),
    ]


def _skirt_front(
    waist: float = 13.0,
    hip: float = 17.0,
    length: float = 22.0,
) -> list[Point]:
    """Simple A-line skirt front."""
    hip_y = 7.0
    hem_width = hip * 1.15

    return [
        (0.0, 0.0),
        (waist / 2, 0.0),
        (hip / 2, hip_y),
        (hem_width / 2, length),
        (0.0, length),
    ]


def _sock(
    foot_length: float = 10.0,
    foot_width: float = 4.0,
    cuff_height: float = 6.0,
) -> list[Point]:
    """Sock template — simplified sole + cuff shape."""
    total_h = foot_length + cuff_height
    toe_curve = foot_width * 0.3

    return [
        # Cuff top
        (0.0, 0.0),
        (foot_width, 0.0),
        # Cuff side to ankle
        (foot_width, cuff_height),
        # Heel curve
        (foot_width + toe_curve, cuff_height + foot_length * 0.3),
        # Sole
        (foot_width, total_h - toe_curve),
        # Toe curve
        (foot_width * 0.5, total_h),
        (0.0, total_h - toe_curve),
        # Back up
        (-toe_curve, cuff_height + foot_length * 0.3),
        (0.0, cuff_height),
    ]


def _hat_crown(
    circumference: float = 22.0,
    depth: float = 7.0,
) -> list[Point]:
    """Hat crown panel (one of 6 wedge sections)."""
    panel_width = circumference / 6
    return [
        (0.0, 0.0),
        (panel_width, 0.0),
        (panel_width * 0.85, depth * 0.4),
        (panel_width * 0.6, depth * 0.75),
        (panel_width * 0.5, depth),
        (0.0, depth),
    ]


def _generic_rectangle(width: float = 10.0, height: float = 15.0) -> list[Point]:
    """Fallback for unknown piece types."""
    return _rectangle(width, height)


# ── Template registry ────────────────────────────────────────────────────────

# Maps (garment_type, piece_name_lower) to template functions.
# piece_name matching is partial — "front" matches "BODICE FRONT", "PANTS FRONT", etc.

TEMPLATES: dict[tuple[str, str], callable] = {
    ("pants", "front"): _pants_front,
    ("pants", "back"):  _pants_back,
    ("dress", "front"): _bodice_front,
    ("dress", "back"):  _bodice_back,
    ("top", "front"):   _bodice_front,
    ("top", "back"):    _bodice_back,
    ("skirt", "front"): _skirt_front,
    ("skirt", "back"):  _skirt_front,  # symmetric
    ("sock", "sole"):   _sock,
    ("sock", ""):       _sock,
    ("hat", "crown"):   _hat_crown,
    ("hat", ""):        _hat_crown,
}


def _match_template(garment_type: str, piece_name: str) -> callable | None:
    """Find the best matching template for a garment type and piece name."""
    gt = garment_type.lower().strip()
    pn = piece_name.lower().strip()

    # Exact match
    if (gt, pn) in TEMPLATES:
        return TEMPLATES[(gt, pn)]

    # Partial piece name match (e.g. "BODICE FRONT" matches "front")
    for (t_gt, t_pn), func in TEMPLATES.items():
        if t_gt == gt and t_pn and t_pn in pn:
            return func

    # Garment type only (empty piece name key)
    if (gt, "") in TEMPLATES:
        return TEMPLATES[(gt, "")]

    return None


# ── Main API ─────────────────────────────────────────────────────────────────


def generate_boundary(
    piece: PatternPiece,
    measurements: dict[str, float] | None = None,
) -> PatternPiece:
    """
    Generate boundary points for a PatternPiece using template shapes.

    Selects a template based on garment_type and piece name, generates
    boundary points scaled to the given measurements, and populates
    grain line, fold line, and notch positions.

    Mutates the piece in place. Returns the piece for chaining.

    Args:
        piece: PatternPiece with garment_type and name set.
        measurements: Optional measurement dict (bust, waist, hip, etc.).
                     If None, uses default template dimensions.

    Returns:
        The piece with boundary_points, grain_line, notches populated.
    """
    m = measurements or {}
    template_fn = _match_template(piece.garment_type, piece.name)

    if template_fn is None:
        # Fallback: generic rectangle
        width = m.get("hip", m.get("bust", 10.0)) / 2
        height = m.get("inseam", m.get("torso_front", 15.0))
        piece.boundary_points = _generic_rectangle(width, height)
    else:
        # Build kwargs from measurements where they match template params
        import inspect
        sig = inspect.signature(template_fn)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in m:
                kwargs[param_name] = m[param_name]
            # Common mappings: template param → measurement key
            elif param_name == "length" and "torso_front" in m:
                kwargs[param_name] = m["torso_front"]
            elif param_name == "shoulder" and "shoulder_width" in m:
                kwargs[param_name] = m["shoulder_width"]
            elif param_name == "rise" and "rise_front" in m:
                kwargs[param_name] = m["rise_front"]

        piece.boundary_points = template_fn(**kwargs)

    # Generate grain line (vertical center, standard orientation)
    if not piece.grain_line and piece.boundary_points:
        xs = [p[0] for p in piece.boundary_points]
        ys = [p[1] for p in piece.boundary_points]
        cx = (min(xs) + max(xs)) / 2
        piece.grain_line = GrainLine(
            start=(cx, min(ys) + 1.0),
            end=(cx, max(ys) - 1.0),
            angle_degrees=0.0,
        )

    # Generate fold line for fold pieces
    if piece.fold_line is None and _is_fold_piece(piece):
        ys = [p[1] for p in piece.boundary_points]
        piece.fold_line = FoldLine(
            start=(0.0, min(ys)),
            end=(0.0, max(ys)),
            axis="vertical",
            position="left",
        )

    # Generate notch positions at key seam intersections
    if not piece.notches:
        piece.notches = _generate_notches(piece)

    return piece


def generate_boundary_batch(
    pieces: list[PatternPiece],
    measurements: dict[str, float] | None = None,
) -> list[PatternPiece]:
    """Generate boundaries for multiple pieces."""
    return [generate_boundary(p, measurements) for p in pieces]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_fold_piece(piece: PatternPiece) -> bool:
    """Heuristic: piece is cut-on-fold if boundary starts at x=0."""
    if not piece.boundary_points:
        return False
    # Check if any boundary points are at x=0 (center front/back)
    x_vals = [p[0] for p in piece.boundary_points]
    has_center_edge = min(x_vals) <= 0.01
    # And name suggests center placement
    name = piece.name.lower()
    fold_indicators = ["front", "back", "center", "fold"]
    return has_center_edge and any(ind in name for ind in fold_indicators)


def _generate_notches(piece: PatternPiece) -> list[Notch]:
    """
    Place notches at standard seam matching points.

    Standard positions:
    - Hip level on side seam
    - Knee level on inseam/outseam (pants)
    - Waist center (if fold piece)
    """
    notches = []
    pts = piece.boundary_points
    if not pts or len(pts) < 4:
        return notches

    n = len(pts)

    # Place a notch roughly at the midpoint of each "long" edge
    # (heuristic: if two consecutive points are far apart)
    for i in range(n):
        j = (i + 1) % n
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        edge_len = math.sqrt(dx * dx + dy * dy)

        if edge_len > 8.0:  # long edge → place a matching notch
            mid = ((pts[i][0] + pts[j][0]) / 2, (pts[i][1] + pts[j][1]) / 2)
            notches.append(Notch(
                position=mid,
                boundary_index=i,
                notch_type="single",
            ))

    return notches
