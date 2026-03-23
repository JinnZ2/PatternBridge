"""
PatternScaler: Parametric scaling engine for PatternBridge.

Takes an encoded PatternPiece and scales it to target body measurements.
Scaling operates on boundary points using grading rules — the same
approach commercial pattern companies use to grade between sizes.

Key concept: we don't redraw the pattern. We move specific landmark
points by calculated amounts, then interpolate the curves between them.

Grading rules define how much each landmark moves per unit of measurement
change. Different body areas grade differently:
    - Waist seams move differently than side seams
    - Inseam length is independent of hip width
    - Muscle definition requires rotational ease, not just circumference

Built-in profiles:
    PROFILE_ZERO_MUSCULAR  — size 0 frame with muscle definition (your measurements)
    PROFILE_TALL_36_36     — tall, 36" waist, 36" inseam (your husband)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .piece import PatternPiece, Point, Dart, SeamAllowance


# ── Measurement keys (standardized across all profiles) ──────────────────────

class Measure:
    """Standardized measurement key names."""
    # Circumferences (inches)
    BUST          = "bust"
    WAIST         = "waist"
    HIP           = "hip"
    THIGH         = "thigh"
    BICEP         = "bicep"
    NECK          = "neck"

    # Lengths (inches)
    INSEAM        = "inseam"
    OUTSEAM       = "outseam"
    TORSO_FRONT   = "torso_front"       # shoulder to waist, front
    TORSO_BACK    = "torso_back"        # shoulder to waist, back
    SHOULDER_WIDTH = "shoulder_width"   # shoulder point to shoulder point
    BACK_WIDTH    = "back_width"        # across back at widest
    SLEEVE_LENGTH = "sleeve_length"
    RISE_FRONT    = "rise_front"        # waist to crotch, front
    RISE_BACK     = "rise_back"         # waist to crotch, back

    # Ease adjustments (added on top of body measurements)
    EASE_BUST     = "ease_bust"
    EASE_WAIST    = "ease_waist"
    EASE_HIP      = "ease_hip"
    EASE_THIGH    = "ease_thigh"
    EASE_BICEP    = "ease_bicep"


# ── Built-in measurement profiles ────────────────────────────────────────────

# Your measurements — size 0 frame, muscular build
PROFILE_ZERO_MUSCULAR: dict[str, float] = {
    Measure.BUST:           32.0,
    Measure.WAIST:          24.0,
    Measure.HIP:            34.0,
    Measure.THIGH:          20.0,   # muscular — larger than standard size 0
    Measure.BICEP:          12.5,   # muscular
    Measure.SHOULDER_WIDTH: 14.5,
    Measure.BACK_WIDTH:     13.5,
    Measure.TORSO_FRONT:    15.5,
    Measure.TORSO_BACK:     16.0,
    Measure.INSEAM:         28.5,   # adjust to your actual
    Measure.RISE_FRONT:      9.5,
    Measure.RISE_BACK:      13.0,
    # Ease — fitted but with muscle room
    Measure.EASE_BUST:       1.5,
    Measure.EASE_WAIST:      0.5,
    Measure.EASE_HIP:        1.0,
    Measure.EASE_THIGH:      1.5,   # extra for muscle
    Measure.EASE_BICEP:      1.5,
}

# Your husband's measurements — tall, 36/36
PROFILE_TALL_36_36: dict[str, float] = {
    Measure.BUST:           40.0,   # chest
    Measure.WAIST:          36.0,
    Measure.HIP:            40.0,
    Measure.THIGH:          23.0,
    Measure.BICEP:          14.0,
    Measure.SHOULDER_WIDTH: 18.5,
    Measure.BACK_WIDTH:     16.5,
    Measure.TORSO_FRONT:    17.5,
    Measure.TORSO_BACK:     18.5,
    Measure.INSEAM:         36.0,
    Measure.RISE_FRONT:     11.0,
    Measure.RISE_BACK:      14.5,
    # Ease — work gear needs movement room
    Measure.EASE_BUST:       2.0,
    Measure.EASE_WAIST:      1.0,
    Measure.EASE_HIP:        2.0,
    Measure.EASE_THIGH:      2.0,
    Measure.EASE_BICEP:      2.0,
}

# Standard size reference points for grading math
# These are the base measurements the original patterns are drafted to
STANDARD_SIZE_0: dict[str, float] = {
    Measure.BUST:           31.5,
    Measure.WAIST:          24.0,
    Measure.HIP:            33.5,
    Measure.THIGH:          19.0,
    Measure.INSEAM:         28.5,
    Measure.RISE_FRONT:      9.0,
    Measure.RISE_BACK:      12.5,
    Measure.TORSO_FRONT:    15.0,
    Measure.TORSO_BACK:     15.5,
    Measure.SHOULDER_WIDTH: 14.0,
}

STANDARD_SIZE_36_36: dict[str, float] = {
    Measure.BUST:           38.0,   # chest
    Measure.WAIST:          36.0,
    Measure.HIP:            38.0,
    Measure.THIGH:          22.0,
    Measure.INSEAM:         36.0,
    Measure.RISE_FRONT:     10.5,
    Measure.RISE_BACK:      14.0,
    Measure.TORSO_FRONT:    17.0,
    Measure.TORSO_BACK:     18.0,
    Measure.SHOULDER_WIDTH: 18.0,
}


# ── Grading rules ─────────────────────────────────────────────────────────────

@dataclass
class GradePoint:
    """
    Defines how much a specific landmark moves per inch of measurement change.
    x_rate and y_rate are in inches of movement per inch of measurement change.
    """
    landmark_name: str
    measurement_key: str        # which measurement drives this grade
    x_rate: float               # horizontal movement per inch of measurement
    y_rate: float               # vertical movement per inch of measurement


# Grading rules for pants pieces
PANTS_GRADE_RULES: list[GradePoint] = [
    # Waist points move with waist measurement
    GradePoint("waist_side",    Measure.WAIST,  0.25,  0.0),
    GradePoint("waist_center",  Measure.WAIST,  0.0,   0.0),   # center stays fixed

    # Hip points move with hip measurement
    GradePoint("hip_side",      Measure.HIP,    0.25,  0.0),

    # Crotch moves with rise measurements
    GradePoint("crotch_front",  Measure.RISE_FRONT, 0.0, 1.0),
    GradePoint("crotch_back",   Measure.RISE_BACK,  0.25, 1.0),

    # Thigh moves with thigh measurement
    GradePoint("thigh_side",    Measure.THIGH,  0.125, 0.0),
    GradePoint("thigh_inner",   Measure.THIGH,  0.125, 0.0),

    # Hem moves with inseam (length only, no width change)
    GradePoint("hem_side",      Measure.INSEAM, 0.0,   1.0),
    GradePoint("hem_inner",     Measure.INSEAM, 0.0,   1.0),
]

# Grading rules for bodice/dress pieces
BODICE_GRADE_RULES: list[GradePoint] = [
    GradePoint("shoulder_side", Measure.SHOULDER_WIDTH, 0.25, 0.0),
    GradePoint("bust_side",     Measure.BUST,           0.25, 0.0),
    GradePoint("waist_side",    Measure.WAIST,          0.25, 0.0),
    GradePoint("waist_center",  Measure.WAIST,          0.0,  0.0),
    GradePoint("hem_side",      Measure.HIP,            0.25, 0.0),
    GradePoint("shoulder_slope",Measure.TORSO_FRONT,    0.0,  0.125),
]


# ── ScaleResult ────────────────────────────────────────────────────────────────

@dataclass
class ScaleResult:
    """Output of a scaling operation."""
    original_piece: PatternPiece
    scaled_piece: PatternPiece
    source_profile: str
    target_profile: str
    measurement_deltas: dict[str, float]    # how much each measurement changed
    grade_movements: list[dict]             # per-landmark movement applied
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Scale: {self.source_profile} → {self.target_profile}",
            f"Piece: {self.original_piece.name}",
            "",
            "Measurement changes:",
        ]
        for key, delta in self.measurement_deltas.items():
            if abs(delta) > 0.01:
                direction = "+" if delta > 0 else ""
                lines.append(f"  {key}: {direction}{delta:.2f}\"")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)


# ── PatternScaler ─────────────────────────────────────────────────────────────

class PatternScaler:
    """
    Scales PatternPiece boundary points from a source measurement profile
    to a target measurement profile.

    Scaling approach:
        1. Identify landmark points on boundary (corners, notches, grade points)
        2. Compute measurement deltas between source and target
        3. Apply grading rules to move landmark points
        4. Interpolate curves between moved landmarks
        5. Update dart dimensions proportionally
        6. Return new PatternPiece at target measurements

    Args:
        source_measurements: Base measurements the pattern is drafted to.
        target_measurements: Desired output measurements.
        ease: Additional ease to add on top of target measurements.
    """

    def __init__(
        self,
        source_measurements: dict[str, float],
        target_measurements: dict[str, float],
        ease: Optional[dict[str, float]] = None,
    ):
        self.source = source_measurements
        self.target = target_measurements
        self.ease = ease or {}

    # ── Factory methods for common profiles ──────────────────────────────────

    @classmethod
    def for_zero_muscular(cls) -> "PatternScaler":
        """Scale to your measurements — size 0 frame with muscle definition."""
        ease = {
            k.replace("ease_", ""): v
            for k, v in PROFILE_ZERO_MUSCULAR.items()
            if k.startswith("ease_")
        }
        target = {
            k: v for k, v in PROFILE_ZERO_MUSCULAR.items()
            if not k.startswith("ease_")
        }
        return cls(
            source_measurements=STANDARD_SIZE_0,
            target_measurements=target,
            ease=ease,
        )

    @classmethod
    def for_tall_36_36(cls) -> "PatternScaler":
        """Scale to your husband's measurements — tall, 36/36."""
        ease = {
            k.replace("ease_", ""): v
            for k, v in PROFILE_TALL_36_36.items()
            if k.startswith("ease_")
        }
        target = {
            k: v for k, v in PROFILE_TALL_36_36.items()
            if not k.startswith("ease_")
        }
        return cls(
            source_measurements=STANDARD_SIZE_36_36,
            target_measurements=target,
            ease=ease,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def scale(
        self,
        piece: PatternPiece,
        grade_rules: Optional[list[GradePoint]] = None,
    ) -> ScaleResult:
        """
        Scale a PatternPiece to target measurements.

        Args:
            piece: PatternPiece with boundary_points populated.
            grade_rules: Grading rules to apply. Auto-detected from
                         garment_type if not provided.

        Returns:
            ScaleResult with scaled PatternPiece and metadata.
        """
        if not piece.boundary_points:
            raise ValueError(f"PatternPiece '{piece.name}' has no boundary points to scale.")

        # Auto-detect grade rules from garment type
        if grade_rules is None:
            grade_rules = self._detect_grade_rules(piece)

        # Compute measurement deltas
        deltas = self._compute_deltas()

        # Scale boundary points
        scaled_points, movements = self._apply_grading(
            piece.boundary_points, piece, grade_rules, deltas
        )

        # Scale darts proportionally
        scaled_darts = self._scale_darts(piece.darts, deltas, piece.garment_type)

        # Update lengthen/shorten lines for height changes
        scaled_ls_lines = self._scale_lengthen_shorten(piece, deltas)

        # Build scaled piece
        scaled = PatternPiece(
            name=piece.name,
            piece_number=piece.piece_number,
            cut_quantity=piece.cut_quantity,
            garment_type=piece.garment_type,
            pattern_brand=piece.pattern_brand,
            pattern_id=piece.pattern_id,
            size_label=self._build_size_label(),
            units=piece.units,
            boundary_points=scaled_points,
            grain_line=piece.grain_line,       # grain direction doesn't change
            fold_line=piece.fold_line,         # fold axis doesn't change
            notches=self._scale_notches(piece, scaled_points),
            darts=scaled_darts,
            lengthen_shorten_lines=scaled_ls_lines,
            seam_allowance=piece.seam_allowance,
            vision_scores=piece.vision_scores,
            total_vision_score=piece.total_vision_score,
            band_label=piece.band_label,
            image_source=piece.image_source,
            target_measurements=self.target,
            ease_allowances=self.ease,
        )

        warnings = self._check_for_warnings(piece, scaled, deltas)

        return ScaleResult(
            original_piece=piece,
            scaled_piece=scaled,
            source_profile=self._profile_label(self.source),
            target_profile=self._build_size_label(),
            measurement_deltas=deltas,
            grade_movements=movements,
            warnings=warnings,
        )

    def scale_batch(
        self,
        pieces: list[PatternPiece],
        grade_rules: Optional[list[GradePoint]] = None,
    ) -> list[ScaleResult]:
        """Scale multiple pieces with the same measurement profile."""
        return [self.scale(p, grade_rules) for p in pieces]

    # ── Grading logic ─────────────────────────────────────────────────────────

    def _compute_deltas(self) -> dict[str, float]:
        """
        Compute measurement deltas: target - source.
        Includes ease in the target.
        """
        deltas = {}
        all_keys = set(self.source.keys()) | set(self.target.keys())
        for key in all_keys:
            source_val = self.source.get(key, 0.0)
            target_val = self.target.get(key, 0.0)
            target_with_ease = target_val + self.ease.get(key, 0.0)
            deltas[key] = target_with_ease - source_val
        return deltas

    def _apply_grading(
        self,
        points: list[Point],
        piece: PatternPiece,
        grade_rules: list[GradePoint],
        deltas: dict[str, float],
    ) -> tuple[list[Point], list[dict]]:
        """
        Apply grading rules to boundary points.

        Strategy:
            - Identify boundary regions that correspond to grade landmarks
            - Move those regions by the graded amount
            - Interpolate smoothly between moved regions

        Returns:
            (scaled_points, movements_applied)
        """
        if not points:
            return points, []

        points_arr = np.array(points, dtype=np.float64)
        movements = []

        # Identify boundary regions by their relative position
        bounds = self._boundary_bounds(points)
        width = bounds["max_x"] - bounds["min_x"]
        height = bounds["max_y"] - bounds["min_y"]

        # Build a displacement field — how much each point moves
        displacements = np.zeros_like(points_arr)

        for rule in grade_rules:
            delta = deltas.get(rule.measurement_key, 0.0)
            if abs(delta) < 0.001:
                continue

            dx = rule.x_rate * delta
            dy = rule.y_rate * delta

            # Apply displacement based on landmark region
            region_mask = self._region_mask(
                points_arr, rule.landmark_name, bounds
            )

            displacements[:, 0] += dx * region_mask
            displacements[:, 1] += dy * region_mask

            movements.append({
                "landmark": rule.landmark_name,
                "measurement": rule.measurement_key,
                "delta_measurement": delta,
                "dx_applied": dx,
                "dy_applied": dy,
            })

        scaled_arr = points_arr + displacements
        scaled_points = [(float(p[0]), float(p[1])) for p in scaled_arr]

        return scaled_points, movements

    def _region_mask(
        self,
        points: np.ndarray,
        landmark_name: str,
        bounds: dict,
    ) -> np.ndarray:
        """
        Return a weight mask (0.0-1.0) for each point indicating
        how much the named landmark's grade movement applies to it.

        Uses soft spatial zones based on boundary position.
        """
        n = len(points)
        mask = np.zeros(n)

        width = bounds["max_x"] - bounds["min_x"]
        height = bounds["max_y"] - bounds["min_y"]

        # Normalize point positions to [0,1]
        norm_x = (points[:, 0] - bounds["min_x"]) / (width + 1e-10)
        norm_y = (points[:, 1] - bounds["min_y"]) / (height + 1e-10)

        # Zone definitions — which region of the boundary each landmark covers
        # These are spatial heuristics based on standard pattern geometry
        zones = {
            # Pants
            "waist_side":    (norm_x > 0.7, norm_y > 0.85),
            "waist_center":  (norm_x < 0.3, norm_y > 0.85),
            "hip_side":      (norm_x > 0.7, (norm_y > 0.6) & (norm_y < 0.85)),
            "crotch_front":  (norm_x < 0.5, (norm_y > 0.4) & (norm_y < 0.7)),
            "crotch_back":   (norm_x > 0.5, (norm_y > 0.4) & (norm_y < 0.7)),
            "thigh_side":    (norm_x > 0.7, (norm_y > 0.2) & (norm_y < 0.5)),
            "thigh_inner":   (norm_x < 0.3, (norm_y > 0.2) & (norm_y < 0.5)),
            "hem_side":      (norm_x > 0.5, norm_y < 0.15),
            "hem_inner":     (norm_x < 0.5, norm_y < 0.15),
            # Bodice
            "shoulder_side": (norm_x > 0.7, norm_y > 0.85),
            "bust_side":     (norm_x > 0.7, (norm_y > 0.6) & (norm_y < 0.85)),
            "shoulder_slope":(norm_x > 0.6, norm_y > 0.75),
            "hem_side_bodice":(norm_x > 0.5, norm_y < 0.15),
        }

        zone = zones.get(landmark_name)
        if zone is None:
            return mask

        # Handle tuple of conditions (AND)
        if isinstance(zone, tuple):
            combined = zone[0]
            for condition in zone[1:]:
                combined = combined & condition
            mask[combined] = 1.0
        else:
            mask[zone] = 1.0

        # Smooth the edges of each zone with a falloff
        mask = self._smooth_mask(mask, points)

        return mask

    def _smooth_mask(self, mask: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian-style smoothing to zone boundaries.
        Prevents hard displacement edges that would create kinks.
        """
        smoothed = mask.copy()
        n = len(mask)
        window = min(5, n // 4)

        for i in range(n):
            if mask[i] > 0:
                # Feather the edges of the active zone
                for offset in range(1, window + 1):
                    weight = 1.0 - (offset / (window + 1))
                    prev_idx = (i - offset) % n
                    next_idx = (i + offset) % n
                    if mask[prev_idx] == 0:
                        smoothed[prev_idx] = max(smoothed[prev_idx], weight)
                    if mask[next_idx] == 0:
                        smoothed[next_idx] = max(smoothed[next_idx], weight)

        return smoothed

    # ── Feature scaling ───────────────────────────────────────────────────────

    def _scale_darts(
        self,
        darts: list[Dart],
        deltas: dict[str, float],
        garment_type: str,
    ) -> list[Dart]:
        """
        Scale dart dimensions proportionally to measurement changes.
        Larger bust/waist delta = larger dart intake.
        """
        if not darts:
            return []

        scaled = []
        for dart in darts:
            # Determine which measurement drives this dart
            if garment_type in ("dress", "top", "jacket"):
                drive_key = Measure.BUST
            elif garment_type in ("pants", "skirt"):
                drive_key = Measure.WAIST
            else:
                drive_key = Measure.BUST

            delta = deltas.get(drive_key, 0.0)

            # Scale intake proportionally (rough heuristic: 1/8" per inch)
            intake_change = delta * 0.125
            new_intake = max(0.25, dart.intake_inches + intake_change)

            # Scale depth proportionally
            depth_ratio = new_intake / max(dart.intake_inches, 0.01)
            new_depth = dart.depth_inches * depth_ratio

            # Move apex proportionally (stays at same relative position)
            # For now, apex position scales with the boundary — handled
            # by the overall boundary scaling above.
            scaled.append(Dart(
                apex=dart.apex,
                leg_start=dart.leg_start,
                leg_end=dart.leg_end,
                intake_inches=round(new_intake, 3),
                depth_inches=round(new_depth, 3),
                dart_type=dart.dart_type,
            ))

        return scaled

    def _scale_lengthen_shorten(
        self,
        piece: PatternPiece,
        deltas: dict[str, float],
    ) -> list:
        """
        Update lengthen/shorten line y-positions for length changes.
        These lines mark where to add/remove length in the pattern.
        """
        ls_lines = piece.lengthen_shorten_lines
        if not ls_lines:
            return []

        # Length delta depends on garment type
        if piece.garment_type == "pants":
            length_delta = deltas.get(Measure.INSEAM, 0.0)
        else:
            length_delta = deltas.get(Measure.TORSO_FRONT, 0.0)

        if abs(length_delta) < 0.01:
            return ls_lines

        # Distribute length change evenly across all lengthen/shorten lines
        delta_per_line = length_delta / len(ls_lines)

        scaled = []
        cumulative = 0.0
        for line in ls_lines:
            from .piece import LengthenShortenLine
            cumulative += delta_per_line
            scaled.append(LengthenShortenLine(
                position=line.position,
                y_coordinate=line.y_coordinate + cumulative,
                label=line.label,
            ))

        return scaled

    def _scale_notches(
        self,
        piece: PatternPiece,
        scaled_points: list[Point],
    ) -> list:
        """
        Re-attach notches to scaled boundary points.
        Notch boundary_index stays the same; position updates to match.
        """
        from .piece import Notch
        scaled_notches = []
        for notch in piece.notches:
            idx = notch.boundary_index
            if idx < len(scaled_points):
                scaled_notches.append(Notch(
                    position=scaled_points[idx],
                    boundary_index=idx,
                    notch_type=notch.notch_type,
                    facing_direction=notch.facing_direction,
                ))
        return scaled_notches

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _detect_grade_rules(self, piece: PatternPiece) -> list[GradePoint]:
        """Auto-select grade rules based on garment type."""
        garment = piece.garment_type.lower()
        if garment in ("pants", "shorts", "skirt"):
            return PANTS_GRADE_RULES
        elif garment in ("dress", "top", "jacket", "coat"):
            return BODICE_GRADE_RULES
        elif garment in ("hat", "sock"):
            # Simple uniform scale for accessories
            return []
        else:
            return PANTS_GRADE_RULES  # fallback

    def _boundary_bounds(self, points: list[Point]) -> dict:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return {
            "min_x": min(xs), "max_x": max(xs),
            "min_y": min(ys), "max_y": max(ys),
        }

    def _build_size_label(self) -> str:
        waist = self.target.get(Measure.WAIST)
        inseam = self.target.get(Measure.INSEAM)
        if waist and inseam:
            return f"W{waist:.0f}/I{inseam:.0f}"
        elif waist:
            return f"W{waist:.0f}"
        return "custom"

    def _profile_label(self, measurements: dict) -> str:
        waist = measurements.get(Measure.WAIST)
        inseam = measurements.get(Measure.INSEAM)
        if waist and inseam:
            return f"W{waist:.0f}/I{inseam:.0f}"
        return "source"

    def _check_for_warnings(
        self,
        original: PatternPiece,
        scaled: PatternPiece,
        deltas: dict[str, float],
    ) -> list[str]:
        """Flag anything that deserves human attention."""
        warnings = []

        # Large measurement jumps
        for key, delta in deltas.items():
            if abs(delta) > 4.0:
                warnings.append(
                    f"Large {key} change ({delta:+.1f}\"). "
                    f"Consider intermediate fitting."
                )

        # Muscle-specific warnings
        thigh_delta = deltas.get(Measure.THIGH, 0.0)
        if thigh_delta > 2.0:
            warnings.append(
                "Thigh increase > 2\". Check crotch curve depth — "
                "muscular thighs may need extra crotch ease."
            )

        bicep_delta = deltas.get(Measure.BICEP, 0.0)
        if bicep_delta > 1.5:
            warnings.append(
                "Bicep increase > 1.5\". Sleeve cap may need reshaping "
                "for muscular arm definition."
            )

        # Dart count check for larger bust
        bust_delta = deltas.get(Measure.BUST, 0.0)
        if bust_delta > 2.0 and not original.has_darts:
            warnings.append(
                "Bust increase > 2\" but no darts in original pattern. "
                "Consider adding a dart for fit."
            )

        return warnings
