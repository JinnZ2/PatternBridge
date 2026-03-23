"""
PatternPiece: Core data structure for PatternBridge.

A PatternPiece represents one pattern piece extracted from an image,
encoded into geometric tokens, and ready for parametric scaling and
multi-format export.

This is the central object the entire pipeline operates on:
    vision output → PatternPiece → geometry encoding → scaled output
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


# ── Primitive geometry types ──────────────────────────────────────────────────

Point = tuple[float, float]         # (x, y) in real-world units
Point3D = tuple[float, float, float]  # (x, y, z) for octahedral encoding


@dataclass
class GrainLine:
    """Straight line indicating fabric grain direction."""
    start: Point
    end: Point
    angle_degrees: float            # 0 = horizontal, 90 = vertical
    is_bidirectional: bool = True   # most grain lines have arrows both ends


@dataclass
class FoldLine:
    """
    Edge along which the pattern is placed on folded fabric.
    When present, the piece is mirrored across this axis on cut.
    SymmetryDetector uses this to halve the encoding.
    """
    start: Point
    end: Point
    axis: str                       # "vertical" | "horizontal" | "diagonal"
    position: str                   # "left" | "right" | "top" | "bottom" | "custom"


@dataclass
class Notch:
    """
    Small triangular or rectangular mark on seam line.
    Used to align pieces during construction.
    """
    position: Point                 # location on boundary
    boundary_index: int             # index into boundary_points list
    notch_type: str = "single"      # "single" | "double" | "triple"
    facing_direction: str = "out"   # "out" (cut out) | "in" (clip in)


@dataclass
class Dart:
    """
    Fold of fabric stitched to shape the piece to body contours.
    Defined by apex (point), two leg lines, and intake width.
    """
    apex: Point                     # tip of the dart (where it ends)
    leg_start: Point                # one end of dart opening on seam
    leg_end: Point                  # other end of dart opening on seam
    intake_inches: float            # width at the opening
    depth_inches: float             # length from opening to apex
    dart_type: str = "standard"     # "standard" | "french" | "contour"


@dataclass
class SeamAllowance:
    """
    Seam allowance specification — global with optional edge overrides.
    Different edges can have different allowances (e.g., hem vs side seam).
    """
    global_inches: float            # default for all edges
    edge_overrides: dict[str, float] = field(default_factory=dict)
    # edge_overrides keys: "top" | "bottom" | "left" | "right" | "curve_N"

    def for_edge(self, edge_name: str) -> float:
        """Return seam allowance for a specific edge, falling back to global."""
        return self.edge_overrides.get(edge_name, self.global_inches)


@dataclass
class LengthenShortenLine:
    """
    Horizontal line indicating where to add or remove length.
    Common in commercial patterns for height adjustments.
    """
    position: Point                 # start of line (left edge)
    y_coordinate: float             # y position in piece coordinates
    label: str = "LENGTHEN OR SHORTEN HERE"


# ── Core PatternPiece ─────────────────────────────────────────────────────────

@dataclass
class PatternPiece:
    """
    A single pattern piece — the central object in PatternBridge.

    Populated by the vision layer, encoded by the geometry layer,
    scaled by the scaler, and consumed by the output layer.

    Units: inches by default. Set units="cm" if working metric.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    name: str                               # e.g. "FRONT", "BACK", "SOLE"
    piece_number: Optional[int]             # e.g. 1, 2, 3
    cut_quantity: int = 1                   # how many times to cut this piece
    garment_type: str = "unknown"           # "pants" | "dress" | "hat" | "sock" | ...
    pattern_brand: str = "unknown"          # "Butterick" | "McCall" | "handdrawn" | ...
    pattern_id: str = ""                    # e.g. "M8171", "S-5474"
    size_label: str = ""                    # e.g. "Size 2", "One Size", "36/36"
    units: str = "inches"                   # "inches" | "cm"

    # ── Geometry ──────────────────────────────────────────────────────────────
    boundary_points: list[Point] = field(default_factory=list)
    # Ordered (x, y) coordinates tracing the full outer boundary clockwise.
    # First and last points are the same (closed polygon).
    # Straight edges: 2 points. Curves: as many as needed for accuracy.

    grain_line: Optional[GrainLine] = None
    fold_line: Optional[FoldLine] = None
    notches: list[Notch] = field(default_factory=list)
    darts: list[Dart] = field(default_factory=list)
    lengthen_shorten_lines: list[LengthenShortenLine] = field(default_factory=list)
    seam_allowance: Optional[SeamAllowance] = None

    # ── Encoding (populated by geometry layer) ────────────────────────────────
    encoded_tokens: list[str] = field(default_factory=list)
    # Geometric tokens from GeometricEncoder — one per boundary landmark.
    # e.g. ["001|O", "010/X", "110|Δ", ...]

    symmetry_detected: bool = False
    # True if SymmetryDetector confirmed fold line or mirror symmetry.
    # When True, encoded_tokens represent only half the boundary.

    encoding_resolution: str = "adaptive"
    # "adaptive" = SpatialGrid controls density
    # "uniform"  = fixed spacing
    # "manual"   = points set explicitly

    # ── Vision analysis metadata ──────────────────────────────────────────────
    vision_scores: dict[str, float] = field(default_factory=dict)
    # Raw rubric scores from PatternPromptEvaluator.
    # Keys match rubric category names.

    total_vision_score: float = 0.0
    band_label: str = ""
    # e.g. "Complete — all features precisely defined, ready for encoding"

    image_source: str = ""          # path to source image
    image_quality_notes: str = ""   # any flags from vision analysis

    # ── Scale parameters (populated by scaler) ────────────────────────────────
    target_measurements: dict[str, float] = field(default_factory=dict)
    # e.g. {"waist": 24.0, "hip": 34.0, "inseam": 28.0, "bust": 32.0}

    ease_allowances: dict[str, float] = field(default_factory=dict)
    # Added to body measurements for movement/comfort.
    # e.g. {"waist": 0.5, "hip": 1.0, "bust": 2.0}

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def is_cut_on_fold(self) -> bool:
        """True if this piece is placed on folded fabric."""
        return self.fold_line is not None

    @property
    def full_cut_quantity(self) -> int:
        """
        Actual pieces cut from fabric.
        Fold pieces yield 2 mirror pieces per cut.
        """
        return self.cut_quantity * (2 if self.is_cut_on_fold else 1)

    @property
    def has_darts(self) -> bool:
        return len(self.darts) > 0

    @property
    def boundary_point_count(self) -> int:
        return len(self.boundary_points)

    @property
    def is_encodable(self) -> bool:
        """True if piece has enough data to attempt geometric encoding."""
        return (
            len(self.boundary_points) >= 3
            and self.total_vision_score >= 51.0
        )

    @property
    def needs_better_image(self) -> bool:
        """True if vision score too low to reliably encode."""
        return self.total_vision_score < 51.0

    def effective_measurement(self, key: str) -> Optional[float]:
        """
        Body measurement + ease for a given dimension.
        Returns None if measurement not set.
        """
        if key not in self.target_measurements:
            return None
        return self.target_measurements[key] + self.ease_allowances.get(key, 0.0)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to plain Python dict for JSON export."""
        return {
            "name": self.name,
            "piece_number": self.piece_number,
            "cut_quantity": self.cut_quantity,
            "full_cut_quantity": self.full_cut_quantity,
            "garment_type": self.garment_type,
            "pattern_brand": self.pattern_brand,
            "pattern_id": self.pattern_id,
            "size_label": self.size_label,
            "units": self.units,
            "boundary_points": self.boundary_points,
            "grain_line": {
                "start": self.grain_line.start,
                "end": self.grain_line.end,
                "angle_degrees": self.grain_line.angle_degrees,
            } if self.grain_line else None,
            "fold_line": {
                "start": self.fold_line.start,
                "end": self.fold_line.end,
                "axis": self.fold_line.axis,
                "position": self.fold_line.position,
            } if self.fold_line else None,
            "notches": [
                {
                    "position": n.position,
                    "boundary_index": n.boundary_index,
                    "notch_type": n.notch_type,
                }
                for n in self.notches
            ],
            "darts": [
                {
                    "apex": d.apex,
                    "leg_start": d.leg_start,
                    "leg_end": d.leg_end,
                    "intake_inches": d.intake_inches,
                    "depth_inches": d.depth_inches,
                    "dart_type": d.dart_type,
                }
                for d in self.darts
            ],
            "seam_allowance": {
                "global_inches": self.seam_allowance.global_inches,
                "edge_overrides": self.seam_allowance.edge_overrides,
            } if self.seam_allowance else None,
            "lengthen_shorten_lines": [
                {"y_coordinate": l.y_coordinate, "label": l.label}
                for l in self.lengthen_shorten_lines
            ],
            "encoded_tokens": self.encoded_tokens,
            "symmetry_detected": self.symmetry_detected,
            "vision_scores": self.vision_scores,
            "total_vision_score": self.total_vision_score,
            "band_label": self.band_label,
            "target_measurements": self.target_measurements,
            "ease_allowances": self.ease_allowances,
            "image_source": self.image_source,
            "image_quality_notes": self.image_quality_notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_vision_result(cls, result: dict, image_source: str = "") -> "PatternPiece":
        """
        Construct a PatternPiece from PatternPromptEvaluator output.
        Boundary points and encoding are empty — filled by geometry layer.
        """
        # Map seam allowance from vision result
        sa_inches = result.get("seam_allowance_inches")
        seam_allowance = SeamAllowance(global_inches=sa_inches) if sa_inches else None

        # Pull vision scores
        vision_scores = {
            k.replace("score_", ""): v
            for k, v in result.items()
            if k.startswith("score_")
        }

        return cls(
            name=result.get("piece_name") or "UNKNOWN",
            piece_number=result.get("piece_number"),
            cut_quantity=result.get("cut_quantity") or 1,
            garment_type=result.get("garment_type") or "unknown",
            pattern_brand=result.get("pattern_brand") or "unknown",
            seam_allowance=seam_allowance,
            vision_scores=vision_scores,
            total_vision_score=result.get("total_score", 0.0),
            band_label=result.get("band_label", ""),
            image_source=image_source,
            image_quality_notes=result.get("image_quality_notes") or "",
        )

    def __repr__(self) -> str:
        fold = " [fold]" if self.is_cut_on_fold else ""
        encoded = f" encoded={len(self.encoded_tokens)}tok" if self.encoded_tokens else ""
        return (
            f"PatternPiece({self.name!r} #{self.piece_number}"
            f" cut×{self.cut_quantity}{fold}"
            f" pts={self.boundary_point_count}"
            f"{encoded}"
            f" score={self.total_vision_score:.0f})"
        )
