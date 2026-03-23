"""
PatternEncoder: Wraps GeometricEncoder from Geometric-to-Binary-Computational-Bridge
for use with sewing pattern pieces.

Converts PatternPiece boundary points into geometric token sequences.
Straight edges get sparse tokens, curved edges get dense tokens.
SpatialGrid controls adaptive resolution automatically.

Depends on:
    Geometric-to-Binary-Computational-Bridge/GEIS/geometric_encoder.py
    Geometric-to-Binary-Computational-Bridge/engine/spatial_grid.py
    Geometric-to-Binary-Computational-Bridge/engine/symmetry_detector.py
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

# Import from Geometric-to-Binary-Computational-Bridge
# Adjust these paths once the repo is copied into pattern_geometry/
from .geometric_encoder import GeometricEncoder
from .octahedral_state import OctahedralState
from .spatial_grid import SpatialGrid
from .symmetry_detector import SymmetryDetector

from .piece import PatternPiece, Point, GrainLine, FoldLine


# ── Constants ─────────────────────────────────────────────────────────────────

# Curvature threshold — boundary segments above this are treated as curves
# and get denser token sampling. Below this = straight edge, sparse tokens.
CURVE_THRESHOLD = 0.15          # radians per unit length

# Minimum points we'll encode even on a dead-straight edge
MIN_POINTS_PER_EDGE = 2

# Pattern coordinate space maps to octahedral space via this scale factor.
# 1 inch in pattern space = SCALE units in octahedral space.
# Keeps coordinates within the [-0.25, 0.25] range of OctahedralState.POSITIONS.
COORDINATE_SCALE = 0.01

# SpatialGrid config for pattern use
ADAPTIVE_THRESHOLD = 0.3        # lower = more aggressive refinement near curves
MAX_DEPTH = 5                   # max octree depth for curve resolution


# ── PatternEncoder ────────────────────────────────────────────────────────────

class PatternEncoder:
    """
    Encodes a PatternPiece's boundary into geometric tokens.

    Pipeline:
        boundary_points
            → curvature analysis (identify straight vs curved segments)
            → adaptive point selection (SpatialGrid)
            → coordinate normalization
            → OctahedralState mapping
            → GeometricEncoder tokenization
            → PatternPiece.encoded_tokens

    Also runs SymmetryDetector to confirm fold lines and mirror symmetry,
    halving the encoding for fold pieces.

    Args:
        vertex_width: Bits per vertex address (default 3 = 8 states).
        curve_threshold: Curvature above which a segment is treated as curved.
        detect_symmetry: Whether to run SymmetryDetector on boundary.
    """

    def __init__(
        self,
        vertex_width: int = 3,
        curve_threshold: float = CURVE_THRESHOLD,
        detect_symmetry: bool = True,
    ):
        self.encoder = GeometricEncoder(vertex_width=vertex_width)
        self.grid = SpatialGrid(
            adaptive_threshold=ADAPTIVE_THRESHOLD,
            max_depth=MAX_DEPTH,
        )
        self.symmetry_detector = SymmetryDetector() if detect_symmetry else None
        self.curve_threshold = curve_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, piece: PatternPiece) -> PatternPiece:
        """
        Encode a PatternPiece's boundary into geometric tokens.
        Mutates piece.encoded_tokens and piece.symmetry_detected in place.
        Returns the piece for chaining.

        Args:
            piece: PatternPiece with boundary_points populated.

        Returns:
            Same piece with encoded_tokens filled.

        Raises:
            ValueError: If piece has fewer than 3 boundary points.
        """
        if len(piece.boundary_points) < 3:
            raise ValueError(
                f"PatternPiece '{piece.name}' needs at least 3 boundary points "
                f"to encode. Got {len(piece.boundary_points)}."
            )

        # Step 1: Check for symmetry — halve the work if fold piece
        working_points = self._apply_symmetry(piece)

        # Step 2: Select encoding points adaptively
        selected = self._select_encoding_points(working_points)

        # Step 3: Normalize to octahedral coordinate space
        normalized = self._normalize_points(selected)

        # Step 4: Map each point to an OctahedralState
        states = self._points_to_states(normalized)

        # Step 5: Assign operators based on boundary role
        operators = self._assign_operators(working_points, selected)

        # Step 6: Assign symbols based on feature context
        symbols = self._assign_symbols(piece, working_points, selected)

        # Step 7: Generate tokens
        tokens = []
        for state, operator, symbol in zip(states, operators, symbols):
            token = state.to_token(operator=operator, symbol=symbol)
            # Validate round-trip
            if self.encoder.validate_token(token):
                tokens.append(token)
            else:
                # Fallback to safe default token for this state
                tokens.append(state.to_token(operator="|", symbol="O"))

        piece.encoded_tokens = tokens
        return piece

    def decode(self, piece: PatternPiece) -> list[Point]:
        """
        Decode a piece's encoded_tokens back to boundary points.
        If symmetry was detected, mirrors the decoded points.

        Args:
            piece: PatternPiece with encoded_tokens populated.

        Returns:
            List of (x, y) boundary points in original coordinate space.
        """
        if not piece.encoded_tokens:
            return []

        points = []
        for token in piece.encoded_tokens:
            state = OctahedralState.from_token(token)
            # Convert octahedral position back to 2D pattern coordinates
            x = state.position[0] / COORDINATE_SCALE
            y = state.position[1] / COORDINATE_SCALE
            points.append((x, y))

        # Mirror if fold piece
        if piece.symmetry_detected and piece.fold_line is not None:
            points = self._mirror_points(points, piece.fold_line)

        return points

    def encode_batch(self, pieces: list[PatternPiece]) -> list[PatternPiece]:
        """Encode multiple pieces. Returns list for chaining."""
        return [self.encode(p) for p in pieces]

    # ── Symmetry ──────────────────────────────────────────────────────────────

    def _apply_symmetry(self, piece: PatternPiece) -> list[Point]:
        """
        Check for symmetry and return the working point set.
        For fold pieces, returns only half the boundary.
        Sets piece.symmetry_detected.
        """
        points = piece.boundary_points

        if self.symmetry_detector is None:
            return points

        # Convert 2D boundary points to 3D sources for SymmetryDetector
        # (z=0 for all pattern points — they live in a plane)
        sources = [
            {
                "position": [p[0], p[1], 0.0],
                "strength": 1.0,
                "type": "charge",
            }
            for p in points
        ]

        bounds = self._compute_bounds(points)
        symmetries = self.symmetry_detector.findSymmetries(sources, bounds)

        # Look for a reflective symmetry that matches the fold line
        has_mirror = any(s["type"] == "reflective" for s in symmetries)

        if has_mirror and piece.fold_line is not None:
            piece.symmetry_detected = True
            # Return only the first half of the boundary
            midpoint = len(points) // 2
            return points[:midpoint + 1]

        # No fold symmetry — encode the full boundary
        piece.symmetry_detected = False
        return points

    # ── Point Selection ───────────────────────────────────────────────────────

    def _select_encoding_points(self, points: list[Point]) -> list[int]:
        """
        Select which boundary point indices to encode.
        Straight segments → sparse (just endpoints).
        Curved segments → dense (all points).

        Returns list of indices into the points list.
        """
        if len(points) <= 4:
            return list(range(len(points)))

        selected = [0]  # always include start
        curvatures = self._compute_curvatures(points)

        i = 1
        while i < len(points) - 1:
            if curvatures[i] > self.curve_threshold:
                # In a curved region — include this point
                selected.append(i)
                i += 1
            else:
                # In a straight region — skip to next curve or endpoint
                # Find where the straight run ends
                j = i
                while j < len(points) - 1 and curvatures[j] <= self.curve_threshold:
                    j += 1
                # Include the endpoint of the straight run
                selected.append(j - 1)
                i = j

        selected.append(len(points) - 1)  # always include end

        # Deduplicate while preserving order
        seen = set()
        return [x for x in selected if not (x in seen or seen.add(x))]

    def _compute_curvatures(self, points: list[Point]) -> list[float]:
        """
        Estimate curvature at each boundary point using the
        angle between successive edge vectors.
        Returns list of curvature values (radians per unit).
        """
        n = len(points)
        curvatures = [0.0] * n

        for i in range(1, n - 1):
            p0 = np.array(points[i - 1])
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])

            v1 = p1 - p0
            v2 = p2 - p1

            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)

            if len1 < 1e-10 or len2 < 1e-10:
                curvatures[i] = 0.0
                continue

            cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
            angle = math.acos(cos_angle)
            avg_len = (len1 + len2) / 2
            curvatures[i] = angle / (avg_len + 1e-10)

        return curvatures

    # ── Coordinate Normalization ──────────────────────────────────────────────

    def _normalize_points(self, indices: list[int]) -> list[tuple[float, float, float]]:
        """
        This method signature is intentionally left incomplete here —
        it receives indices but needs the original points list.
        See _encode_points_at_indices() for the full call.

        Normalize selected boundary points into octahedral coordinate space.
        Maps (x, y) in inches → (x, y, z) in [-0.25, 0.25].
        z is always 0 — patterns live in a plane.
        """
        # Note: this is called via _encode_points_at_indices with points passed in
        raise NotImplementedError("Call _encode_points_at_indices instead")

    def _encode_points_at_indices(
        self,
        points: list[Point],
        indices: list[int],
    ) -> list[tuple[float, float, float]]:
        """
        Normalize selected points to octahedral coordinate space.
        """
        selected_pts = [points[i] for i in indices]

        if not selected_pts:
            return []

        # Find bounding box for normalization
        xs = [p[0] for p in selected_pts]
        ys = [p[1] for p in selected_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max(max_x - min_x, 1e-10)
        range_y = max(max_y - min_y, 1e-10)

        # Scale to [-0.25, 0.25] — octahedral coordinate range
        normalized = []
        for x, y in selected_pts:
            nx = ((x - min_x) / range_x - 0.5) * 0.5
            ny = ((y - min_y) / range_y - 0.5) * 0.5
            normalized.append((nx, ny, 0.0))

        return normalized

    # ── State Mapping ─────────────────────────────────────────────────────────

    def _points_to_states(
        self, normalized: list[tuple[float, float, float]]
    ) -> list[OctahedralState]:
        """
        Map normalized 3D points to the nearest OctahedralState.
        Each point snaps to the closest of the 8 vertex positions.
        """
        states = []
        for nx, ny, nz in normalized:
            point = np.array([nx, ny, nz])
            best_idx = 0
            best_dist = float("inf")

            for idx in range(8):
                state = OctahedralState(idx)
                dist = np.linalg.norm(point - state.position)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            states.append(OctahedralState(best_idx))

        return states

    # ── Operator and Symbol Assignment ────────────────────────────────────────

    def _assign_operators(
        self,
        points: list[Point],
        selected_indices: list[int],
    ) -> list[str]:
        """
        Assign operators to each selected point based on its role.

        '|' (radial) → points that are landmarks: corners, notch positions,
                        dart endpoints, grain line intersections.
        '/' (tangential) → points on smooth curves between landmarks.
        """
        operators = []
        n = len(points)

        for i in selected_indices:
            # Corners and endpoints are radial (landmarks)
            if i == 0 or i == n - 1:
                operators.append("|")
                continue

            # Check for high curvature (corner/landmark)
            curvatures = self._compute_curvatures(points)
            if curvatures[i] > self.curve_threshold * 2:
                operators.append("|")  # sharp corner = radial landmark
            else:
                operators.append("/")  # smooth curve = tangential

        return operators

    def _assign_symbols(
        self,
        piece: PatternPiece,
        points: list[Point],
        selected_indices: list[int],
    ) -> list[str]:
        """
        Assign geometric symbols based on feature context.

        'O' (Octahedral) → standard boundary point
        'I' (Inversion)  → point near a notch
        'X' (Exchange)   → point near a dart
        'Δ' (Delta)      → point near grain line or fold line intersection
        """
        symbols = []

        # Build lookup sets for feature-proximate points
        notch_positions = {n.boundary_index for n in piece.notches}
        dart_boundary_indices = self._dart_boundary_indices(piece, points)

        for i in selected_indices:
            if i in notch_positions:
                symbols.append("I")      # notch = inversion marker
            elif i in dart_boundary_indices:
                symbols.append("X")      # dart = exchange marker
            elif self._near_grain_or_fold(i, piece, points):
                symbols.append("Δ")      # grain/fold intersection = delta
            else:
                symbols.append("O")      # standard point

        return symbols

    def _dart_boundary_indices(
        self, piece: PatternPiece, points: list[Point]
    ) -> set[int]:
        """Find boundary point indices near dart leg endpoints."""
        indices = set()
        for dart in piece.darts:
            for leg_pt in [dart.leg_start, dart.leg_end]:
                closest = self._nearest_boundary_index(leg_pt, points)
                indices.add(closest)
        return indices

    def _near_grain_or_fold(
        self, idx: int, piece: PatternPiece, points: list[Point]
    ) -> bool:
        """True if boundary point is near the grain line or fold line."""
        pt = np.array(points[idx])
        threshold = 0.5  # inches

        if piece.grain_line:
            grain_start = np.array(piece.grain_line.start)
            grain_end = np.array(piece.grain_line.end)
            if (np.linalg.norm(pt - grain_start) < threshold or
                    np.linalg.norm(pt - grain_end) < threshold):
                return True

        if piece.fold_line:
            fold_start = np.array(piece.fold_line.start)
            fold_end = np.array(piece.fold_line.end)
            if (np.linalg.norm(pt - fold_start) < threshold or
                    np.linalg.norm(pt - fold_end) < threshold):
                return True

        return False

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _nearest_boundary_index(
        self, target: Point, points: list[Point]
    ) -> int:
        """Return index of boundary point closest to target."""
        target_arr = np.array(target)
        best_idx = 0
        best_dist = float("inf")
        for i, pt in enumerate(points):
            d = np.linalg.norm(np.array(pt) - target_arr)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _compute_bounds(self, points: list[Point]) -> dict:
        """Compute bounding box dict for SpatialGrid."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return {
            "min": [min(xs), min(ys), -0.1],
            "max": [max(xs), max(ys),  0.1],
        }

    def _mirror_points(
        self, points: list[Point], fold_line: FoldLine
    ) -> list[Point]:
        """
        Mirror a list of points across the fold line axis.
        Returns original points + mirrored points as a closed boundary.
        """
        mirrored = []
        for x, y in points:
            if fold_line.axis == "vertical":
                mirrored.append((-x, y))
            elif fold_line.axis == "horizontal":
                mirrored.append((x, -y))
            else:
                # Diagonal or custom — just return original for now
                mirrored.append((x, y))

        # Combine: original forward + mirror in reverse (closed polygon)
        return points + list(reversed(mirrored[:-1]))

    def token_summary(self, piece: PatternPiece) -> dict:
        """
        Return a summary of token composition for a piece.
        Useful for debugging and validation.
        """
        if not piece.encoded_tokens:
            return {"error": "No tokens encoded yet"}

        operators = {"radial": 0, "tangential": 0}
        symbols = {"O": 0, "I": 0, "X": 0, "Δ": 0}
        states = {}

        for token in piece.encoded_tokens:
            vertex_bits, op, symbol = self.encoder.get_components(token)
            operators["radial" if op == "|" else "tangential"] += 1
            symbols[symbol] = symbols.get(symbol, 0) + 1
            states[vertex_bits] = states.get(vertex_bits, 0) + 1

        return {
            "piece": piece.name,
            "total_tokens": len(piece.encoded_tokens),
            "symmetry_applied": piece.symmetry_detected,
            "operators": operators,
            "symbols": symbols,
            "state_distribution": states,
        }
