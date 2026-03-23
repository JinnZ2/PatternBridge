"""
SpatialGrid: Adaptive spatial decomposition using an octree.

Ported from Geometric-to-Binary-Computational-Bridge/Engine/spatial_grid.py
for use in PatternBridge's pattern encoding pipeline.

Regions near sources are subdivided to higher resolution; distant regions
stay coarse. This maps naturally to octahedral geometry (8 children per node).

Also includes a curvature-adaptive refine() method for 2D boundary point
densification used in PatternBridge's boundary pipeline.
"""

from __future__ import annotations

import math
import numpy as np


class SpatialGrid:
    """Adaptive spatial grid with octree decomposition and boundary refinement."""

    def __init__(self, adaptive_threshold: float = 0.5, max_depth: int = 4):
        self.adaptive_threshold = adaptive_threshold
        self.max_depth = max_depth

    # ── Octree decomposition (from Geometric-to-Binary) ──────────────────────

    def adaptiveDecomposition(self, bounds: dict, sources: list[dict]) -> list[dict]:
        """
        Recursively subdivide the domain, refining near sources.

        Args:
            bounds: dict with 'min' and 'max' (3-element lists).
            sources: list of source dicts with 'position' and 'strength'.

        Returns:
            list of region dicts, each containing grid points for field eval.
        """
        regions: list[dict] = []
        self._subdivide(bounds, sources, depth=0, regions=regions)
        return regions

    def _subdivide(
        self,
        bounds: dict,
        sources: list[dict],
        depth: int,
        regions: list[dict],
    ) -> None:
        """Recursive octree subdivision."""
        bmin = np.array(bounds["min"])
        bmax = np.array(bounds["max"])
        center = (bmin + bmax) / 2
        size = float(np.linalg.norm(bmax - bmin))

        # Calculate minimum distance from any source to this cell center
        min_dist = float("inf")
        for s in sources:
            pos = np.array(s["position"])
            dist = float(np.linalg.norm(pos - center))
            min_dist = min(min_dist, dist)

        # Field influence metric: should we refine this cell?
        influence = size / (min_dist + 1e-10)
        should_refine = influence > self.adaptive_threshold and depth < self.max_depth

        if should_refine:
            # Subdivide into 8 octants (octahedral decomposition)
            for i in range(8):
                child_min = np.array([
                    bmin[0] if (i & 1) == 0 else center[0],
                    bmin[1] if (i & 2) == 0 else center[1],
                    bmin[2] if (i & 4) == 0 else center[2],
                ])
                child_max = np.array([
                    center[0] if (i & 1) == 0 else bmax[0],
                    center[1] if (i & 2) == 0 else bmax[1],
                    center[2] if (i & 4) == 0 else bmax[2],
                ])
                child_bounds = {
                    "min": child_min.tolist(),
                    "max": child_max.tolist(),
                }
                self._subdivide(child_bounds, sources, depth + 1, regions)
        else:
            # Leaf node: create evaluation region with grid points
            region = self.createRegion(bounds, sources)
            regions.append(region)

    def createRegion(self, bounds: dict, sources: list[dict]) -> dict:
        """Create a leaf region with sample points for field evaluation."""
        bmin = np.array(bounds["min"])
        bmax = np.array(bounds["max"])
        center = (bmin + bmax) / 2
        size = float(np.linalg.norm(bmax - bmin))

        # Single sample point at cell center
        points = [center.tolist()]

        # Estimate field intensity at center for priority sorting
        field_intensity = 0.0
        for s in sources:
            pos = np.array(s["position"])
            dist = float(np.linalg.norm(pos - center))
            strength = abs(s.get("strength", 1.0))
            field_intensity += strength / (dist * dist + 1e-20)

        return {
            "bounds": bounds,
            "center": center.tolist(),
            "points": points,
            "fieldIntensity": float(field_intensity),
            "size": size,
        }

    def generateUniformGrid(self, bounds: dict, resolution: int) -> list[list[float]]:
        """
        Generate a uniform 3D grid of evaluation points.
        Used as fallback when adaptive decomposition isn't needed.

        Args:
            bounds: dict with 'min' and 'max'.
            resolution: number of points per axis.

        Returns:
            list of [x, y, z] points.
        """
        bmin = np.array(bounds["min"])
        bmax = np.array(bounds["max"])

        x = np.linspace(bmin[0], bmax[0], resolution)
        y = np.linspace(bmin[1], bmax[1], resolution)
        z = np.linspace(bmin[2], bmax[2], resolution)

        points = []
        for xi in x:
            for yi in y:
                for zi in z:
                    points.append([float(xi), float(yi), float(zi)])

        return points

    # ── Curvature-adaptive refinement (PatternBridge extension) ──────────────

    def refine(self, points: list, bounds: dict) -> list:
        """
        Refine point list by inserting midpoints in high-curvature regions.

        For each triplet of consecutive points, if the angle between the
        two edges exceeds adaptive_threshold, insert a midpoint on each
        edge. Repeats up to max_depth times until no further refinement
        is needed.

        Args:
            points: List of points (2D or 3D tuples/lists).
            bounds: Dict with "min" and "max" keys (bounding box).

        Returns:
            Refined point list with denser sampling around curves.
        """
        if len(points) < 3:
            return points

        dim = len(points[0])
        refined = list(points)

        for _ in range(self.max_depth):
            new_points = [refined[0]]
            inserted_any = False

            for i in range(1, len(refined) - 1):
                prev = refined[i - 1]
                curr = refined[i]
                nxt = refined[i + 1]

                curvature = self._angle_at(prev, curr, nxt)

                if curvature > self.adaptive_threshold:
                    # Insert midpoint before curr
                    mid_before = tuple(
                        (prev[d] + curr[d]) / 2 for d in range(dim)
                    )
                    new_points.append(mid_before)
                    new_points.append(curr)
                    # Insert midpoint after curr
                    mid_after = tuple(
                        (curr[d] + nxt[d]) / 2 for d in range(dim)
                    )
                    new_points.append(mid_after)
                    inserted_any = True
                else:
                    new_points.append(curr)

            new_points.append(refined[-1])
            refined = new_points

            if not inserted_any:
                break

        return refined

    @staticmethod
    def _angle_at(p0, p1, p2) -> float:
        """Compute the turning angle (in radians) at p1 between edges p0->p1 and p1->p2."""
        dim = min(len(p0), len(p1), len(p2))

        v1 = [p1[d] - p0[d] for d in range(dim)]
        v2 = [p2[d] - p1[d] for d in range(dim)]

        len1 = math.sqrt(sum(x * x for x in v1))
        len2 = math.sqrt(sum(x * x for x in v2))

        if len1 < 1e-10 or len2 < 1e-10:
            return 0.0

        dot = sum(v1[d] * v2[d] for d in range(dim))
        cos_angle = max(-1.0, min(1.0, dot / (len1 * len2)))
        return math.acos(cos_angle)
