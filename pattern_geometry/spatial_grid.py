"""
SpatialGrid stub — minimal interface matching Geometric-to-Binary-Computational-Bridge.

Adaptive spatial partitioning for controlling encoding resolution.
Curves get denser point sampling; straight edges stay sparse.

Replace this file with the real implementation from:
    https://github.com/JinnZ2/Geometric-to-Binary-Computational-Bridge
"""

from __future__ import annotations

import math


class SpatialGrid:
    """
    Adaptive spatial grid that controls encoding resolution.

    Higher curvature regions get finer grid cells (more tokens).
    Straight regions get coarser cells (fewer tokens).
    """

    def __init__(
        self,
        adaptive_threshold: float = 0.3,
        max_depth: int = 5,
    ):
        self.adaptive_threshold = adaptive_threshold
        self.max_depth = max_depth

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
        """Compute the turning angle (in radians) at p1 between edges p0→p1 and p1→p2."""
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
