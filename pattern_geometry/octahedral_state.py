"""
OctahedralState stub — minimal interface matching Geometric-to-Binary-Computational-Bridge.

Represents one of 8 vertices of an octahedron in [-0.25, 0.25] space.
Replace this file with the real implementation from:
    https://github.com/JinnZ2/Geometric-to-Binary-Computational-Bridge
"""

from __future__ import annotations

import numpy as np


# 8 vertices of a regular octahedron inscribed in [-0.25, 0.25]^3
_POSITIONS = np.array([
    [ 0.25,  0.00,  0.00],   # 0: +X
    [-0.25,  0.00,  0.00],   # 1: -X
    [ 0.00,  0.25,  0.00],   # 2: +Y
    [ 0.00, -0.25,  0.00],   # 3: -Y
    [ 0.00,  0.00,  0.25],   # 4: +Z
    [ 0.00,  0.00, -0.25],   # 5: -Z
    [ 0.125, 0.125, 0.00],   # 6: +XY edge midpoint
    [-0.125, 0.125, 0.00],   # 7: -XY edge midpoint
], dtype=np.float64)


class OctahedralState:
    """
    One of 8 octahedral vertex states used for geometric token encoding.

    Each state has:
        - An index (0-7)
        - A 3D position in [-0.25, 0.25] space
        - Token serialization: "{vertex_bits}{operator}{symbol}"
    """

    POSITIONS = _POSITIONS

    def __init__(self, index: int):
        if not 0 <= index < 8:
            raise ValueError(f"OctahedralState index must be 0-7, got {index}")
        self.index = index
        self.position: np.ndarray = _POSITIONS[index].copy()

    def to_token(self, operator: str = "|", symbol: str = "O") -> str:
        """Serialize this state to a token string."""
        vertex_bits = format(self.index, "03b")
        return f"{vertex_bits}{operator}{symbol}"

    @classmethod
    def from_token(cls, token: str, vertex_width: int = 3) -> OctahedralState:
        """Deserialize a token string back to an OctahedralState."""
        vertex_bits = token[:vertex_width]
        index = int(vertex_bits, 2)
        return cls(index)

    def __repr__(self) -> str:
        return f"OctahedralState({self.index}, pos={self.position.tolist()})"
