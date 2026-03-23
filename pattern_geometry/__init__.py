"""Pattern geometry layer — piece data structures, encoding, and scaling."""

from .piece import PatternPiece, Point
from .encoder import PatternEncoder
from .scaler import PatternScaler
from .boundary import generate_boundary, generate_boundary_batch

__all__ = [
    "PatternPiece", "Point", "PatternEncoder", "PatternScaler",
    "generate_boundary", "generate_boundary_batch",
]
