"""Pattern geometry layer — piece data structures, encoding, and scaling."""

from .piece import PatternPiece, Point
from .encoder import PatternEncoder
from .scaler import PatternScaler

__all__ = ["PatternPiece", "Point", "PatternEncoder", "PatternScaler"]
