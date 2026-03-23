"""
Quick start example: Minimal end-to-end pipeline.

Creates a single pattern piece, encodes it, scales it,
and exports to SVG — all in ~20 lines.
"""

from pattern_geometry.piece import PatternPiece, GrainLine, SeamAllowance
from pattern_geometry.encoder import PatternEncoder
from pattern_geometry.scaler import PatternScaler
from pattern_output.svg_writer import SVGWriter


# 1. Define a pattern piece
piece = PatternPiece(
    name="BODICE_FRONT",
    piece_number=1,
    garment_type="dress",
    boundary_points=[
        (0, 0), (8, 0), (9, 4), (9, 12),
        (7, 16), (4, 17), (1, 16),
        (0, 12), (-0.5, 4),
    ],
    grain_line=GrainLine(start=(4, 1), end=(4, 15), angle_degrees=0.0),
    seam_allowance=SeamAllowance(global_inches=0.625),
)

# 2. Encode to geometric tokens
encoder = PatternEncoder()
encoder.encode(piece)
print(f"Encoded {len(piece.encoded_tokens)} tokens: {piece.encoded_tokens[:3]}...")

# 3. Scale to target measurements
scaler = PatternScaler.for_zero_muscular()
result = scaler.scale(piece)
scaled = result.scaled_piece
print(f"Scaled from {piece.size_label or 'base'} → {scaled.size_label}")

# 4. Export SVG
writer = SVGWriter()
svg = writer.to_string(scaled)
print(f"SVG: {len(svg)} characters")

# Save to file
import os
os.makedirs("output", exist_ok=True)
path = writer.save(scaled, "output/quick_start.svg")
print(f"Saved to {path}")
