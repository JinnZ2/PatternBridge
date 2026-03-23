"""
Example: Sundress pattern — bodice front + back with boundary generation.

Demonstrates using sample pattern data with the boundary generation
pipeline, then encoding, scaling, and exporting to all formats.
"""

from pattern_geometry.piece import PatternPiece
from pattern_geometry.boundary import generate_boundary
from pattern_geometry.encoder import PatternEncoder
from pattern_geometry.scaler import PatternScaler
from pattern_output.svg_writer import SVGWriter
from pattern_output.pdf_writer import PDFWriter
from pattern_output.data_export import save_pattern_set
from patterns import get_sample


def main():
    print("=== PatternBridge: Sundress Example ===\n")

    # 1. Load sample data (simulating vision AI output)
    front_data = get_sample("sundress_front")
    back_data = get_sample("sundress_back")

    # 2. Create pieces from vision results
    front = PatternPiece.from_vision_result(front_data["vision_result"])
    back = PatternPiece.from_vision_result(back_data["vision_result"])
    pieces = [front, back]
    print(f"Created {len(pieces)} pieces from vision data")

    # 3. Generate boundary points from garment templates + measurements
    for piece, data in zip(pieces, [front_data, back_data]):
        generate_boundary(piece, data["measurements"])
        print(f"  {piece.name}: {len(piece.boundary_points)} boundary points, "
              f"fold={'yes' if piece.fold_line else 'no'}")

    # 4. Encode
    encoder = PatternEncoder()
    for piece in pieces:
        encoder.encode(piece)
        summary = encoder.token_summary(piece)
        print(f"  {piece.name}: {summary['total_tokens']} tokens")

    # 5. Scale to size 0 muscular
    scaler = PatternScaler.for_zero_muscular()
    results = scaler.scale_batch(pieces)
    scaled = [r.scaled_piece for r in results]
    print(f"\nScaled {len(scaled)} pieces to zero_muscular profile")

    # 6. Export SVG
    svg_writer = SVGWriter()
    svg_path = svg_writer.save_sheet(scaled, "output/sundress_demo.svg")
    print(f"\nSVG sheet: {svg_path}")

    # 7. Export PDF
    pdf_writer = PDFWriter()
    pdf_path = pdf_writer.save_all(scaled, "output/sundress_demo.pdf")
    print(f"PDF: {pdf_path}")

    # 8. Export JSON (with manifest)
    written = save_pattern_set(
        scaled, "output/sundress", pattern_name="Sundress Demo"
    )
    for name, path in written.items():
        print(f"  {name}: {path}")

    print("\nDone!")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    main()
