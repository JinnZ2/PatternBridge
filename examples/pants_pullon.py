"""
Example: Pull-on pants pattern — encode, scale, and export.

Demonstrates the full pipeline using synthetic boundary data
(simulating what vision analysis would produce from a real image).
"""

from pattern_geometry.piece import (
    PatternPiece,
    GrainLine,
    Notch,
    Dart,
    SeamAllowance,
    LengthenShortenLine,
)
from pattern_geometry.encoder import PatternEncoder
from pattern_geometry.scaler import PatternScaler, PANTS_GRADE_RULES
from pattern_output.svg_writer import SVGWriter
from pattern_output.pdf_writer import PDFWriter


def create_pants_front() -> PatternPiece:
    """Create a synthetic pants front piece with typical features."""
    return PatternPiece(
        name="FRONT",
        piece_number=1,
        cut_quantity=2,
        garment_type="pants",
        pattern_brand="McCall",
        pattern_id="M-DEMO",
        size_label="Size 2",
        boundary_points=[
            # Waistband top
            (0.0, 0.0), (5.5, 0.0), (11.0, 0.0),
            # Side seam
            (11.5, 5.0), (12.0, 10.0), (11.8, 15.0),
            (11.5, 20.0), (11.0, 25.0), (10.5, 28.0),
            # Hem
            (10.0, 30.0), (5.5, 30.5), (1.0, 30.0),
            # Inseam
            (0.5, 28.0), (0.0, 25.0), (-0.5, 20.0),
            (-0.8, 15.0), (-0.5, 10.0), (-0.2, 5.0),
        ],
        grain_line=GrainLine(
            start=(5.5, 2.0), end=(5.5, 28.0), angle_degrees=0.0
        ),
        notches=[
            Notch(position=(0.0, 10.0), boundary_index=16, notch_type="single"),
            Notch(position=(12.0, 10.0), boundary_index=4, notch_type="single"),
            Notch(position=(5.5, 0.0), boundary_index=1, notch_type="single"),
        ],
        darts=[
            Dart(
                apex=(4.0, 3.5),
                leg_start=(3.0, 0.0),
                leg_end=(5.0, 0.0),
                intake_inches=1.0,
                depth_inches=3.5,
            ),
        ],
        seam_allowance=SeamAllowance(
            global_inches=0.625,
            edge_overrides={"waist": 1.0, "hem": 1.5},
        ),
        lengthen_shorten_lines=[
            LengthenShortenLine(
                position=(5.5, 18.0), y_coordinate=18.0,
                label="LENGTHEN OR SHORTEN HERE",
            ),
        ],
        total_vision_score=88.0,
        band_label="Good",
    )


def create_pants_back() -> PatternPiece:
    """Create a synthetic pants back piece."""
    return PatternPiece(
        name="BACK",
        piece_number=2,
        cut_quantity=2,
        garment_type="pants",
        pattern_brand="McCall",
        pattern_id="M-DEMO",
        size_label="Size 2",
        boundary_points=[
            (0.0, 0.0), (6.0, 0.0), (12.0, 0.0),
            (12.5, 5.0), (13.0, 10.0), (12.8, 15.0),
            (12.5, 20.0), (12.0, 25.0), (11.5, 28.0),
            (11.0, 30.0), (6.0, 30.5), (1.0, 30.0),
            (0.5, 28.0), (0.0, 25.0), (-0.5, 20.0),
            (-1.0, 15.0), (-0.8, 10.0), (-0.3, 5.0),
        ],
        grain_line=GrainLine(
            start=(6.0, 2.0), end=(6.0, 28.0), angle_degrees=0.0
        ),
        notches=[
            Notch(position=(0.0, 10.0), boundary_index=16, notch_type="single"),
            Notch(position=(13.0, 10.0), boundary_index=4, notch_type="single"),
        ],
        darts=[
            Dart(
                apex=(6.0, 4.0),
                leg_start=(4.5, 0.0),
                leg_end=(7.5, 0.0),
                intake_inches=1.5,
                depth_inches=4.0,
            ),
        ],
        seam_allowance=SeamAllowance(
            global_inches=0.625,
            edge_overrides={"waist": 1.0, "hem": 1.5},
        ),
        total_vision_score=82.0,
        band_label="Good",
    )


def main():
    print("=== PatternBridge: Pull-on Pants Example ===\n")

    # Create pieces
    front = create_pants_front()
    back = create_pants_back()
    pieces = [front, back]
    print(f"Created {len(pieces)} pieces: {[p.name for p in pieces]}")

    # Encode
    encoder = PatternEncoder()
    for piece in pieces:
        encoder.encode(piece)
        summary = encoder.token_summary(piece)
        print(f"  {piece.name}: {summary['total_tokens']} tokens, "
              f"symmetry={summary['symmetry_applied']}")

    # Scale to "size 0 muscular" profile
    scaler = PatternScaler.for_zero_muscular()
    print(f"\nScaling to zero_muscular profile...")
    results = scaler.scale_batch(pieces)
    for result in results:
        print(f"  {result.scaled_piece.name}: {result.summary()[:80]}...")
        if result.warnings:
            for w in result.warnings:
                print(f"    WARNING: {w}")

    scaled_pieces = [r.scaled_piece for r in results]

    # Export SVG
    svg_writer = SVGWriter()
    svg_path = svg_writer.save_sheet(scaled_pieces, "output/pants_demo.svg")
    print(f"\nSVG sheet: {svg_path}")

    # Export individual SVGs
    for piece in scaled_pieces:
        path = svg_writer.save(piece, f"output/pants_{piece.name.lower()}.svg")
        print(f"  {piece.name}: {path}")

    # Export PDF
    pdf_writer = PDFWriter()
    pdf_path = pdf_writer.save_all(scaled_pieces, "output/pants_demo.pdf")
    print(f"\nPDF: {pdf_path}")

    # Export JSON
    for piece in scaled_pieces:
        j = piece.to_json()
        json_path = f"output/pants_{piece.name.lower()}.json"
        with open(json_path, "w") as f:
            f.write(j)
        print(f"  JSON: {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    main()
