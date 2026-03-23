"""
Example: Sock pattern — simple single-piece with encoding and export.

Demonstrates a minimal garment type with the boundary template system.
Socks use an oval-based sole template with toe/heel shaping.
"""

from pattern_geometry.piece import PatternPiece
from pattern_geometry.boundary import generate_boundary
from pattern_geometry.encoder import PatternEncoder
from pattern_output.svg_writer import SVGWriter
from pattern_output.data_export import piece_to_json, save_json
from patterns import get_sample


def main():
    print("=== PatternBridge: Sock Pattern Example ===\n")

    # 1. Load sample
    sock_data = get_sample("sock_sole")

    # 2. Create piece from vision result
    sole = PatternPiece.from_vision_result(sock_data["vision_result"])
    print(f"Piece: {sole.name} ({sole.garment_type})")
    print(f"  Cut quantity: {sole.cut_quantity}")

    # 3. Generate boundary
    generate_boundary(sole, sock_data["measurements"])
    print(f"  Boundary: {len(sole.boundary_points)} points")
    print(f"  Grain line: {sole.grain_line is not None}")

    # 4. Encode
    encoder = PatternEncoder(detect_symmetry=True)
    encoder.encode(sole)
    summary = encoder.token_summary(sole)
    print(f"\nEncoding:")
    print(f"  Tokens: {summary['total_tokens']}")
    print(f"  Symmetry detected: {summary['symmetry_applied']}")
    print(f"  Operators: {summary['operators']}")
    print(f"  Symbols: {summary['symbols']}")

    # 5. Export SVG
    svg_writer = SVGWriter()
    svg_path = svg_writer.save(sole, "output/sock_sole.svg")
    print(f"\nSVG: {svg_path}")

    # 6. Export JSON
    json_path = save_json(sole, "output/sock_sole.json")
    print(f"JSON: {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    main()
