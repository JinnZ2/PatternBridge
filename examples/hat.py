"""
Example: Hat crown pattern — multi-cut piece with encoding.

Demonstrates a piece with cut_quantity=6 (six-panel cap).
Hat crowns use a dart-shaped panel template.
"""

from pattern_geometry.piece import PatternPiece
from pattern_geometry.boundary import generate_boundary
from pattern_geometry.encoder import PatternEncoder
from pattern_output.svg_writer import SVGWriter
from pattern_output.data_export import save_json, build_manifest
from patterns import get_sample


def main():
    print("=== PatternBridge: Hat Crown Example ===\n")

    # 1. Load sample
    hat_data = get_sample("hat_crown")

    # 2. Create piece
    crown = PatternPiece.from_vision_result(hat_data["vision_result"])
    print(f"Piece: {crown.name} ({crown.garment_type})")
    print(f"  Cut quantity: {crown.cut_quantity} panels")

    # 3. Generate boundary from hat template
    generate_boundary(crown, hat_data["measurements"])
    print(f"  Boundary: {len(crown.boundary_points)} points")

    # 4. Encode
    encoder = PatternEncoder()
    encoder.encode(crown)
    summary = encoder.token_summary(crown)
    print(f"\nEncoding:")
    print(f"  Tokens: {summary['total_tokens']}")
    print(f"  State distribution: {summary['state_distribution']}")

    # 5. Show token details
    print(f"\nToken sequence:")
    for i, token in enumerate(crown.encoded_tokens):
        vertex, op, sym = encoder.encoder.get_components(token)
        op_name = "radial" if op == "|" else "tangential"
        print(f"  [{i}] {token}  (vertex={vertex}, {op_name}, {sym})")

    # 6. Export
    svg_writer = SVGWriter()
    svg_path = svg_writer.save(crown, "output/hat_crown.svg")
    print(f"\nSVG: {svg_path}")

    json_path = save_json(crown, "output/hat_crown.json")
    print(f"JSON: {json_path}")

    # 7. Build manifest showing all 6 panels cut from this piece
    manifest = build_manifest(
        [crown], pattern_name="Six-Panel Baseball Cap"
    )
    print(f"\nManifest: {manifest['piece_count']} piece(s), "
          f"{manifest['total_cut_count']} total cuts")

    print("\nDone!")


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    main()
