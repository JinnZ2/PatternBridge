"""
Data export: JSON and Python dict export for pattern pieces.

Provides structured export formats beyond SVG/PDF for integration
with external tools, databases, and web applications.

Formats:
    - JSON: Human-readable, suitable for APIs and web frontends
    - Dict: Python native, suitable for database insertion or further processing
    - Manifest: Summary of all pieces in a pattern set
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pattern_geometry.piece import PatternPiece


def piece_to_dict(piece: PatternPiece, include_tokens: bool = True) -> dict:
    """
    Export a PatternPiece as a plain dict with all features.

    Extends piece.to_dict() with computed properties and
    structured sub-objects.

    Args:
        piece: PatternPiece to export.
        include_tokens: Whether to include encoded_tokens (can be large).

    Returns:
        Flat dict with all piece data.
    """
    d = piece.to_dict()

    # Add computed properties
    d["computed"] = {
        "is_encodable": piece.is_encodable,
        "is_cut_on_fold": piece.is_cut_on_fold,
        "needs_better_image": piece.needs_better_image,
        "has_darts": piece.has_darts,
        "boundary_point_count": len(piece.boundary_points),
        "token_count": len(piece.encoded_tokens),
    }

    if not include_tokens:
        d.pop("encoded_tokens", None)

    return d


def piece_to_json(
    piece: PatternPiece,
    include_tokens: bool = True,
    indent: int = 2,
) -> str:
    """
    Export a PatternPiece as formatted JSON.

    Args:
        piece: PatternPiece to export.
        include_tokens: Whether to include encoded tokens.
        indent: JSON indentation level.

    Returns:
        JSON string.
    """
    d = piece_to_dict(piece, include_tokens=include_tokens)
    return json.dumps(d, indent=indent, default=str)


def save_json(
    piece: PatternPiece,
    path: str | Path,
    include_tokens: bool = True,
) -> Path:
    """
    Save a PatternPiece as a JSON file.

    Args:
        piece: PatternPiece to save.
        path: Output file path.
        include_tokens: Whether to include encoded tokens.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(piece_to_json(piece, include_tokens=include_tokens))
    return path


def build_manifest(
    pieces: list[PatternPiece],
    pattern_name: str = "Unnamed Pattern",
    notes: str = "",
) -> dict:
    """
    Build a manifest summarizing a complete pattern set.

    Args:
        pieces: All pieces in the pattern.
        pattern_name: Display name for the pattern.
        notes: Optional free-text notes.

    Returns:
        Manifest dict with piece summaries, counts, and metadata.
    """
    piece_summaries = []
    for p in pieces:
        piece_summaries.append({
            "name": p.name,
            "piece_number": p.piece_number,
            "cut_quantity": p.cut_quantity,
            "garment_type": p.garment_type,
            "is_cut_on_fold": p.is_cut_on_fold,
            "boundary_points": len(p.boundary_points),
            "tokens": len(p.encoded_tokens),
            "vision_score": p.total_vision_score,
        })

    return {
        "pattern_name": pattern_name,
        "piece_count": len(pieces),
        "total_cut_count": sum(p.cut_quantity for p in pieces),
        "garment_types": sorted(set(p.garment_type for p in pieces if p.garment_type)),
        "pieces": piece_summaries,
        "notes": notes,
    }


def save_manifest(
    pieces: list[PatternPiece],
    path: str | Path,
    pattern_name: str = "Unnamed Pattern",
    notes: str = "",
) -> Path:
    """
    Save a pattern manifest as JSON.

    Args:
        pieces: All pieces in the pattern.
        path: Output file path.
        pattern_name: Display name.
        notes: Optional notes.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(pieces, pattern_name=pattern_name, notes=notes)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def save_pattern_set(
    pieces: list[PatternPiece],
    output_dir: str | Path,
    pattern_name: str = "Unnamed Pattern",
    include_tokens: bool = True,
) -> dict[str, Path]:
    """
    Export a complete pattern set: individual JSON files + manifest.

    Args:
        pieces: All pieces to export.
        output_dir: Directory for output files.
        pattern_name: Display name for the manifest.
        include_tokens: Whether to include tokens in piece files.

    Returns:
        Dict mapping filenames to written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # Individual piece files
    for piece in pieces:
        filename = f"{piece.name.lower().replace(' ', '_')}.json"
        path = save_json(piece, output_dir / filename, include_tokens=include_tokens)
        written[filename] = path

    # Manifest
    manifest_path = save_manifest(
        pieces, output_dir / "manifest.json", pattern_name=pattern_name
    )
    written["manifest.json"] = manifest_path

    return written
