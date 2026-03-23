"""
Sample pattern data for testing and demonstration.

Since real pattern images require copyright clearance, this module provides
synthetic pattern metadata that simulates vision AI output. Each sample
includes the feature dict that PatternPromptEvaluator would produce from
a real image, plus reference measurements for boundary generation.

Usage:
    from patterns import SAMPLES, get_sample

    # Get a specific sample
    pants = get_sample("pants_front")

    # Use in pipeline
    piece = PatternPiece.from_vision_result(pants["vision_result"])
    generate_boundary(piece, pants["measurements"])
"""

from __future__ import annotations


# ── Sample pattern data ──────────────────────────────────────────────────────

SAMPLES: dict[str, dict] = {

    "pants_front": {
        "description": "Women's pull-on pants front piece, size 0",
        "vision_result": {
            "piece_name": "FRONT",
            "piece_number": 1,
            "cut_quantity": 2,
            "garment_type": "pants",
            "pattern_brand": "McCall",
            "seam_allowance_inches": 0.625,
            "fold_line_present": True,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 3,
            "dart_count": 0,
            "is_graded_multi_size": False,
            "total_score": 82.0,
            "band_label": "Good",
        },
        "measurements": {
            "waist": 12.0,
            "hip": 13.0,
            "rise_front": 9.5,
            "inseam": 28.5,
            "hem": 8.0,
        },
    },

    "pants_back": {
        "description": "Women's pull-on pants back piece, size 0",
        "vision_result": {
            "piece_name": "BACK",
            "piece_number": 2,
            "cut_quantity": 2,
            "garment_type": "pants",
            "pattern_brand": "McCall",
            "seam_allowance_inches": 0.625,
            "fold_line_present": False,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 3,
            "dart_count": 0,
            "is_graded_multi_size": False,
            "total_score": 78.0,
            "band_label": "Good",
        },
        "measurements": {
            "waist": 13.0,
            "hip": 14.0,
            "rise": 13.0,
            "inseam": 28.5,
            "hem": 9.0,
        },
    },

    "bodice_front": {
        "description": "Fitted bodice front, cut-on-fold, size 8",
        "vision_result": {
            "piece_name": "FRONT",
            "piece_number": 1,
            "cut_quantity": 1,
            "garment_type": "dress",
            "pattern_brand": "Simplicity",
            "seam_allowance_inches": 0.625,
            "fold_line_present": True,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 4,
            "dart_count": 2,
            "is_graded_multi_size": False,
            "total_score": 88.0,
            "band_label": "Good",
        },
        "measurements": {
            "bust": 34.0,
            "waist": 27.0,
            "shoulder_width": 15.0,
            "torso_front": 16.0,
        },
    },

    "bodice_back": {
        "description": "Fitted bodice back, center seam, size 8",
        "vision_result": {
            "piece_name": "BACK",
            "piece_number": 2,
            "cut_quantity": 2,
            "garment_type": "dress",
            "pattern_brand": "Simplicity",
            "seam_allowance_inches": 0.625,
            "fold_line_present": False,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 3,
            "dart_count": 1,
            "is_graded_multi_size": False,
            "total_score": 85.0,
            "band_label": "Good",
        },
        "measurements": {
            "bust": 36.0,
            "waist": 28.0,
            "shoulder_width": 15.5,
            "torso_back": 17.0,
        },
    },

    "skirt_front": {
        "description": "A-line skirt front, cut-on-fold, knee length",
        "vision_result": {
            "piece_name": "FRONT",
            "piece_number": 1,
            "cut_quantity": 1,
            "garment_type": "skirt",
            "pattern_brand": "Butterick",
            "seam_allowance_inches": 0.625,
            "fold_line_present": True,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 2,
            "dart_count": 2,
            "is_graded_multi_size": True,
            "total_score": 90.0,
            "band_label": "Excellent",
        },
        "measurements": {
            "waist": 13.0,
            "hip": 17.0,
            "length": 22.0,
        },
    },

    "sock_sole": {
        "description": "Basic sock pattern, one-size template",
        "vision_result": {
            "piece_name": "SOLE",
            "piece_number": 1,
            "cut_quantity": 2,
            "garment_type": "sock",
            "pattern_brand": "handdrawn",
            "seam_allowance_inches": 0.25,
            "fold_line_present": False,
            "grain_line_angle_degrees": None,
            "notch_count": 2,
            "dart_count": 0,
            "is_graded_multi_size": False,
            "total_score": 65.0,
            "band_label": "Adequate",
        },
        "measurements": {
            "foot_length": 10.0,
            "foot_width": 4.0,
            "cuff_height": 6.0,
        },
    },

    "hat_crown": {
        "description": "Six-panel baseball cap crown section",
        "vision_result": {
            "piece_name": "CROWN",
            "piece_number": 1,
            "cut_quantity": 6,
            "garment_type": "hat",
            "pattern_brand": "handdrawn",
            "seam_allowance_inches": 0.375,
            "fold_line_present": False,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 2,
            "dart_count": 0,
            "is_graded_multi_size": False,
            "total_score": 70.0,
            "band_label": "Adequate",
        },
        "measurements": {
            "circumference": 22.0,
            "depth": 7.0,
        },
    },

    "sundress_front": {
        "description": "Sundress bodice front with spaghetti straps",
        "vision_result": {
            "piece_name": "FRONT",
            "piece_number": 1,
            "cut_quantity": 1,
            "garment_type": "dress",
            "pattern_brand": "Vogue",
            "seam_allowance_inches": 0.625,
            "fold_line_present": True,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 4,
            "dart_count": 2,
            "is_graded_multi_size": False,
            "total_score": 92.0,
            "band_label": "Excellent",
        },
        "measurements": {
            "bust": 32.0,
            "waist": 25.0,
            "shoulder_width": 14.0,
            "torso_front": 15.0,
        },
    },

    "sundress_back": {
        "description": "Sundress bodice back with zipper",
        "vision_result": {
            "piece_name": "BACK",
            "piece_number": 2,
            "cut_quantity": 2,
            "garment_type": "dress",
            "pattern_brand": "Vogue",
            "seam_allowance_inches": 0.625,
            "fold_line_present": False,
            "grain_line_angle_degrees": 0.0,
            "notch_count": 3,
            "dart_count": 0,
            "is_graded_multi_size": False,
            "total_score": 88.0,
            "band_label": "Good",
        },
        "measurements": {
            "bust": 34.0,
            "waist": 26.0,
            "shoulder_width": 14.5,
            "torso_back": 16.0,
        },
    },
}


def get_sample(name: str) -> dict:
    """
    Get a sample pattern by name.

    Args:
        name: Sample key (e.g., "pants_front", "bodice_front", "hat_crown").

    Returns:
        Dict with 'description', 'vision_result', and 'measurements'.

    Raises:
        KeyError: If sample name not found.
    """
    if name not in SAMPLES:
        available = ", ".join(sorted(SAMPLES.keys()))
        raise KeyError(f"Unknown sample '{name}'. Available: {available}")
    return SAMPLES[name]


def list_samples() -> list[str]:
    """Return list of available sample names."""
    return sorted(SAMPLES.keys())
