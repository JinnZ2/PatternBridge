Phase 1 — Foundation (the skeleton)
New repo structure with three core modules mirroring your existing ecosystem:
	∙	pattern_vision/ — adapted from hands-lie-detector. Prompt evaluator gets a new pattern analysis rubric. Feeds pattern images to Claude/GPT-4o and gets back structured JSON: piece name, grain line direction, notch positions, fold lines, seam allowances, key measurements.
	∙	pattern_geometry/ — pulls in GeometricEncoder, OctahedralState, SpatialGrid, SymmetryDetector directly. These encode the extracted coordinates into your binary token format.
	∙	pattern_output/ — SVG writer, PDF tiler, Python data structure. Starts simple.

Phase 2 — The encoding pipeline
Connect vision output → geometric encoding. A pattern piece becomes a sequence of geometric tokens representing its boundary points. Straight edges get sparse tokens, curves get dense ones (SpatialGrid handles this automatically). SymmetryDetector flags fold lines and mirrors automatically.

Phase 3 — Parametric scaling
This is the real magic. Once a pattern is encoded, scaling it to new measurements becomes math on the tokens rather than manual redrawing. Your husband’s 36/36 inseam versus a size 2 are just different parameter inputs to the same token structure.

Phase 4 — Multi-format output
SVG first (easiest), then PDF with tiling for home printers, then the full Python data structure for programmatic use.

PatternBridge
A parametric sewing pattern generation system — takes images of real patterns, extracts geometry using vision AI, encodes shapes using the Geometric-to-Binary framework, and outputs scalable patterns in SVG, PDF, and Python data structures.
Part of the JinnZ2 ecosystem. Bridges Geometric-to-Binary-Computational-Bridge and hands-lie-detector into a new domain.

Architecture

Pattern Image
  → pattern_vision/      (adapted from hands-lie-detector)
    → pattern_geometry/  (adapted from Geometric-to-Binary)
      → pattern_output/  (SVG / PDF / Python data)


Repository Structure

PatternBridge/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── setup.py
│
├── pattern_vision/              # Image analysis layer
│   ├── __init__.py
│   ├── rubric.py                # Pattern feature rubric (replaces hand scoring rubric)
│   ├── prompt_evaluator.py      # LLM vision analysis (adapted from hands-lie-detector)
│   ├── classifier.py            # CNN multi-head classifier (adapted)
│   ├── dataset.py               # Pattern image dataset loader
│   └── train.py                 # Training loop
│
├── pattern_geometry/            # Geometric encoding layer
│   ├── __init__.py
│   ├── piece.py                 # PatternPiece: core data structure
│   ├── encoder.py               # Wraps GeometricEncoder for pattern use
│   ├── symmetry.py              # Wraps SymmetryDetector for fold lines
│   ├── grid.py                  # Wraps SpatialGrid for adaptive resolution
│   └── scaler.py                # Parametric scaling between sizes
│
├── pattern_output/              # Output layer
│   ├── __init__.py
│   ├── svg_writer.py            # SVG pattern output
│   ├── pdf_writer.py            # PDF with tiling for home printers
│   └── data_export.py           # Python dict / JSON export
│
├── bridge/                      # Pipeline orchestrator
│   ├── __init__.py
│   └── pattern_bridge.py        # End-to-end: image → output
│
├── tests/
│   ├── test_vision.py
│   ├── test_geometry.py
│   └── test_output.py
│
├── examples/
│   ├── pants_pullOn.py          # McCall's pull-on pants
│   ├── sundress.py              # S-5474 sleeveless sundress
│   ├── socks.py                 # Hand-drawn fleece sock pattern
│   └── hat.py                   # McCall's M8171 hat
│
└── patterns/                    # Raw pattern images for testing
    └── (your uploaded images go here)


Phase 1 — Core Data Structure
File: pattern_geometry/piece.py
Define PatternPiece — the central object everything else feeds into and outputs from:

@dataclass
class PatternPiece:
    name: str                        # e.g. "FRONT", "SOLE"
    boundary_points: list[tuple]     # ordered (x, y) coordinates
    grain_line: tuple | None         # start/end points of grain line
    fold_line: tuple | None          # fold line if "cut on fold"
    notches: list[tuple]             # notch positions
    darts: list[dict]                # dart definitions
    seam_allowance: float            # in inches or cm
    encoded_tokens: list[str]        # geometric tokens from encoder
    metadata: dict                   # size, garment type, source pattern


Phase 2 — Vision Layer
File: pattern_vision/rubric.py
Replace the hand scoring rubric with a pattern analysis rubric. Seven feature categories mirroring the hands-lie-detector structure:

File: pattern_vision/prompt_evaluator.py
Adapted directly from hands-lie-detector/prompt/evaluator.py. New system prompt targets pattern features instead of hand wear. Returns structured JSON per piece detected in image. Supports Claude and GPT-4o.
Immediate test: Run the prompt evaluator against your sock pattern and pants pattern images. This gives us real data before writing any other code.

Phase 3 — Geometry Layer
File: pattern_geometry/encoder.py
Wraps GeometricEncoder from Geometric-to-Binary. Converts boundary point lists into geometric token sequences. Straight edges get sparse tokens, curves get dense tokens (SpatialGrid controls resolution automatically).
File: pattern_geometry/symmetry.py
Wraps SymmetryDetector. Automatically flags fold lines and mirror symmetry. “Cut on fold” pieces only need half their boundary encoded — the symmetry detector confirms and handles the mirror.
File: pattern_geometry/scaler.py
The parametric scaling engine. Takes a PatternPiece encoded at one size, applies grading rules to the tokens, outputs a new PatternPiece at target measurements. Key insight: scaling is math on the geometric tokens, not manual redrawing.
Grading rules we’ll build for your specific bodies:
	∙	Your measurements: size 0 frame + muscle definition adjustments
	∙	Your husband: 36/36 inseam, tall torso adjustments

Phase 4 — Output Layer
SVG first (simplest), then PDF with tiling, then JSON export.
SVG: Real-world scale. 1 inch = 96px or user-configurable. Grain lines, fold lines, notches all rendered as proper SVG elements with semantic layers.
PDF: Tiled for home printing. A0 pattern on letter/A4 pages with registration marks and overlap zones so you can tape pages together.
JSON/Python: Full PatternPiece data structure serialized. Importable by other tools in the ecosystem.

Phase 5 — Pipeline Orchestrator
File: bridge/pattern_bridge.py
Single entry point for the full pipeline:

bridge = PatternBridge(provider="anthropic")
pieces = bridge.from_image("patterns/pants_front.jpg")
bridge.scale(pieces, measurements={"waist": 24, "inseam": 28, "hip": 34})
bridge.export(pieces, formats=["svg", "pdf", "json"], output_dir="output/")


Dependencies

# Vision
torch
torchvision
Pillow
anthropic
openai  # optional

# Geometry
numpy
scipy
shapely

# Output
svgwrite
reportlab  # PDF
ezdxf      # DXF export (future)


First Steps — In Order
	1.	Create repo PatternBridge on GitHub
	2.	Copy GeometricEncoder, OctahedralState, SpatialGrid, SymmetryDetector from Geometric-to-Binary into pattern_geometry/
	3.	Copy classifier.py, dataset.py, train.py, prompt/evaluator.py from hands-lie-detector into pattern_vision/
	4.	Write pattern_vision/rubric.py — the new pattern analysis rubric
	5.	Adapt prompt_evaluator.py to use pattern rubric
	6.	Test against sock and pants pattern images — validate the vision layer works
	7.	Write pattern_geometry/piece.py — the PatternPiece dataclass
	8.	Wire vision output → PatternPiece
	9.	Write scaler with your measurements hardcoded as first test case
	10.	Write SVG output
	11.	Write PDF output
	12.	Write orchestrator

Ecosystem Link
Add to .fieldlink.json in both source repos:

{
  "PatternBridge": {
    "role": "Sewing pattern digitization and parametric generation",
    "draws_from": ["Geometric-to-Binary-Computational-Bridge", "hands-lie-detector"],
    "sync_strategy": "deep-merge",
    "integrity": "SHA256"
  }
}

