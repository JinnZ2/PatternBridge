# PatternBridge

A parametric sewing pattern generation system that takes images of real patterns, extracts geometry using vision AI, encodes shapes using the Geometric-to-Binary framework, and outputs scalable patterns in SVG, PDF, and JSON.

Part of the **JinnZ2 ecosystem**. Bridges [Geometric-to-Binary-Computational-Bridge](https://github.com/JinnZ2/Geometric-to-Binary-Computational-Bridge) and [hands-lie-detector](https://github.com/JinnZ2/hands-lie-detector) into the sewing pattern domain.

---

## Architecture

```
Pattern Image
  → pattern_vision/      Vision AI extracts piece features (rubric + LLM)
    → pattern_geometry/  Encodes boundary → geometric tokens, scales parametrically
      → pattern_output/  Renders SVG, tiled PDF, or JSON
```

Orchestrated by `bridge/pattern_bridge.py` — single entry point for the full pipeline.

---

## Repository Structure (Actual)

```
PatternBridge/
├── CLAUDE.md                       # This file — project guide for AI assistants
├── README.md                       # Brief project description
├── LICENSE                         # Apache 2.0
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Package configuration (pip installable)
├── .gitignore                      # Git ignore rules
│
├── pattern_vision/                 # Image analysis layer
│   ├── __init__.py
│   ├── rubic.py                    # 7-category pattern scoring rubric (100 pts)
│   ├── prompt_evaluator.py         # Vision LLM analysis (Anthropic / OpenAI)
│   ├── classifier.py               # CNN multi-head classifier (ResNet-18 backbone)
│   ├── dataset.py                  # PyTorch Dataset for labeled pattern images
│   └── train.py                    # Training loop with multi-task loss + CLI
│
├── pattern_geometry/               # Geometric encoding layer
│   ├── __init__.py
│   ├── piece.py                    # PatternPiece dataclass — core data structure
│   ├── boundary.py                 # Template-based boundary generation (vision→geometry bridge)
│   ├── encoder.py                  # Boundary → geometric token encoding
│   ├── scaler.py                   # Parametric scaling with grading rules
│   ├── geometric_encoder.py        # Bidirectional token↔binary encoding
│   ├── octahedral_state.py         # 8-vertex cubic coordinate states
│   ├── spatial_grid.py             # Octree-based adaptive spatial decomposition
│   └── symmetry_detector.py        # Reflective and rotational symmetry detection
│
├── pattern_output/                 # Output layer
│   ├── __init__.py
│   ├── svg_writer.py               # SVG at real-world scale (96 px/inch)
│   ├── pdf_writer.py               # Tiled PDF for home printers
│   └── data_export.py              # JSON/dict export with manifest generation
│
├── bridge/
│   ├── __init__.py
│   └── pattern_bridge.py           # End-to-end orchestrator
│
├── patterns/
│   └── __init__.py                 # Synthetic sample data (8 garment pieces)
│
├── tests/                          # Unit tests (187 tests, all passing)
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures (pieces, vision results)
│   ├── test_vision.py              # Rubric, scoring, prompt evaluator
│   ├── test_geometry.py            # PatternPiece, encoder, scaler
│   ├── test_boundary.py            # Boundary generation, templates, full pipeline
│   ├── test_output.py              # SVG and PDF writers
│   ├── test_pipeline.py            # End-to-end pipeline wiring
│   ├── test_samples.py             # Sample data, classifier/dataset/train imports
│   ├── test_data_export.py         # JSON/data export tests
│   └── test_capture_server.py      # Mobile capture server tests
│
├── tools/
│   ├── __init__.py
│   └── capture_server.py           # Mobile capture server for phone-based data collection
│
├── weights/                        # Trained classifier weights (empty until training)
│
└── examples/                       # Usage examples
    ├── quick_start.py              # Minimal pipeline in ~20 lines
    ├── pants_pullon.py             # Multi-piece pants with encode → scale → export
    ├── sundress.py                 # Full pipeline: sample data → SVG + PDF + JSON
    ├── socks.py                    # Single-piece pipeline with symmetry detection
    └── hat.py                      # Multi-cut piece with token introspection
```

### Known filename issue

`pattern_vision/rubic.py` — note the typo (should be "rubric"). Preserve existing name to avoid breaking imports unless explicitly asked to rename.

---

## Implementation Status

### Fully implemented
- **pattern_vision/rubic.py** (163 lines) — 7-category rubric: Piece Identification, Grain Line, Fold Line, Notch Positions, Dart Definitions, Seam Allowance, Boundary Traceability. 100-point scale with 6 interpretation bands.
- **pattern_vision/prompt_evaluator.py** (297 lines) — `PatternPromptEvaluator` class. Supports `anthropic` (claude-sonnet-4-6) and `openai` (gpt-4o) providers. Encodes images to base64, sends with structured prompt, returns list of piece dicts.
- **pattern_geometry/piece.py** (322 lines) — `PatternPiece` dataclass plus supporting types (`GrainLine`, `FoldLine`, `Notch`, `Dart`, `SeamAllowance`, `LengthenShortenLine`). Factory method `from_vision_result()`. Serialization via `to_dict()` / `to_json()`.
- **pattern_geometry/scaler.py** (721 lines) — `PatternScaler` with zone-based grading. Built-in profiles: `PROFILE_ZERO_MUSCULAR` (size 0 + muscle), `PROFILE_TALL_36_36` (36/36 inseam). Grade rules for pants and bodice. Factory methods `for_zero_muscular()`, `for_tall_36_36()`.
- **pattern_output/svg_writer.py** (752 lines) — `SVGWriter` renders boundary, seam line, grain line, fold line, notches, darts, labels. Multi-piece sheet layout. Requires `svgwrite`.
- **pattern_output/pdf_writer.py** (848 lines) — `PDFWriter` tiles large patterns across Letter/A4 pages with registration marks, overlap zones, assembly instructions, optional cover page. Requires `reportlab`.
- **bridge/pattern_bridge.py** (461 lines) — `PatternBridge` orchestrator with `run()`, `from_image()`, `scale()`, `export()`, `analyze()` methods. `PipelineResult` tracks pieces through all stages.

- **pattern_geometry/encoder.py** (524 lines) — Code is complete and **runs against real implementations** of the four Geometric-to-Binary classes ported from the sibling repo.
- **pattern_geometry/geometric_encoder.py** — Bidirectional token↔binary encoding with round-trip validation. Supports nested operators (`||`), colon alias (`:`).
- **pattern_geometry/octahedral_state.py** — 8-vertex cubic coordinate states with inversion, distance, and dot product operations.
- **pattern_geometry/spatial_grid.py** — Octree-based adaptive spatial decomposition + curvature-adaptive boundary point refinement.
- **pattern_geometry/symmetry_detector.py** — Reflective and rotational symmetry detection using Rodrigues' rotation formula.
- **pattern_vision/classifier.py** — CNN multi-head classifier (ResNet-18 backbone). Predicts garment type, piece name, fold/grain lines, notch/dart counts. Requires `torch`/`torchvision` (import-guarded).
- **pattern_output/data_export.py** — JSON/dict export with computed properties, manifest generation, and full pattern set export.
- **patterns/__init__.py** — Synthetic sample pattern data (8 samples: pants front/back, bodice front/back, skirt front, sock sole, hat crown, sundress front/back) with vision results and measurements.
- **pattern_vision/dataset.py** (230 lines) — `PatternDataset` PyTorch Dataset. Scans garment_type/piece_name directory tree, supports augmentation (RandomResizedCrop, flip, rotation, color jitter), optional per-image JSON annotations for fold/grain/notch/dart labels.
- **pattern_vision/train.py** (299 lines) — `train()` function + `MultiTaskLoss` class. Weighted multi-task loss (CE for classification, BCE for binary, MSE for regression). AdamW + CosineAnnealingLR. Train/val split, best-model checkpointing, CLI entry point via `python -m pattern_vision.train`.
- **tools/capture_server.py** — Flask web server for phone-based pattern image capture. Mobile-friendly UI with garment type/piece name selection, per-image annotation (fold/grain/notch/dart), and live capture history. Saves directly into PatternDataset directory structure.
- **examples/sundress.py** — Full pipeline: sample data → boundary → encode → scale → SVG + PDF + JSON.
- **examples/socks.py** — Single-piece pipeline with symmetry detection.
- **examples/hat.py** — Multi-cut piece with token introspection.

### Not yet created
| Planned file | Purpose |
|---|---|
| `pattern_vision/dataset.py` | ~~Pattern image dataset loader~~ — **Done** |
| `pattern_vision/train.py` | ~~Training loop for classifier~~ — **Done** |
| `weights/` | Trained classifier weights (directory ready, needs training data) |

---

## Key Classes and Entry Points

### PatternPiece (`pattern_geometry/piece.py`)
The central data structure. Everything feeds into it and outputs from it.

```python
@dataclass
class PatternPiece:
    name: str                           # "FRONT", "SOLE", etc.
    boundary_points: list[Point]        # ordered (x, y) coordinates
    grain_line: GrainLine | None
    fold_line: FoldLine | None
    notches: list[Notch]
    darts: list[Dart]
    seam_allowance: SeamAllowance
    encoded_tokens: list[str]           # geometric tokens from encoder
    # ... plus vision metadata, scaling fields, etc.
```

Key properties: `is_cut_on_fold`, `is_encodable` (≥3 boundary points), `needs_better_image` (score < 40).

### PatternBridge (`bridge/pattern_bridge.py`)
Primary user-facing API:

```python
bridge = PatternBridge(provider="anthropic")
result = bridge.run(
    image_path="patterns/pants_front.jpg",
    profile="zero_muscular",        # or "tall_36_36" or custom dict
    formats=["svg", "pdf", "json"],
    output_dir="output/"
)
```

Individual pipeline stages: `from_image()` → `scale()` → `export()`.

### generate_boundary (`pattern_geometry/boundary.py`)
Bridges vision output (no coordinates) to geometry layer (needs boundary points):

```python
from pattern_geometry.boundary import generate_boundary

piece = PatternPiece.from_vision_result(vision_dict)
generate_boundary(piece, measurements={"waist": 12, "hip": 13, "inseam": 28})
# piece.boundary_points, grain_line, fold_line, notches now populated
```

Templates for: pants front/back, bodice front/back, skirt, sock, hat crown. Falls back to rectangle for unknown garment types.

### PatternScaler (`pattern_geometry/scaler.py`)
Built-in measurement profiles:
- `"zero_muscular"` — Size 0 frame + muscle definition (bust 31, waist 24, hip 34, inseam 28)
- `"tall_36_36"` — 36 waist, 36 inseam, tall torso (bust 40, waist 36, hip 40, inseam 36)

---

## Dependencies

### Required (see requirements.txt)

```
# Vision
anthropic          # Claude API for pattern analysis
Pillow             # Image loading and base64 encoding

# Geometry
numpy              # Spatial math, polygon operations
scipy              # Curve fitting, Gaussian smoothing
shapely            # Computational geometry (used in encoder)

# Output
svgwrite           # SVG generation
reportlab          # PDF generation

# Tools
flask              # Mobile capture server

# Optional
openai             # GPT-4o support
torch, torchvision # CNN classifier (training + inference)
ezdxf              # DXF export (future)
```

### External repo dependency
Four classes from **Geometric-to-Binary-Computational-Bridge** are ported into `pattern_geometry/`:
- `GeometricEncoder`
- `OctahedralState`
- `SpatialGrid`
- `SymmetryDetector`

### Python version
Requires **Python 3.10+** (uses `X | Y` union syntax and `from __future__ import annotations`).

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (187 tests)
pytest tests/ -v

# Run by layer
pytest tests/test_vision.py -v     # rubric, scoring, prompt evaluator
pytest tests/test_geometry.py -v   # PatternPiece, encoder, scaler
pytest tests/test_output.py -v     # SVG and PDF writers
pytest tests/test_pipeline.py -v   # end-to-end pipeline wiring
```

## Running Examples

```bash
# From project root
python examples/quick_start.py      # minimal pipeline demo
python examples/pants_pullon.py     # full multi-piece pants example

# Output goes to output/ directory (SVG, PDF, JSON)
```

---

## Development Conventions

### Code style
- Dataclasses for data structures (not plain dicts)
- Type hints throughout (using `tuple[float, float]` style, not `Tuple`)
- Import guards for optional dependencies (`try: import svgwrite except ImportError: ...`)
- Constants defined at module level in UPPER_SNAKE_CASE
- Factory methods as `@classmethod` on dataclasses (e.g., `PatternPiece.from_vision_result()`)

### Units
- Default unit is **inches** throughout the codebase
- SVG: 96 pixels per inch (configurable via `px_per_inch`)
- PDF: 72 points per inch (reportlab default)
- Metric conversion constants in svg_writer.py: `MM_PER_INCH = 25.4`, `CM_PER_INCH = 2.54`

### Coordinate system
- Pattern pieces use (x, y) tuples in real-world units (inches)
- Origin at top-left of bounding box for output
- Grain lines stored as start/end point pairs with angle

### Token encoding format
Geometric tokens use the Geometric-to-Binary framework:
- Operators: `|` (radial/straight), `/` (tangential/curved)
- Symbols: `O` (octahedral point), `I` (notch), `X` (dart apex), `Δ` (grain/fold marker)
- Straight edges get sparse tokens; curves get dense tokens (curvature-adaptive)

---

## Pipeline Stages

1. **Vision** — `PatternPromptEvaluator.evaluate(image_path)` → list of piece dicts with features scored against the 7-category rubric
2. **Structuring** — `PatternPiece.from_vision_result(piece_dict)` → typed dataclass (no boundary points yet)
3. **Boundary** — `generate_boundary(piece, measurements)` → fills `boundary_points`, `grain_line`, `fold_line`, `notches` from garment-type templates
4. **Encoding** — `PatternEncoder.encode(piece)` → fills `encoded_tokens` (uses real Geometric-to-Binary implementations)
5. **Scaling** — `PatternScaler.scale(piece)` → new `PatternPiece` at target measurements
6. **Output** — `SVGWriter.save()` / `PDFWriter.save()` / `piece.to_json()`

---

## Blockers and Next Steps

### Resolved blockers
1. ~~Geometric-to-Binary classes not bundled~~ — **Resolved**: real implementations ported from sibling repo (GeometricEncoder, OctahedralState, SpatialGrid, SymmetryDetector).
2. ~~No `__init__.py` files~~ — **Resolved**: all four packages have `__init__.py`.
3. ~~No `requirements.txt`~~ — **Resolved**: `requirements.txt` created.
4. ~~Cross-package relative imports broke~~ — **Resolved**: converted `from ..package` to absolute `from package` imports.
5. ~~`prompt_evaluator.py` imported `.rubric` instead of `.rubic`~~ — **Resolved**: fixed to match actual filename.
6. ~~`encoder.py` called `_normalize_points()` which raised NotImplementedError~~ — **Resolved**: changed to call `_encode_points_at_indices()`.
7. ~~No boundary extraction between vision and geometry~~ — **Resolved**: `boundary.py` generates template-based boundary points from garment type and measurements.
8. ~~`STANDARD_SIZE_0` and `STANDARD_SIZE_36_36` missing `bicep` and `back_width`~~ — **Resolved**: added missing keys, eliminating 14"+ phantom deltas.
9. ~~`SpatialGrid.refine()` was a no-op~~ — **Resolved**: now does curvature-adaptive midpoint insertion.
10. ~~`svg_writer.py` passed strings to `dwg.polygon(points=...)`~~ — **Resolved**: uses list of tuples per svgwrite API.
11. ~~`scaler.py` `np.array` from int tuples gave int64 dtype~~ — **Resolved**: explicit `dtype=np.float64`.

### Priority next steps
1. ~~Replace Geometric-to-Binary stub classes with real implementations~~ — **Done**
2. Test vision layer against real pattern images
3. ~~Add sample pattern data to `patterns/` directory~~ — **Done** (synthetic samples)
4. ~~Create additional example scripts (sundress, socks, hat)~~ — **Done**
5. ~~Add CNN classifier (`pattern_vision/classifier.py`)~~ — **Done** (needs training data + weights)
6. ~~Add JSON/data export module~~ — **Done** (`pattern_output/data_export.py`)
7. ~~Create `pattern_vision/dataset.py` and `train.py` for classifier training~~ — **Done** (dataset loader + multi-task training loop with CLI)
8. Collect and curate real pattern images for training and testing

---

## Ecosystem Links

PatternBridge draws from two sibling repos:
- **Geometric-to-Binary-Computational-Bridge** — geometric encoding framework (encoder, octahedral states, spatial grid, symmetry detection)
- **hands-lie-detector** — vision AI inference architecture (prompt evaluation, rubric scoring, image analysis pipeline)

Linked via `.fieldlink.json`:
```json
{
  "PatternBridge": {
    "role": "Sewing pattern digitization and parametric generation",
    "draws_from": ["Geometric-to-Binary-Computational-Bridge", "hands-lie-detector"],
    "sync_strategy": "deep-merge",
    "integrity": "SHA256"
  }
}
```

---

## Design Notes

### Parametric scaling approach
Scaling works on geometric tokens rather than manual redrawing. A pattern encoded at one size can be graded to any target measurements by applying grade rules (x/y displacement rates) to landmark points, with zone-based Gaussian smoothing to blend regions naturally. The scaler handles dart adjustment, notch repositioning, and lengthen/shorten line movement automatically.

### PDF tiling strategy
Large patterns are tiled across standard paper sizes (Letter/A4) with configurable overlap zones (default 0.5 inches). Each tile page includes registration marks (crosshairs in corners), row/column labels, and assembly hints. An optional cover page lists all pieces with dimensions and assembly order.

### Vision scoring
The rubric evaluates pattern images on a 100-point scale across 7 categories. Pieces scoring below 40 are flagged via `needs_better_image`. The minimum vision score for pipeline inclusion is configurable in `PatternBridge` (default: 30).
