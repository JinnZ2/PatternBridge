"""
PatternBridge: End-to-end pipeline orchestrator.

Single entry point for the full pipeline:
    image → vision analysis → PatternPiece → geometry encoding
         → parametric scaling → multi-format output

Usage:
    bridge = PatternBridge(provider="anthropic")

    # From image
    pieces = bridge.from_image("patterns/pants_front.jpg")

    # Scale to measurements
    scaled = bridge.scale(pieces, profile="zero_muscular")

    # Export
    bridge.export(scaled, formats=["svg", "pdf", "json"], output_dir="output/")

    # Or all at once
    bridge.run(
        image_path="patterns/pants_front.jpg",
        profile="zero_muscular",
        output_dir="output/",
        formats=["svg", "pdf", "json"],
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pattern_vision.rubic import PatternRubric
from pattern_vision.prompt_evaluator import PatternPromptEvaluator
from pattern_geometry.piece import PatternPiece
from pattern_geometry.encoder import PatternEncoder
from pattern_geometry.scaler import (
    PatternScaler,
    PROFILE_ZERO_MUSCULAR,
    PROFILE_TALL_36_36,
    STANDARD_SIZE_0,
    STANDARD_SIZE_36_36,
)
from pattern_output.svg_writer import SVGWriter
from pattern_output.pdf_writer import PDFWriter
from pattern_output.data_export import save_json, save_manifest


# ── Built-in measurement profiles ────────────────────────────────────────────

PROFILES = {
    "zero_muscular": {
        "label": "Size 0 muscular",
        "source": STANDARD_SIZE_0,
        "target": PROFILE_ZERO_MUSCULAR,
        "factory": PatternScaler.for_zero_muscular,
    },
    "tall_36_36": {
        "label": "Tall 36/36",
        "source": STANDARD_SIZE_36_36,
        "target": PROFILE_TALL_36_36,
        "factory": PatternScaler.for_tall_36_36,
    },
}


# ── PipelineResult ────────────────────────────────────────────────────────────

class PipelineResult:
    """
    Full result of a PatternBridge pipeline run.
    Contains pieces at each stage and all output file paths.
    """

    def __init__(self):
        self.source_image: str = ""
        self.raw_pieces: list[PatternPiece] = []        # from vision
        self.encoded_pieces: list[PatternPiece] = []    # after encoding
        self.scaled_pieces: list[PatternPiece] = []     # after scaling
        self.output_files: dict[str, list[Path]] = {    # by format
            "svg": [],
            "pdf": [],
            "json": [],
        }
        self.warnings: list[str] = []
        self.skipped_pieces: list[str] = []             # pieces below score threshold

    def summary(self) -> str:
        lines = [
            "── PatternBridge Pipeline Result ──────────────────",
            f"Source: {self.source_image}",
            f"Pieces detected:  {len(self.raw_pieces)}",
            f"Pieces encoded:   {len(self.encoded_pieces)}",
            f"Pieces scaled:    {len(self.scaled_pieces)}",
        ]

        if self.skipped_pieces:
            lines.append(f"Skipped (low score): {', '.join(self.skipped_pieces)}")

        lines.append("")
        lines.append("Output files:")
        for fmt, paths in self.output_files.items():
            for p in paths:
                lines.append(f"  [{fmt.upper()}] {p}")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")

        lines.append("────────────────────────────────────────────────")
        return "\n".join(lines)


# ── PatternBridge ─────────────────────────────────────────────────────────────

class PatternBridge:
    """
    End-to-end sewing pattern pipeline.

    Coordinates vision analysis, geometric encoding, parametric scaling,
    and multi-format output.

    Args:
        provider: Vision provider — "anthropic", "openai", or "classifier".
            "classifier" uses the CNN PatternClassifier (requires torch and
            a weights file passed via classifier_weights).
        api_key: API key for LLM providers. If None, reads from environment.
        model: Model name override for LLM providers.
        classifier_weights: Path to .pt weights file when provider="classifier".
        encode: Whether to run geometric encoding. Default True.
        min_vision_score: Pieces below this score are skipped. Default 51.
        px_per_inch: SVG scale factor. Default 96 (SVG standard).
        page_size: PDF page size. "letter" or "a4". Default "letter".
    """

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        classifier_weights: Optional[str] = None,
        encode: bool = True,
        min_vision_score: float = 51.0,
        px_per_inch: float = 96.0,
        page_size: str = "letter",
    ):
        self.provider = provider
        self.encode = encode
        self.min_vision_score = min_vision_score

        # Vision layer
        self.rubric = PatternRubric()
        self.classifier = None

        if provider == "classifier":
            from pattern_vision.classifier import PatternClassifier
            self.classifier = PatternClassifier()
            if classifier_weights:
                self.classifier.load(classifier_weights)
            self.evaluator = None
        else:
            self.evaluator = PatternPromptEvaluator(
                provider=provider,
                api_key=api_key,
                model=model,
                rubric=self.rubric,
            )

        # Geometry layer
        self.encoder = PatternEncoder() if encode else None

        # Output layer
        from pattern_output.pdf_writer import PAGE_LETTER, PAGE_A4
        pdf_page = PAGE_LETTER if page_size == "letter" else PAGE_A4
        self.svg_writer = SVGWriter(px_per_inch=px_per_inch)
        self.pdf_writer = PDFWriter(page_size=pdf_page)

    # ── Main API ──────────────────────────────────────────────────────────────

    def run(
        self,
        image_path: str | Path,
        profile: str | dict = "zero_muscular",
        output_dir: str | Path = "output",
        formats: list[str] | None = None,
    ) -> PipelineResult:
        """
        Full pipeline: image → analysis → encode → scale → export.

        Args:
            image_path: Path to pattern image.
            profile: Measurement profile name ("zero_muscular", "tall_36_36")
                     or custom dict with keys "source", "target", "ease".
            output_dir: Directory for output files.
            formats: List of output formats. Default ["svg", "pdf", "json"].

        Returns:
            PipelineResult with all pieces and output paths.
        """
        formats = formats or ["svg", "pdf", "json"]
        result = PipelineResult()
        result.source_image = str(image_path)

        print(f"\n── PatternBridge ──────────────────────────────────")
        print(f"Image:   {image_path}")
        print(f"Profile: {profile if isinstance(profile, str) else 'custom'}")
        print(f"Output:  {output_dir}")
        print(f"Formats: {', '.join(formats)}")
        print()

        # Stage 1: Vision analysis
        print("Stage 1/4  Vision analysis...")
        pieces = self.from_image(image_path)
        result.raw_pieces = pieces

        if not pieces:
            print("  No pieces detected. Check image quality.")
            return result

        for p in pieces:
            print(f"  ✓ {p.name}  score={p.total_vision_score:.0f}  [{p.band_label}]")

        # Stage 2: Geometric encoding
        if self.encode and self.encoder:
            print("\nStage 2/4  Geometric encoding...")
            encoded = []
            for piece in pieces:
                if piece.needs_better_image:
                    print(f"  ✗ {piece.name}  skipped (score {piece.total_vision_score:.0f} < {self.min_vision_score})")
                    result.skipped_pieces.append(piece.name)
                    continue
                if piece.boundary_points:
                    self.encoder.encode(piece)
                    summary = self.encoder.token_summary(piece)
                    print(f"  ✓ {piece.name}  {summary['total_tokens']} tokens")
                    encoded.append(piece)
                else:
                    print(f"  ✗ {piece.name}  no boundary points yet")
                    result.skipped_pieces.append(piece.name)
            result.encoded_pieces = encoded
        else:
            result.encoded_pieces = [
                p for p in pieces if not p.needs_better_image
            ]

        # Stage 3: Parametric scaling
        print("\nStage 3/4  Scaling to measurements...")
        scaled = self.scale(result.encoded_pieces, profile=profile)
        result.scaled_pieces = scaled

        for s in scaled:
            print(f"  ✓ {s.name}  → {s.size_label}")

        # Stage 4: Export
        print("\nStage 4/4  Exporting...")
        output_files = self.export(
            scaled,
            formats=formats,
            output_dir=output_dir,
            stem=Path(image_path).stem,
        )
        result.output_files = output_files

        for fmt, paths in output_files.items():
            for p in paths:
                print(f"  ✓ [{fmt.upper()}] {p}")

        # Collect warnings from scaling
        scaler = self._build_scaler(profile)
        for piece in scaled:
            for w in self._check_muscle_warnings(piece):
                result.warnings.append(w)

        print()
        print(result.summary())
        return result

    def from_image(self, image_path: str | Path) -> list[PatternPiece]:
        """
        Stage 1: Run vision analysis on a pattern image.
        Returns list of PatternPiece objects with metadata populated.
        Boundary points are empty at this stage.

        Args:
            image_path: Path to pattern image.

        Returns:
            List of PatternPiece objects from detected pieces.
        """
        if self.classifier is not None:
            prediction = self.classifier.predict(image_path)
            vision_result = self.classifier.to_vision_result(prediction)
            pieces = [
                PatternPiece.from_vision_result(
                    vision_result, image_source=str(image_path)
                )
            ]
        else:
            vision_results = self.evaluator.evaluate(image_path)
            pieces = [
                PatternPiece.from_vision_result(r, image_source=str(image_path))
                for r in vision_results
            ]
        return pieces

    def scale(
        self,
        pieces: list[PatternPiece],
        profile: str | dict = "zero_muscular",
    ) -> list[PatternPiece]:
        """
        Stage 3: Scale pieces to target measurements.

        Args:
            pieces: List of PatternPieces (with or without boundary points).
            profile: Profile name or custom dict.

        Returns:
            List of scaled PatternPieces.
        """
        scaler = self._build_scaler(profile)
        scaled = []
        for piece in pieces:
            if piece.boundary_points:
                result = scaler.scale(piece)
                scaled.append(result.scaled_piece)
                if result.warnings:
                    for w in result.warnings:
                        print(f"  ⚠  {piece.name}: {w}")
            else:
                # No boundary yet — just update measurements metadata
                piece.target_measurements = scaler.target
                piece.ease_allowances = scaler.ease
                piece.size_label = scaler._build_size_label()
                scaled.append(piece)
        return scaled

    def export(
        self,
        pieces: list[PatternPiece],
        formats: list[str] | None = None,
        output_dir: str | Path = "output",
        stem: str = "pattern",
    ) -> dict[str, list[Path]]:
        """
        Stage 4: Export pieces to one or more formats.

        Args:
            pieces: List of PatternPieces to export.
            formats: ["svg", "pdf", "json"] or subset.
            output_dir: Output directory.
            stem: Base filename stem.

        Returns:
            Dict mapping format → list of output Paths.
        """
        formats = formats or ["svg", "pdf", "json"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files: dict[str, list[Path]] = {f: [] for f in formats}

        renderable = [p for p in pieces if p.boundary_points]
        metadata_only = [p for p in pieces if not p.boundary_points]

        # SVG — one file per piece
        if "svg" in formats and renderable:
            for piece in renderable:
                svg_path = output_dir / f"{stem}_{piece.name.lower()}.svg"
                self.svg_writer.save(piece, svg_path)
                output_files["svg"].append(svg_path)

            # Also save a combined sheet if multiple pieces
            if len(renderable) > 1:
                sheet_path = output_dir / f"{stem}_all.svg"
                self.svg_writer.save_sheet(renderable, sheet_path)
                output_files["svg"].append(sheet_path)

        # PDF — all pieces in one tiled document
        if "pdf" in formats and renderable:
            pdf_path = output_dir / f"{stem}_print.pdf"
            self.pdf_writer.save_all(renderable, pdf_path)
            output_files["pdf"].append(pdf_path)

        # JSON — individual piece files + manifest
        if "json" in formats:
            for piece in pieces:
                piece_path = output_dir / f"{stem}_{piece.name.lower()}.json"
                save_json(piece, piece_path)
                output_files["json"].append(piece_path)

            # Manifest with summary
            manifest_path = output_dir / f"{stem}_manifest.json"
            save_manifest(
                pieces, manifest_path, pattern_name=stem,
                notes=f"source: {stem}, renderable: {len(renderable)}, metadata_only: {len(metadata_only)}",
            )
            output_files["json"].append(manifest_path)

        return output_files

    # ── Convenience methods ───────────────────────────────────────────────────

    def analyze(self, image_path: str | Path) -> None:
        """
        Quick analysis — print a human-readable summary of what
        the vision layer detects in an image. No encoding or export.
        Useful for testing and debugging.
        """
        print(f"\nAnalyzing: {image_path}")
        print("─" * 50)

        pieces = self.from_image(image_path)

        if not pieces:
            print("No pieces detected.")
            return

        for i, piece in enumerate(pieces, 1):
            print(f"\nPiece {i}: {piece.name}")
            print(f"  Brand:         {piece.pattern_brand}")
            print(f"  Garment type:  {piece.garment_type}")
            print(f"  Cut quantity:  {piece.cut_quantity}")
            print(f"  Cut on fold:   {piece.is_cut_on_fold}")
            print(f"  Seam allow:    {piece.seam_allowance.global_inches if piece.seam_allowance else 'unknown'}\"")
            print(f"  Vision score:  {piece.total_vision_score:.0f}/100  [{piece.band_label}]")
            print(f"  Quality notes: {piece.image_quality_notes or 'none'}")

            if piece.vision_scores:
                print("  Rubric scores:")
                for cat, score in piece.vision_scores.items():
                    bar = "█" * int(score / 2)
                    print(f"    {cat:<30} {score:5.1f}  {bar}")

        print()

    def analyze_batch(self, image_paths: list[str | Path]) -> None:
        """Analyze multiple images sequentially."""
        for path in image_paths:
            self.analyze(path)

    def encode_piece(self, piece: PatternPiece) -> PatternPiece:
        """Encode a single piece. Requires boundary_points to be set."""
        if self.encoder is None:
            raise RuntimeError("Encoder disabled (encode=False at init).")
        return self.encoder.encode(piece)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_scaler(self, profile: str | dict) -> PatternScaler:
        """Build a PatternScaler from a profile name or custom dict."""
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise ValueError(
                    f"Unknown profile '{profile}'. "
                    f"Available: {list(PROFILES.keys())}"
                )
            return PROFILES[profile]["factory"]()

        # Custom profile dict
        source = profile.get("source", STANDARD_SIZE_0)
        target = profile.get("target", {})
        ease   = profile.get("ease", {})
        return PatternScaler(
            source_measurements=source,
            target_measurements=target,
            ease=ease,
        )

    def _check_muscle_warnings(self, piece: PatternPiece) -> list[str]:
        """Check for muscle-specific fit warnings on a scaled piece."""
        warnings = []
        ease = piece.ease_allowances

        if ease.get("thigh", 0) < 1.0 and piece.garment_type == "pants":
            warnings.append(
                f"{piece.name}: Thigh ease < 1\". May be tight for muscular build."
            )
        if ease.get("bicep", 0) < 1.0 and piece.garment_type in ("top", "jacket", "coat"):
            warnings.append(
                f"{piece.name}: Bicep ease < 1\". May restrict arm movement."
            )
        return warnings
