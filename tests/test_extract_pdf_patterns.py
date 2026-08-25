"""Tests for the pattern PDF piece extractor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

import fitz
from PIL import Image

from pattern_vision.classifier import GARMENT_TYPES, PIECE_NAMES
from tools.extract_pdf_patterns import (
    PATTERN_PDFS,
    AMELIA_COAT,
    LUXURY_FUR_COAT,
    ZUNES_KIDS_PANTS,
    ExtractResult,
    PatternPDF,
    PieceSpec,
    TileLayout,
    assemble_sheet,
    autotrim,
    crop_fraction,
    downscale,
    extract_all,
    extract_pattern,
    main,
    render_page,
)


def _make_pdf(path: Path, pages: int = 2, size: tuple[float, float] = (612, 792)) -> None:
    """Write a PDF whose Nth page carries a black square in a distinct spot."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        # Square drifts down-right per page so pages are distinguishable.
        x, y = 100 + i * 40, 100 + i * 40
        page.draw_rect(fitz.Rect(x, y, x + 120, y + 120), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(str(path))
    doc.close()


class TestRegistry(TestCase):
    """The shipped registry must stay consistent with the classifier vocabulary."""

    def test_keys_are_unique(self):
        keys = [p.key for p in PATTERN_PDFS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_labels_are_in_classifier_vocabulary(self):
        for pattern in PATTERN_PDFS:
            for spec in pattern.pieces:
                self.assertIn(spec.garment_type, GARMENT_TYPES, spec.name)
                self.assertIn(spec.piece_name, PIECE_NAMES, spec.name)

    def test_piece_output_names_are_unique(self):
        names = [s.name for p in PATTERN_PDFS for s in p.pieces]
        self.assertEqual(len(names), len(set(names)))

    def test_crops_are_ordered_fractions(self):
        for pattern in PATTERN_PDFS:
            for spec in pattern.pieces:
                x0, y0, x1, y1 = spec.crop
                self.assertLess(x0, x1, spec.name)
                self.assertLess(y0, y1, spec.name)
                for v in spec.crop:
                    self.assertGreaterEqual(v, 0.0, spec.name)
                    self.assertLessEqual(v, 1.0, spec.name)

    def test_every_piece_has_a_source(self):
        for pattern in PATTERN_PDFS:
            for spec in pattern.pieces:
                self.assertTrue(
                    spec.sheet or spec.page is not None,
                    f"{spec.name} declares neither page nor sheet",
                )
                if spec.sheet:
                    self.assertIsNotNone(pattern.tiles, spec.name)

    def test_restricted_entries_carry_a_notice(self):
        for pattern in PATTERN_PDFS:
            if not pattern.redistributable:
                self.assertTrue(pattern.notice.strip(), pattern.key)

    def test_patterns_with_explicit_no_distribution_terms_are_restricted(self):
        # These two PDFs forbid redistribution in their own text; keeping the
        # flag correct is what stops them being committed to a public repo.
        self.assertFalse(AMELIA_COAT.redistributable)
        self.assertFalse(LUXURY_FUR_COAT.redistributable)


class TestTileLayout(TestCase):
    def test_rows_for_full_grid(self):
        self.assertEqual(TileLayout(pages=[1, 2, 3, 4], columns=2).rows, 2)

    def test_rows_round_up_for_ragged_grid(self):
        self.assertEqual(TileLayout(pages=[1, 2, 3], columns=2).rows, 2)

    def test_kids_pants_content_box_is_inset_from_the_page(self):
        # These pages overlap rather than butt-join; the trim box is what makes
        # the assembled curves meet, so a full-page box here would be a bug.
        x0, y0, x1, y1 = ZUNES_KIDS_PANTS.tiles.content_box
        self.assertGreater(x0, 0.0)
        self.assertGreater(y0, 0.0)
        self.assertLess(x1, 1.0)
        self.assertLess(y1, 1.0)

    def test_amelia_grid_covers_every_tile_page(self):
        layout = AMELIA_COAT.tiles
        self.assertEqual(layout.columns, 8)
        self.assertEqual(layout.rows, 5)
        real = [p for p in layout.pages if p is not None]
        self.assertEqual(real, list(range(5, 43)))


class TestImageHelpers(TestCase):
    def test_crop_fraction_takes_requested_region(self):
        img = Image.new("RGB", (200, 100), "white")
        self.assertEqual(crop_fraction(img, (0.0, 0.0, 0.5, 1.0)).size, (100, 100))

    def test_autotrim_reduces_to_ink_plus_margin(self):
        img = Image.new("RGB", (400, 400), "white")
        img.paste(Image.new("RGB", (100, 100), "black"), (150, 150))
        out = autotrim(img, margin=10)
        self.assertEqual(out.size, (120, 120))

    def test_autotrim_ignores_pale_background(self):
        # A pale grid (like Butterick's enlargement grid) must not defeat the trim.
        img = Image.new("RGB", (300, 300), (225, 235, 250))
        img.paste(Image.new("RGB", (60, 60), "black"), (120, 120))
        self.assertEqual(autotrim(img, margin=0).size, (60, 60))

    def test_autotrim_returns_blank_image_unchanged(self):
        img = Image.new("RGB", (50, 60), "white")
        self.assertEqual(autotrim(img).size, (50, 60))

    def test_downscale_caps_longest_side_and_keeps_aspect(self):
        out = downscale(Image.new("RGB", (4000, 1000)), max_dim=2000)
        self.assertEqual(out.size, (2000, 500))

    def test_downscale_caps_on_height_when_portrait(self):
        out = downscale(Image.new("RGB", (500, 5000)), max_dim=1000)
        self.assertEqual(out.size, (100, 1000))

    def test_downscale_leaves_small_images_alone(self):
        img = Image.new("RGB", (300, 200))
        self.assertEqual(downscale(img, max_dim=2000).size, (300, 200))

    def test_downscale_disabled_by_zero(self):
        img = Image.new("RGB", (9000, 20))
        self.assertEqual(downscale(img, max_dim=0).size, (9000, 20))

    def test_downscale_never_produces_a_zero_dimension(self):
        out = downscale(Image.new("RGB", (10000, 3)), max_dim=100)
        self.assertGreaterEqual(min(out.size), 1)


class TestRendering(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf = Path(self.tmp.name) / "sample.pdf"
        _make_pdf(self.pdf, pages=4)
        self.doc = fitz.open(str(self.pdf))

    def tearDown(self):
        self.doc.close()
        self.tmp.cleanup()

    def test_render_page_is_one_indexed(self):
        img = render_page(self.doc, 1, dpi=36)
        self.assertEqual(img.mode, "RGB")
        self.assertGreater(img.width, 0)

    def test_render_page_honours_fractional_box(self):
        full = render_page(self.doc, 1, dpi=36)
        half = render_page(self.doc, 1, dpi=36, box=(0.0, 0.0, 0.5, 1.0))
        self.assertAlmostEqual(half.width, full.width // 2, delta=2)

    def test_assemble_sheet_builds_the_full_grid(self):
        layout = TileLayout(pages=[1, 2, 3, 4], columns=2)
        tile = render_page(self.doc, 1, dpi=36)
        sheet = assemble_sheet(self.doc, layout, dpi=36)
        self.assertEqual(sheet.size, (tile.width * 2, tile.height * 2))

    def test_assemble_sheet_leaves_blank_cells_white(self):
        layout = TileLayout(pages=[1, None, None, None], columns=2)
        sheet = assemble_sheet(self.doc, layout, dpi=36)
        # Bottom-right cell had no page, so it stays white.
        self.assertEqual(sheet.getpixel((sheet.width - 5, sheet.height - 5)), (255, 255, 255))


class TestExtractPattern(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pdf = self.root / "sample.pdf"
        _make_pdf(self.pdf, pages=4)
        self.data = self.root / "data"
        self.pattern = PatternPDF(
            key="sample",
            filename="sample.pdf",
            title="Sample",
            source_name="test",
            license="test-license",
            attribution="tester",
            redistributable=True,
            pieces=[PieceSpec("sample_front", "jacket", "front", page=1, notch_count=2)],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_image_into_label_directories(self):
        res = extract_pattern(self.pdf, self.pattern, data_dir=self.data, dpi=36)
        self.assertEqual(res.written, 1)
        self.assertTrue((self.data / "jacket" / "front" / "sample_front.png").exists())

    def test_writes_sidecar_with_annotations_and_provenance(self):
        extract_pattern(self.pdf, self.pattern, data_dir=self.data, dpi=36)
        sidecar = self.data / "jacket" / "front" / "sample_front.png.json"
        meta = json.loads(sidecar.read_text())
        self.assertEqual(meta["notch_count"], 2)
        self.assertEqual(meta["license"], "test-license")
        self.assertEqual(meta["attribution"], "tester")
        self.assertTrue(meta["redistributable"])

    def test_saved_image_respects_max_dimension(self):
        extract_pattern(
            self.pdf, self.pattern, data_dir=self.data, dpi=150, max_dimension=64
        )
        img = Image.open(self.data / "jacket" / "front" / "sample_front.png")
        self.assertLessEqual(max(img.size), 64)

    def test_dry_run_writes_nothing(self):
        res = extract_pattern(self.pdf, self.pattern, data_dir=self.data, dpi=36, dry_run=True)
        self.assertEqual(res.written, 1)
        self.assertFalse(self.data.exists())

    def test_existing_image_is_skipped(self):
        extract_pattern(self.pdf, self.pattern, data_dir=self.data, dpi=36)
        res = extract_pattern(self.pdf, self.pattern, data_dir=self.data, dpi=36)
        self.assertEqual(res.written, 0)
        self.assertEqual(res.skipped, 1)

    def test_overwrite_replaces_existing_image(self):
        extract_pattern(self.pdf, self.pattern, data_dir=self.data, dpi=36)
        res = extract_pattern(
            self.pdf, self.pattern, data_dir=self.data, dpi=36, overwrite=True
        )
        self.assertEqual(res.written, 1)

    def test_sheet_piece_without_tile_layout_is_reported_as_failure(self):
        broken = PatternPDF(
            key="broken", filename="sample.pdf", title="Broken", source_name="test",
            license="x", attribution="y", redistributable=True,
            pieces=[PieceSpec("broken_front", "jacket", "front", sheet=True)],
        )
        res = extract_pattern(self.pdf, broken, data_dir=self.data, dpi=36)
        self.assertEqual(res.failed, 1)
        self.assertEqual(res.written, 0)

    def test_tiled_piece_is_cropped_from_the_assembled_sheet(self):
        tiled = PatternPDF(
            key="tiled", filename="sample.pdf", title="Tiled", source_name="test",
            license="x", attribution="y", redistributable=True,
            tiles=TileLayout(pages=[1, 2, 3, 4], columns=2),
            pieces=[PieceSpec("tiled_front", "jacket", "front", sheet=True)],
        )
        res = extract_pattern(self.pdf, tiled, data_dir=self.data, dpi=36)
        self.assertEqual(res.written, 1)
        self.assertTrue((self.data / "jacket" / "front" / "tiled_front.png").exists())


class TestExtractAll(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_pdf_is_skipped_not_fatal(self):
        res = extract_all(pdf_dir=self.root, data_dir=self.data, dry_run=True)
        self.assertEqual(res.written, 0)
        self.assertEqual(res.failed, 0)

    def test_restricted_entries_are_skipped_by_default(self):
        _make_pdf(self.root / AMELIA_COAT.filename, pages=45)
        res = extract_all(
            pdf_dir=self.root, data_dir=self.data, key="amelia_coat", dry_run=True
        )
        self.assertEqual(res.written, 0)
        self.assertEqual(res.skipped, len(AMELIA_COAT.pieces))

    def test_restricted_entries_run_when_explicitly_included(self):
        _make_pdf(self.root / AMELIA_COAT.filename, pages=45)
        res = extract_all(
            pdf_dir=self.root, data_dir=self.data, key="amelia_coat",
            dry_run=True, include_restricted=True,
        )
        self.assertEqual(res.written, len(AMELIA_COAT.pieces))


class TestCLI(TestCase):
    def test_list_exits_clean(self):
        self.assertEqual(main(["--list"]), 0)

    def test_run_with_no_pdfs_present_exits_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--pdf-dir", tmp, "--data-dir", str(Path(tmp) / "out"), "--dry-run"])
            self.assertEqual(code, 0)


class TestExtractResult(TestCase):
    def test_defaults_are_zero(self):
        res = ExtractResult()
        self.assertEqual((res.written, res.skipped, res.failed), (0, 0, 0))
        self.assertEqual(res.paths, [])
