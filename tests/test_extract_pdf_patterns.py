"""Tests for the pattern PDF piece extractor."""

from __future__ import annotations

import hashlib
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
    BUTTERICK_RETRO_WRAP,
    LUXURY_FUR_COAT,
    ZUNES_KIDS_PANTS,
    ExtractResult,
    PatternPDF,
    PieceSpec,
    TileLayout,
    assemble_sheet,
    autotrim,
    check_paths,
    check_pdf,
    crop_fraction,
    downscale,
    find_registry_entry,
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


def _text_pdf(path: Path, body: str, pages: int = 1) -> None:
    """Write a PDF whose first page carries ``body`` as real, searchable text."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(fitz.Rect(40, 40, 572, 752), body if i == 0 else "x", fontsize=9)
    doc.save(str(path))
    doc.close()


# Enough filler to clear MIN_TEXT_FOR_VERDICT so the scanned-PDF path is not
# what a terms test ends up exercising.
_FILLER = ("This pattern includes seam allowances and a full set of pieces. " * 6)


class TestLicenseCheck(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _check(self, body: str, name: str = "x.pdf"):
        path = self.root / name
        _text_pdf(path, body)
        return check_pdf(path)

    def test_personal_use_only_is_restricted(self):
        res = self._check(_FILLER + " This pattern is for personal use only.")
        self.assertEqual(res.verdict, "restricted")
        self.assertTrue(res.evidence)

    def test_no_redistribution_clause_is_restricted(self):
        res = self._check(_FILLER + " Pattern may not be shared, sold or re-distributed.")
        self.assertEqual(res.verdict, "restricted")

    def test_no_reposting_clause_is_restricted(self):
        res = self._check(_FILLER + " You may not re-post the patterns to the web.")
        self.assertEqual(res.verdict, "restricted")

    def test_not_to_be_reprinted_is_restricted(self):
        res = self._check(_FILLER + " Not to be reprinted. All rights reserved.")
        self.assertEqual(res.verdict, "restricted")

    def test_bare_all_rights_reserved_is_not_restricted(self):
        # Free promotional patterns carry this alongside permission to
        # download; treating it as a block would reject most usable patterns.
        res = self._check(_FILLER + " (c)2008 The Pattern Company, All rights reserved.")
        self.assertEqual(res.verdict, "no terms found")

    def test_commercial_use_ban_alone_is_not_restricted(self):
        res = self._check(_FILLER + " Commercial or industrial use prohibited.")
        self.assertEqual(res.verdict, "no terms found")

    def test_clause_split_across_lines_is_still_caught(self):
        res = self._check(_FILLER + " Pattern may not be\nshared or re-distributed\nwithout consent.")
        self.assertEqual(res.verdict, "restricted")

    def test_pdf_without_text_layer_is_not_given_an_all_clear(self):
        path = self.root / "scan.pdf"
        _make_pdf(path, pages=2)  # drawings only, no text
        res = check_pdf(path)
        self.assertEqual(res.verdict, "no text layer")
        self.assertNotEqual(res.verdict, "no terms found")

    def test_evidence_quotes_the_matching_phrase(self):
        res = self._check(_FILLER + " This pattern is for personal use only.")
        self.assertIn("personal use only", res.evidence[0])

    def test_registry_match_by_content_hash(self):
        path = self.root / "renamed-by-the-user.pdf"
        _text_pdf(path, "whatever")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = PatternPDF(
            key="probe", filename="not-this-name.pdf", sha256=digest,
            title="t", source_name="s", license="l", attribution="a",
            redistributable=True,
        )
        PATTERN_PDFS.append(entry)
        try:
            self.assertIs(find_registry_entry(path), entry)
        finally:
            PATTERN_PDFS.remove(entry)

    def test_registry_match_by_filename_when_hash_is_unknown(self):
        path = self.root / BUTTERICK_RETRO_WRAP.filename
        _text_pdf(path, "different bytes entirely")
        self.assertIs(find_registry_entry(path), BUTTERICK_RETRO_WRAP)

    def test_unrelated_file_matches_nothing(self):
        path = self.root / "brand-new-pattern.pdf"
        _text_pdf(path, "hello")
        self.assertIsNone(find_registry_entry(path))

    def test_check_paths_walks_a_directory(self):
        _text_pdf(self.root / "a.pdf", _FILLER)
        _text_pdf(self.root / "b.pdf", _FILLER + " for personal use only")
        results = check_paths([self.root])
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {r.path.name: r.verdict for r in results},
            {"a.pdf": "no terms found", "b.pdf": "restricted"},
        )

    def test_check_paths_ignores_non_pdfs(self):
        (self.root / "notes.txt").write_text("for personal use only")
        self.assertEqual(check_paths([self.root]), [])

    def test_identity_reports_a_publisher_line(self):
        res = self._check(_FILLER + " (c)MMV Kwik-Sew Pattern Co., Inc.")
        joined = " ".join(res.identity)
        self.assertIn("Kwik-Sew", joined)

    def test_identity_reports_a_url(self):
        res = self._check(_FILLER + " Find more at www.example-patterns.com today")
        self.assertTrue(any("example-patterns.com" in line for line in res.identity))

    def test_identity_reports_a_pattern_number(self):
        res = self._check(_FILLER + " Pattern 5001 for the clutch purse")
        self.assertTrue(any("5001" in line for line in res.identity))

    def test_identity_is_capped(self):
        noisy = _FILLER + " ".join(f"http://example{i}.com" for i in range(30))
        self.assertLessEqual(len(self._check(noisy).identity), 6)

    def test_identity_is_empty_when_the_file_says_nothing(self):
        path = self.root / "silent.pdf"
        _make_pdf(path, pages=1)  # drawings only, no text, no metadata
        self.assertEqual(check_pdf(path).identity, [])

    def test_identity_does_not_change_the_verdict(self):
        # Naming a publisher is not a restriction.
        res = self._check(_FILLER + " (c)MMV Kwik-Sew Pattern Co., Inc.")
        self.assertEqual(res.verdict, "no terms found")

    def test_registry_hashes_are_well_formed(self):
        for pattern in PATTERN_PDFS:
            if pattern.sha256:
                self.assertEqual(len(pattern.sha256), 64, pattern.key)
                int(pattern.sha256, 16)  # raises if not hex

    def test_registry_hashes_are_unique(self):
        hashes = [p.sha256 for p in PATTERN_PDFS if p.sha256]
        self.assertEqual(len(hashes), len(set(hashes)))


class TestExtractResult(TestCase):
    def test_defaults_are_zero(self):
        res = ExtractResult()
        self.assertEqual((res.written, res.skipped, res.failed), (0, 0, 0))
        self.assertEqual(res.paths, [])
