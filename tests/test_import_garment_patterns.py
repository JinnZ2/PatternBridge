"""Tests for the Garment-Pattern-Generator geometry importer."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest import TestCase

from pattern_geometry.encoder import PatternEncoder
from pattern_vision.classifier import GARMENT_TYPES, PIECE_NAMES
from tools.import_garment_patterns import (
    CM_PER_INCH,
    DEFAULT_CURVE_SAMPLES,
    ImportResult,
    control_point,
    find_templates,
    garment_type_for,
    import_templates,
    main,
    order_edges,
    panel_boundary,
    piece_name_for,
    sample_quadratic,
    template_to_pieces,
)

# A unit square panel: vertices counter-clockwise, all edges straight.
SQUARE_PANEL = {
    "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
    "edges": [
        {"endpoints": [0, 1]},
        {"endpoints": [1, 2]},
        {"endpoints": [2, 3]},
        {"endpoints": [3, 0]},
    ],
}


def _template(panels: dict, units: int = 100) -> dict:
    return {
        "pattern": {"panels": panels},
        "properties": {"units_in_meter": units, "curvature_coords": "relative"},
    }


def _write(path: Path, panels: dict, units: int = 100) -> Path:
    path.write_text(json.dumps(_template(panels, units)))
    return path


class TestControlPoint(TestCase):
    """The convention is copied from the generator's _control_to_abs_coord."""

    def test_midpoint_offset_is_perpendicular_to_the_edge(self):
        # Edge along +x, so the perpendicular is +y in this frame.
        got = control_point((0.0, 0.0), (10.0, 0.0), [0.5, 0.1])
        self.assertAlmostEqual(got[0], 5.0)
        self.assertAlmostEqual(got[1], 1.0)

    def test_zero_curvature_lands_on_the_edge(self):
        got = control_point((0.0, 0.0), (10.0, 0.0), [0.5, 0.0])
        self.assertAlmostEqual(got[1], 0.0)

    def test_negative_perpendicular_bends_the_other_way(self):
        got = control_point((0.0, 0.0), (10.0, 0.0), [0.5, -0.1])
        self.assertAlmostEqual(got[1], -1.0)

    def test_offset_scales_with_edge_length(self):
        short = control_point((0.0, 0.0), (10.0, 0.0), [0.5, 0.1])
        long = control_point((0.0, 0.0), (20.0, 0.0), [0.5, 0.1])
        self.assertAlmostEqual(long[1], short[1] * 2)

    def test_works_on_a_diagonal_edge(self):
        got = control_point((0.0, 0.0), (0.0, 10.0), [0.5, 0.1])
        # Edge along +y, perpendicular is -x.
        self.assertAlmostEqual(got[0], -1.0)
        self.assertAlmostEqual(got[1], 5.0)


class TestSampleQuadratic(TestCase):
    def test_returns_requested_count_and_starts_at_start(self):
        pts = sample_quadratic((0.0, 0.0), (5.0, 5.0), (10.0, 0.0), 8)
        self.assertEqual(len(pts), 8)
        self.assertAlmostEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[0][1], 0.0)

    def test_excludes_the_end_point(self):
        pts = sample_quadratic((0.0, 0.0), (5.0, 5.0), (10.0, 0.0), 8)
        self.assertNotAlmostEqual(pts[-1][0], 10.0)

    def test_straight_control_gives_a_straight_line(self):
        pts = sample_quadratic((0.0, 0.0), (5.0, 0.0), (10.0, 0.0), 5)
        for _, y in pts:
            self.assertAlmostEqual(y, 0.0)

    def test_curve_bulges_toward_the_control_point(self):
        pts = sample_quadratic((0.0, 0.0), (5.0, 10.0), (10.0, 0.0), 9)
        self.assertGreater(max(y for _, y in pts), 0.0)


class TestOrderEdges(TestCase):
    def test_already_ordered_loop_is_preserved(self):
        loop = order_edges(SQUARE_PANEL["edges"])
        self.assertEqual([(a, b) for a, b, _ in loop], [(0, 1), (1, 2), (2, 3), (3, 0)])

    def test_shuffled_edges_are_chained_into_a_loop(self):
        shuffled = [
            {"endpoints": [2, 3]},
            {"endpoints": [0, 1]},
            {"endpoints": [3, 0]},
            {"endpoints": [1, 2]},
        ]
        loop = order_edges(shuffled)
        self.assertEqual(len(loop), 4)
        # Each edge must start where the previous one ended.
        for (_, prev_end, _), (next_start, _, _) in zip(loop, loop[1:]):
            self.assertEqual(prev_end, next_start)

    def test_reversed_edge_is_flipped_and_its_curvature_adjusted(self):
        edges = [
            {"endpoints": [0, 1]},
            {"endpoints": [2, 1], "curvature": [0.25, 0.1]},
        ]
        loop = order_edges(edges)
        start, end, curvature = loop[1]
        self.assertEqual((start, end), (1, 2))
        self.assertAlmostEqual(curvature[0], 0.75)
        self.assertAlmostEqual(curvature[1], -0.1)

    def test_empty_edge_list_gives_empty_loop(self):
        self.assertEqual(order_edges([]), [])

    def test_disjoint_edges_are_kept_not_dropped(self):
        edges = [{"endpoints": [0, 1]}, {"endpoints": [5, 6]}]
        self.assertEqual(len(order_edges(edges)), 2)


class TestPanelBoundary(TestCase):
    def test_straight_panel_gives_one_point_per_edge(self):
        boundary = panel_boundary(SQUARE_PANEL)
        self.assertEqual(len(boundary), 4)

    def test_centimetres_are_converted_to_inches(self):
        boundary = panel_boundary(SQUARE_PANEL)
        xs = [x for x, _ in boundary]
        self.assertAlmostEqual(max(xs) - min(xs), 10.0 / CM_PER_INCH)

    def test_other_unit_scales_are_honoured(self):
        # units_in_meter=1000 means millimetres, so the panel is 10x smaller.
        boundary = panel_boundary(SQUARE_PANEL, units_in_meter=1000)
        xs = [x for x, _ in boundary]
        self.assertAlmostEqual(max(xs) - min(xs), 1.0 / CM_PER_INCH)

    def test_y_axis_is_flipped_to_this_projects_convention(self):
        # Template y-up: vertex [0, 10] is the top. After the flip it must be
        # the smallest y, because output space is y-down.
        boundary = panel_boundary(SQUARE_PANEL)
        top_vertex = boundary[2]  # vertices[2] == [10, 10]
        self.assertLess(top_vertex[1], 0)

    def test_curved_edge_adds_sampled_points(self):
        panel = json.loads(json.dumps(SQUARE_PANEL))
        panel["edges"][0]["curvature"] = [0.5, 0.2]
        boundary = panel_boundary(panel, samples=6)
        self.assertEqual(len(boundary), 3 + 6)

    def test_panel_without_vertices_gives_empty_boundary(self):
        self.assertEqual(panel_boundary({"vertices": [], "edges": []}), [])

    def test_out_of_range_endpoints_are_skipped_not_fatal(self):
        panel = {"vertices": [[0, 0], [1, 0]], "edges": [{"endpoints": [0, 9]}]}
        self.assertEqual(panel_boundary(panel), [])


class TestLabelling(TestCase):
    def test_sleeve_wins_over_front_in_compound_names(self):
        self.assertEqual(piece_name_for("lfsleeve"), "sleeve")
        self.assertEqual(piece_name_for("rbsleeve"), "sleeve")

    def test_waistband_prefix(self):
        self.assertEqual(piece_name_for("wb_front"), "waistband")

    def test_plain_front_and_back(self):
        self.assertEqual(piece_name_for("front"), "front")
        self.assertEqual(piece_name_for("top_back"), "back")

    def test_side_panels(self):
        self.assertEqual(piece_name_for("right_0"), "side")
        self.assertEqual(piece_name_for("left_2"), "side")

    def test_unknown_panel_falls_back_to_other(self):
        self.assertEqual(piece_name_for("gusset"), "other")

    def test_every_rule_targets_a_real_piece_name(self):
        for name in ("lfsleeve", "wb_front", "front", "back", "left", "hood", "zzz"):
            self.assertIn(piece_name_for(name), PIECE_NAMES)

    def test_jacket_filename_beats_the_basic_tee_directory(self):
        # The generator files jackets alongside tees; a jacket is not a top.
        self.assertEqual(garment_type_for(Path("basic tee/jacket.json")), "jacket")
        self.assertEqual(garment_type_for(Path("basic tee/tee.json")), "top")

    def test_directory_drives_the_common_cases(self):
        self.assertEqual(garment_type_for(Path("skirts/skirt_4_panels.json")), "skirt")
        self.assertEqual(garment_type_for(Path("pants/pants_flare.json")), "pants")

    def test_combos_resolve_by_filename(self):
        self.assertEqual(garment_type_for(Path("combos/dress.json")), "dress")
        self.assertEqual(garment_type_for(Path("combos/jumpsuit.json")), "other")


class TestTemplateToPieces(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_piece_per_panel(self):
        path = _write(self.root / "skirt_test.json",
                      {"front": SQUARE_PANEL, "back": SQUARE_PANEL})
        self.assertEqual(len(template_to_pieces(path)), 2)

    def test_piece_records_its_provenance(self):
        path = _write(self.root / "skirt_test.json", {"front": SQUARE_PANEL})
        piece = template_to_pieces(path)[0]
        self.assertEqual(piece.pattern_brand, "garment-pattern-generator")
        self.assertEqual(piece.pattern_id, "skirt_test")
        self.assertIn("skirt_test.json#front", piece.image_source)

    def test_piece_clears_the_encodable_gate(self):
        # The gate keys off vision score; imported geometry is exact, so it
        # must not be treated as an untrustworthy photograph.
        path = _write(self.root / "skirt_test.json", {"front": SQUARE_PANEL})
        piece = template_to_pieces(path)[0]
        self.assertTrue(piece.is_encodable)
        self.assertFalse(piece.needs_better_image)

    def test_band_label_says_what_the_score_means(self):
        path = _write(self.root / "skirt_test.json", {"front": SQUARE_PANEL})
        piece = template_to_pieces(path)[0]
        self.assertIn("Imported geometry", piece.band_label)

    def test_piece_gets_a_default_grain_line(self):
        path = _write(self.root / "skirt_test.json", {"front": SQUARE_PANEL})
        grain = template_to_pieces(path)[0].grain_line
        self.assertIsNotNone(grain)
        self.assertAlmostEqual(grain.angle_degrees, 90.0)
        self.assertAlmostEqual(grain.start[0], grain.end[0])

    def test_degenerate_panel_is_skipped(self):
        thin = {"vertices": [[0, 0], [1, 1]], "edges": [{"endpoints": [0, 1]}]}
        path = _write(self.root / "skirt_test.json", {"front": thin})
        self.assertEqual(template_to_pieces(path), [])

    def test_imported_piece_encodes(self):
        path = _write(self.root / "skirt_test.json", {"front": SQUARE_PANEL})
        piece = template_to_pieces(path)[0]
        PatternEncoder().encode(piece)
        self.assertTrue(piece.encoded_tokens)

    def test_labels_stay_inside_the_classifier_vocabulary(self):
        path = _write(self.root / "skirt_test.json",
                      {"front": SQUARE_PANEL, "lfsleeve": SQUARE_PANEL})
        for piece in template_to_pieces(path):
            self.assertIn(piece.garment_type, GARMENT_TYPES)


class TestFindAndImport(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patterns = self.root / "data_generation" / "Patterns" / "skirts"
        self.patterns.mkdir(parents=True)
        _write(self.patterns / "skirt_a.json", {"front": SQUARE_PANEL})
        _write(self.patterns / "skirt_b.json", {"front": SQUARE_PANEL, "back": SQUARE_PANEL})

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_templates_from_the_repo_root(self):
        self.assertEqual(len(find_templates(self.root)), 2)

    def test_finds_templates_from_the_patterns_directory(self):
        self.assertEqual(len(find_templates(self.patterns)), 2)

    def test_non_template_json_is_ignored(self):
        (self.patterns / "config.json").write_text(json.dumps({"some": "config"}))
        self.assertEqual(len(find_templates(self.root)), 2)

    def test_unreadable_json_is_ignored(self):
        (self.patterns / "broken.json").write_text("{not json")
        self.assertEqual(len(find_templates(self.root)), 2)

    def test_missing_source_returns_nothing(self):
        self.assertEqual(find_templates(self.root / "nope"), [])

    def test_import_counts_templates_and_pieces(self):
        result = import_templates(self.root)
        self.assertEqual(result.templates, 2)
        self.assertEqual(result.pieces, 3)
        self.assertEqual(result.failed, 0)

    def test_import_writes_files_under_garment_type(self):
        out = self.root / "out"
        import_templates(self.root, out_dir=out)
        self.assertTrue((out / "skirt" / "skirt_a" / "front.json").exists())
        self.assertTrue((out / "skirt" / "skirt_a" / "manifest.json").exists())

    def test_dry_run_writes_nothing(self):
        out = self.root / "out"
        result = import_templates(self.root, out_dir=out, dry_run=True)
        self.assertEqual(result.pieces, 3)
        self.assertFalse(out.exists())

    def test_written_piece_json_round_trips(self):
        out = self.root / "out"
        import_templates(self.root, out_dir=out)
        data = json.loads((out / "skirt" / "skirt_a" / "front.json").read_text())
        self.assertEqual(data["name"], "FRONT")
        self.assertEqual(data["units"], "inches")
        self.assertGreaterEqual(len(data["boundary_points"]), 3)


class TestCLI(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        patterns = self.root / "Patterns"
        patterns.mkdir()
        _write(patterns / "skirt_a.json", {"front": SQUARE_PANEL})

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_exits_clean(self):
        self.assertEqual(main(["--src", str(self.root), "--list"]), 0)

    def test_dry_run_exits_clean(self):
        self.assertEqual(main(["--src", str(self.root), "--dry-run"]), 0)

    def test_missing_source_is_not_fatal(self):
        self.assertEqual(main(["--src", str(self.root / "nope"), "--dry-run"]), 0)


class TestImportResult(TestCase):
    def test_defaults_are_zero(self):
        result = ImportResult()
        self.assertEqual((result.templates, result.pieces, result.failed), (0, 0, 0))
        self.assertEqual(result.paths, [])
