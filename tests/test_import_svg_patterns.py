"""Tests for the SVG pattern importer."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest import TestCase

from pattern_geometry.encoder import PatternEncoder
from pattern_output.svg_writer import SVGWriter
from tools.import_svg_patterns import (
    DEFAULT_MIN_AREA,
    GENERIC_LABELS,
    SvgImportResult,
    apply_matrix,
    import_svg,
    main,
    parse_length,
    parse_path,
    parse_transform,
    polygon_area,
    svg_to_pieces,
    units_per_inch,
)
import xml.etree.ElementTree as ET


def _svg(body: str, width: str = "96px", view_box: str | None = None) -> str:
    box = f' viewBox="{view_box}"' if view_box else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{width}"{box}>{body}</svg>'
    )


def _write(root: Path, body: str, **kwargs) -> Path:
    path = root / "p.svg"
    path.write_text(_svg(body, **kwargs))
    return path


# A 10x10 inch square at the default 96 user units per inch.
SQUARE = '<path id="front" d="M 0,0 H 960 V 960 H 0 Z"/>'


class TestLengthsAndScale(TestCase):
    def test_parses_value_and_unit(self):
        self.assertEqual(parse_length("210mm"), (210.0, "mm"))
        self.assertEqual(parse_length("8.5in"), (8.5, "in"))

    def test_bare_number_has_no_unit(self):
        self.assertEqual(parse_length("300"), (300.0, ""))

    def test_rejects_nonsense(self):
        self.assertIsNone(parse_length("wide"))
        self.assertIsNone(parse_length(None))

    def test_unitless_width_is_ninety_six_per_inch(self):
        root = ET.fromstring(_svg("", width="960"))
        self.assertAlmostEqual(units_per_inch(root), 96.0)

    def test_millimetre_width_with_viewbox(self):
        # 254mm wide == 10in, mapped onto a 1000-unit viewBox.
        root = ET.fromstring(_svg("", width="254mm", view_box="0 0 1000 1000"))
        self.assertAlmostEqual(units_per_inch(root), 100.0, places=6)

    def test_inch_width_without_viewbox(self):
        root = ET.fromstring(_svg("", width="10in"))
        self.assertAlmostEqual(units_per_inch(root), 1.0)

    def test_percentage_width_falls_back(self):
        root = ET.fromstring(_svg("", width="100%"))
        self.assertAlmostEqual(units_per_inch(root), 96.0)

    def test_override_wins(self):
        root = ET.fromstring(_svg("", width="254mm"))
        self.assertAlmostEqual(units_per_inch(root, 25.4), 25.4)


class TestTransforms(TestCase):
    def test_identity_for_no_transform(self):
        self.assertEqual(parse_transform(None), (1, 0, 0, 1, 0, 0))

    def test_translate(self):
        self.assertAlmostEqual(apply_matrix(parse_transform("translate(5,7)"), (1, 2))[0], 6)
        self.assertAlmostEqual(apply_matrix(parse_transform("translate(5,7)"), (1, 2))[1], 9)

    def test_scale_with_one_argument_is_uniform(self):
        got = apply_matrix(parse_transform("scale(3)"), (2, 4))
        self.assertAlmostEqual(got[0], 6)
        self.assertAlmostEqual(got[1], 12)

    def test_rotate_ninety_degrees(self):
        got = apply_matrix(parse_transform("rotate(90)"), (1, 0))
        self.assertAlmostEqual(got[0], 0, places=6)
        self.assertAlmostEqual(got[1], 1, places=6)

    def test_rotate_about_a_centre_leaves_it_fixed(self):
        got = apply_matrix(parse_transform("rotate(37, 4, 9)"), (4, 9))
        self.assertAlmostEqual(got[0], 4, places=6)
        self.assertAlmostEqual(got[1], 9, places=6)

    def test_transforms_compose_left_to_right(self):
        # translate then scale: the translation is scaled too.
        got = apply_matrix(parse_transform("scale(2) translate(10,0)"), (0, 0))
        self.assertAlmostEqual(got[0], 20)

    def test_matrix_form(self):
        got = apply_matrix(parse_transform("matrix(1,0,0,1,3,4)"), (0, 0))
        self.assertEqual(got, (3, 4))


class TestPathParsing(TestCase):
    def test_absolute_line_square_closes(self):
        subs = parse_path("M 0,0 L 10,0 L 10,10 L 0,10 Z")
        self.assertEqual(len(subs), 1)
        self.assertTrue(subs[0]["closed"])
        self.assertAlmostEqual(polygon_area(subs[0]["points"]), 100.0)

    def test_relative_commands(self):
        subs = parse_path("m 0,0 l 10,0 l 0,10 l -10,0 z")
        self.assertAlmostEqual(polygon_area(subs[0]["points"]), 100.0)

    def test_horizontal_and_vertical(self):
        subs = parse_path("M 0,0 H 10 V 10 H 0 Z")
        self.assertAlmostEqual(polygon_area(subs[0]["points"]), 100.0)

    def test_implicit_lineto_after_moveto(self):
        # Extra coordinate pairs after M are implicit L commands.
        subs = parse_path("M 0,0 10,0 10,10 0,10 Z")
        self.assertAlmostEqual(polygon_area(subs[0]["points"]), 100.0)

    def test_open_path_is_not_closed(self):
        subs = parse_path("M 0,0 L 10,0")
        self.assertFalse(subs[0]["closed"])

    def test_multiple_subpaths(self):
        subs = parse_path("M 0,0 H 10 V 10 H 0 Z M 20,20 H 30 V 30 H 20 Z")
        self.assertEqual(len(subs), 2)
        self.assertTrue(all(s["closed"] for s in subs))

    def test_cubic_curve_is_sampled(self):
        subs = parse_path("M 0,0 C 0,10 10,10 10,0", samples=8)
        self.assertEqual(len(subs[0]["points"]), 9)   # start + 8 samples

    def test_quadratic_curve_is_sampled(self):
        subs = parse_path("M 0,0 Q 5,10 10,0", samples=6)
        self.assertEqual(len(subs[0]["points"]), 7)

    def test_smooth_cubic_reflects_previous_control(self):
        # Symmetric S after C should come back to y=0 at the end.
        subs = parse_path("M 0,0 C 0,5 5,5 5,0 S 10,-5 10,0", samples=4)
        self.assertAlmostEqual(subs[0]["points"][-1][0], 10.0, places=6)
        self.assertAlmostEqual(subs[0]["points"][-1][1], 0.0, places=6)

    def test_smooth_quadratic(self):
        subs = parse_path("M 0,0 Q 5,10 10,0 T 20,0", samples=4)
        self.assertAlmostEqual(subs[0]["points"][-1][0], 20.0, places=6)

    def test_arc_traces_a_semicircle_of_the_right_radius(self):
        # Half circle of radius 5 from (0,0) to (10,0).
        subs = parse_path("M 0,0 A 5,5 0 0 1 10,0", samples=24)
        points = subs[0]["points"]
        self.assertAlmostEqual(points[-1][0], 10.0, places=4)
        peak = max(abs(y) for _, y in points)
        self.assertAlmostEqual(peak, 5.0, places=2)

    def test_arc_with_zero_radius_degrades_to_a_line(self):
        subs = parse_path("M 0,0 A 0,0 0 0 1 10,0")
        self.assertAlmostEqual(subs[0]["points"][-1][0], 10.0)

    def test_empty_path_is_empty(self):
        self.assertEqual(parse_path(""), [])

    def test_garbage_does_not_raise(self):
        self.assertIsInstance(parse_path("M banana Z"), list)


class TestPolygonArea(TestCase):
    def test_square(self):
        self.assertAlmostEqual(polygon_area([(0, 0), (4, 0), (4, 4), (0, 4)]), 16.0)

    def test_winding_direction_does_not_matter(self):
        clockwise = [(0, 0), (0, 4), (4, 4), (4, 0)]
        self.assertAlmostEqual(polygon_area(clockwise), 16.0)

    def test_degenerate_is_zero(self):
        self.assertEqual(polygon_area([(0, 0), (1, 1)]), 0.0)


class TestSvgToPieces(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_square_comes_back_at_the_right_size(self):
        piece = svg_to_pieces(_write(self.root, SQUARE))[0]
        xs = [p[0] for p in piece.boundary_points]
        self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=6)

    def test_millimetre_document_scales_correctly(self):
        body = '<path id="front" d="M 0,0 H 254 V 254 H 0 Z"/>'
        path = _write(self.root, body, width="254mm", view_box="0 0 254 254")
        piece = svg_to_pieces(path)[0]
        xs = [p[0] for p in piece.boundary_points]
        self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=4)

    def test_piece_name_from_id(self):
        self.assertEqual(svg_to_pieces(_write(self.root, SQUARE))[0].name, "FRONT")

    def test_generic_group_name_is_ignored_in_favour_of_the_piece_group(self):
        body = ('<g id="piece_0_BACK"><g id="boundary">'
                '<path d="M 0,0 H 960 V 960 H 0 Z"/></g></g>')
        self.assertEqual(svg_to_pieces(_write(self.root, body))[0].name, "BACK")

    def test_small_paths_are_treated_as_marks(self):
        body = SQUARE + '<path id="notch" d="M 0,0 H 10 V 10 H 0 Z"/>'
        pieces = svg_to_pieces(_write(self.root, body))
        self.assertEqual(len(pieces), 1)

    def test_min_area_is_adjustable(self):
        body = SQUARE + '<path id="pocket" d="M 0,0 H 200 V 200 H 0 Z"/>'
        self.assertEqual(len(svg_to_pieces(_write(self.root, body), min_area=0.5)), 2)

    def test_open_paths_are_not_pieces(self):
        body = '<path id="grain" d="M 0,0 L 960,960"/>'
        self.assertEqual(svg_to_pieces(_write(self.root, body)), [])

    def test_pieces_come_back_largest_first(self):
        body = ('<path id="small" d="M 0,0 H 300 V 300 H 0 Z"/>'
                '<path id="big" d="M 0,0 H 960 V 960 H 0 Z"/>')
        pieces = svg_to_pieces(_write(self.root, body), min_area=0.5)
        self.assertEqual(pieces[0].name, "BIG")

    def test_group_transform_is_applied(self):
        body = f'<g transform="translate(960,0)">{SQUARE}</g>'
        piece = svg_to_pieces(_write(self.root, body))[0]
        self.assertAlmostEqual(min(p[0] for p in piece.boundary_points), 10.0, places=6)

    def test_nested_transforms_compose(self):
        body = f'<g transform="translate(480,0)"><g transform="translate(480,0)">{SQUARE}</g></g>'
        piece = svg_to_pieces(_write(self.root, body))[0]
        self.assertAlmostEqual(min(p[0] for p in piece.boundary_points), 10.0, places=6)

    def test_rect_element_is_supported(self):
        body = '<rect id="band" x="0" y="0" width="960" height="960"/>'
        self.assertEqual(len(svg_to_pieces(_write(self.root, body))), 1)

    def test_polygon_element_is_supported(self):
        body = '<polygon id="wedge" points="0,0 960,0 960,960 0,960"/>'
        self.assertEqual(len(svg_to_pieces(_write(self.root, body))), 1)

    def test_piece_records_provenance_and_clears_the_encodable_gate(self):
        piece = svg_to_pieces(_write(self.root, SQUARE))[0]
        self.assertEqual(piece.pattern_brand, "svg-import")
        self.assertTrue(piece.is_encodable)
        self.assertIn("Imported geometry", piece.band_label)

    def test_piece_gets_a_vertical_grain_line(self):
        grain = svg_to_pieces(_write(self.root, SQUARE))[0].grain_line
        self.assertIsNotNone(grain)
        self.assertAlmostEqual(grain.angle_degrees, 90.0)

    def test_imported_piece_encodes(self):
        piece = svg_to_pieces(_write(self.root, SQUARE))[0]
        PatternEncoder().encode(piece)
        self.assertTrue(piece.encoded_tokens)


class TestRoundTrip(TestCase):
    """Write a piece with SVGWriter, read it back, and compare."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _round_trip(self, boundary):
        from pattern_geometry.piece import PatternPiece
        original = PatternPiece(
            name="FRONT", piece_number=1, garment_type="skirt",
            boundary_points=boundary, total_vision_score=100.0,
        )
        svg = self.root / "rt.svg"
        SVGWriter().save(original, svg)
        return original, svg_to_pieces(svg)[0]

    def test_rectangle_survives_the_trip(self):
        original, back = self._round_trip([(0, 0), (12, 0), (12, 20), (0, 20)])
        for piece in (original, back):
            xs = [p[0] for p in piece.boundary_points]
            ys = [p[1] for p in piece.boundary_points]
            self.assertAlmostEqual(max(xs) - min(xs), 12.0, places=3)
            self.assertAlmostEqual(max(ys) - min(ys), 20.0, places=3)

    def test_curved_outline_keeps_its_dimensions(self):
        boundary = [
            (math.cos(math.radians(a)) * 6 + 6, math.sin(math.radians(a)) * 9 + 9)
            for a in range(0, 360, 12)
        ]
        original, back = self._round_trip(boundary)
        ow = max(p[0] for p in original.boundary_points) - min(p[0] for p in original.boundary_points)
        bw = max(p[0] for p in back.boundary_points) - min(p[0] for p in back.boundary_points)
        self.assertAlmostEqual(ow, bw, places=3)

    def test_name_survives_the_trip(self):
        _, back = self._round_trip([(0, 0), (12, 0), (12, 20), (0, 20)])
        self.assertEqual(back.name, "FRONT")


class TestImportAndCLI(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.svg = _write(self.root, SQUARE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_import_counts_pieces(self):
        self.assertEqual(import_svg(self.svg).pieces, 1)

    def test_import_writes_piece_json(self):
        out = self.root / "out"
        import_svg(self.svg, out_dir=out)
        self.assertTrue((out / "p" / "front.json").exists())

    def test_cli_list_succeeds(self):
        self.assertEqual(main([str(self.svg), "--list"]), 0)

    def test_cli_reports_failure_when_nothing_found(self):
        empty = self.root / "empty.svg"
        empty.write_text(_svg('<path d="M 0,0 L 5,5"/>'))
        self.assertEqual(main([str(empty), "--list"]), 1)

    def test_generic_labels_are_lowercase(self):
        for label in GENERIC_LABELS:
            self.assertEqual(label, label.lower())

    def test_result_defaults(self):
        result = SvgImportResult()
        self.assertEqual((result.pieces, result.paths), (0, []))
        self.assertGreater(DEFAULT_MIN_AREA, 0)
