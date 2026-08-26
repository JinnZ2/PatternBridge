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
    looks_like_freesewing,
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


# Built to the structure FreeSewing's own renderer emits, read from
# packages/core/src/svg.mjs: width/height in mm, a viewBox carrying the same
# numbers (so one user unit is one millimetre), and nested groups named
# "<prefix>stack-<stackId>[-part-<partName>]".
FREESEWING_SVG = """<svg xmlns="http://www.w3.org/2000/svg"
  width="254mm" height="254mm" viewBox="0 0 254 254">
  <!-- Start of group #fs-stack-front -->
  <g id="fs-stack-front">
    <g id="fs-stack-front-part-front">
      <path id="fs-1" d="M 0,0 L 254,0 L 254,254 L 0,254 z"/>
      <path id="fs-2" d="M 20,20 L 40,20"/>
    </g>
  </g>
</svg>"""

# Same pattern with embed on: FreeSewing omits width and height entirely.
FREESEWING_EMBEDDED = """<svg xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 254 254">
  <g id="fs-stack-front"><g id="fs-stack-front-part-back">
    <path id="fs-1" d="M 0,0 L 254,0 L 254,254 L 0,254 z"/>
  </g></g>
</svg>"""


class TestFreeSewingShape(TestCase):
    """Against a fixture matching FreeSewing's documented output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, markup: str) -> Path:
        path = self.root / "fs.svg"
        path.write_text(markup)
        return path

    def test_millimetre_viewbox_gives_true_size(self):
        # 254mm square == 10 inches.
        piece = svg_to_pieces(self._write(FREESEWING_SVG))[0]
        xs = [p[0] for p in piece.boundary_points]
        self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=4)

    def test_part_name_is_recovered_from_the_group_id(self):
        # Not "FS STACK FRONT PART FRONT".
        self.assertEqual(svg_to_pieces(self._write(FREESEWING_SVG))[0].name, "FRONT")

    def test_stack_id_used_when_there_is_no_part_group(self):
        markup = FREESEWING_SVG.replace('<g id="fs-stack-front-part-front">', "<g>")
        self.assertEqual(svg_to_pieces(self._write(markup))[0].name, "FRONT")

    def test_custom_id_prefix_still_resolves(self):
        markup = FREESEWING_SVG.replace("fs-", "mypattern-")
        self.assertEqual(svg_to_pieces(self._write(markup))[0].name, "FRONT")

    def test_helper_lines_are_not_pieces(self):
        # The open fs-2 path is a marking, not an outline.
        self.assertEqual(len(svg_to_pieces(self._write(FREESEWING_SVG))), 1)

    def test_embedded_pattern_is_still_millimetres(self):
        # No width/height: assuming CSS pixels would shrink this 3.8x.
        piece = svg_to_pieces(self._write(FREESEWING_EMBEDDED))[0]
        xs = [p[0] for p in piece.boundary_points]
        self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=4)
        self.assertEqual(piece.name, "BACK")

    def test_freesewing_is_detected(self):
        self.assertTrue(looks_like_freesewing(ET.fromstring(FREESEWING_SVG)))

    def test_ordinary_svg_is_not_mistaken_for_freesewing(self):
        self.assertFalse(looks_like_freesewing(ET.fromstring(_svg(SQUARE))))

    def test_plain_viewbox_svg_still_defaults_to_pixels(self):
        markup = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 960">' + SQUARE + "</svg>"
        piece = svg_to_pieces(self._write(markup))[0]
        xs = [p[0] for p in piece.boundary_points]
        self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=4)


class TestNonRenderedElements(TestCase):
    """<defs> and friends define things; they are not drawn."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clip_path_rectangle_is_not_a_piece(self):
        # A PDF-derived SVG hides a page-sized clip rect in <defs>; taking it
        # for a pattern piece produces a confident, completely wrong answer.
        body = ('<defs><clipPath id="clip_1">'
                '<path d="M 0,0 H 4000 V 4000 H 0 Z"/></clipPath></defs>' + SQUARE)
        pieces = svg_to_pieces(_write(self.root, body))
        self.assertEqual(len(pieces), 1)
        xs = [p[0] for p in pieces[0].boundary_points]
        self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=6)

    def test_glyph_outlines_in_defs_are_not_pieces(self):
        body = ('<defs><path id="font_2_36" d="M 0,0 H 900 V 900 H 0 Z"/></defs>'
                + SQUARE)
        self.assertEqual(len(svg_to_pieces(_write(self.root, body))), 1)

    def test_mask_symbol_and_marker_are_skipped(self):
        big = '<path d="M 0,0 H 4000 V 4000 H 0 Z"/>'
        for tag in ("mask", "symbol", "marker", "pattern"):
            body = f"<{tag} id=\"x\">{big}</{tag}>" + SQUARE
            with self.subTest(tag=tag):
                self.assertEqual(len(svg_to_pieces(_write(self.root, body))), 1)


class TestRealFreeSewingGeometry(TestCase):
    """
    Path data lifted from an actual FreeSewing Aaron v4.10.1 export.

    The pattern prints a calibration box and states its true size on the sheet
    — "the (black) outside of this box should measure 4in x 2in" — which makes
    it ground truth for the whole chain: path parsing, transform composition
    and unit scaling together.
    """

    # Verbatim from the export. 163.66 - 62.06 = 101.6 units wide, and
    # 101.6mm is 4 inches, so FreeSewing's user units are millimetres; the
    # 2.834646 scale in the transform is 72/25.4, points per millimetre.
    BOX_D = "M62.06 234.6V285.4H163.66V234.6H62.06Z"
    BOX_TRANSFORM = "matrix(2.834646,0,0,2.834646,685.1906,6.932923)"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _box(self) -> Path:
        # PyMuPDF writes PDF points as unitless numbers, so 72 per inch.
        body = f'<g transform="{self.BOX_TRANSFORM}"><path id="cal" d="{self.BOX_D}"/></g>'
        path = self.root / "cal.svg"
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="612" height="792" '
            f'viewBox="0 0 612 792">{body}</svg>'
        )
        return path

    def test_calibration_box_measures_exactly_four_by_two_inches(self):
        piece = svg_to_pieces(self._box(), min_area=0.05,
                              units_per_inch_override=72.0)[0]
        xs = [p[0] for p in piece.boundary_points]
        ys = [p[1] for p in piece.boundary_points]
        self.assertAlmostEqual(max(xs) - min(xs), 4.0, places=4)
        self.assertAlmostEqual(max(ys) - min(ys), 2.0, places=4)

    def test_freesewing_user_units_are_millimetres(self):
        # The same box read as raw user units must be 101.6 x 50.8 mm.
        subs = parse_path(self.BOX_D)
        xs = [p[0] for p in subs[0]["points"]]
        ys = [p[1] for p in subs[0]["points"]]
        self.assertAlmostEqual(max(xs) - min(xs), 101.6, places=4)
        self.assertAlmostEqual(max(ys) - min(ys), 50.8, places=4)

    def test_vertical_and_horizontal_shorthand_in_real_data(self):
        # The export uses V/H shorthand with no separators before the digits.
        self.assertTrue(parse_path(self.BOX_D)[0]["closed"])
        self.assertAlmostEqual(polygon_area(parse_path(self.BOX_D)[0]["points"]),
                               101.6 * 50.8, places=3)


class TestRealAaronSheet(TestCase):
    """
    Path data lifted verbatim from an untiled FreeSewing Aaron v4.10.1 sheet.

    The untiled export puts the whole pattern on one page with no clipping, so
    these are the real pieces at their real size. FreeSewing's user units are
    millimetres and PyMuPDF writes PDF points unitless, hence the 2.834646
    (72/25.4) scale in each transform and the 72-per-inch override.
    """

    FRONT_D = ("M0 90V495H225.72C225.72 323.2 213.74 196.41 213.74 196.41 "
               "136.33 196.41 92.52 108.71 126.11 21.02L96.19 9.55C65.37 90 "
               "65.37 90 0 90Z")
    BINDING_D = "M0 0V466.95H60V0H0Z"
    CAL_INNER_D = "M62.86 235V285H162.86V235H62.86Z"
    SCALE = "matrix(2.834646,0,0,2.834646,0,0)"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _measure(self, d: str):
        body = f'<g transform="{self.SCALE}"><path id="p" d="{d}"/></g>'
        path = self.root / "aaron.svg"
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1665.1842" '
            'height="1387.4054" viewBox="0 0 1665.1842 1387.4054">'
            f"{body}</svg>"
        )
        piece = svg_to_pieces(path, min_area=0.05, units_per_inch_override=72.0)[0]
        xs = [q[0] for q in piece.boundary_points]
        ys = [q[1] for q in piece.boundary_points]
        return max(xs) - min(xs), max(ys) - min(ys)

    def test_front_piece_is_its_real_size(self):
        width, height = self._measure(self.FRONT_D)
        # 225.72mm x 485.45mm, an A-shirt front cut on the fold.
        self.assertAlmostEqual(width * 25.4, 225.72, places=2)
        self.assertAlmostEqual(height * 25.4, 485.45, places=2)

    def test_front_piece_curve_is_followed(self):
        # The cubic in the armhole must contribute samples, not a straight
        # chord: a 7-command path would otherwise yield a handful of points.
        body = f'<g transform="{self.SCALE}"><path id="p" d="{self.FRONT_D}"/></g>'
        path = self.root / "aaron.svg"
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1665.1842" '
            f'height="1387.4054" viewBox="0 0 1665.1842 1387.4054">{body}</svg>'
        )
        piece = svg_to_pieces(path, min_area=0.05, units_per_inch_override=72.0)[0]
        self.assertGreater(len(piece.boundary_points), 20)

    def test_binding_is_exactly_sixty_millimetres_wide(self):
        width, height = self._measure(self.BINDING_D)
        self.assertAlmostEqual(width * 25.4, 60.0, places=3)
        self.assertAlmostEqual(height * 25.4, 466.95, places=2)

    def test_inner_calibration_box_is_exactly_ten_by_five_centimetres(self):
        # FreeSewing prints "the (white) inside of this box should measure
        # 10cm x 5cm" on the sheet, which makes it self-declared ground truth.
        width, height = self._measure(self.CAL_INNER_D)
        self.assertAlmostEqual(width * 25.4, 100.0, places=3)
        self.assertAlmostEqual(height * 25.4, 50.0, places=3)


class TestImplicitClosure(TestCase):
    """Outlines that return to their start without writing Z."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_path_returning_to_start_counts_as_closed(self):
        body = '<path id="front" d="M 0,0 L 960,0 L 960,960 L 0,960 L 0,0"/>'
        self.assertEqual(len(svg_to_pieces(_write(self.root, body))), 1)

    def test_genuinely_open_path_is_still_skipped(self):
        body = '<path id="grain" d="M 0,0 L 960,0 L 960,960"/>'
        self.assertEqual(svg_to_pieces(_write(self.root, body)), [])


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
