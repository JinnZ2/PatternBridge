"""Tests for pattern_output/data_export.py — JSON export, manifests, pattern sets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from pattern_output.data_export import (
    piece_to_dict,
    piece_to_json,
    save_json,
    build_manifest,
    save_manifest,
    save_pattern_set,
)


# ── piece_to_dict ────────────────────────────────────────────────────────────


class TestPieceToDict:
    def test_has_computed_properties(self, pants_front):
        d = piece_to_dict(pants_front)
        assert "computed" in d
        computed = d["computed"]
        assert computed["is_encodable"] is True
        assert computed["boundary_point_count"] == 4
        assert computed["has_darts"] is True

    def test_includes_tokens_by_default(self, pants_front):
        d = piece_to_dict(pants_front)
        assert "encoded_tokens" in d

    def test_excludes_tokens_when_asked(self, pants_front):
        d = piece_to_dict(pants_front, include_tokens=False)
        assert "encoded_tokens" not in d

    def test_includes_name_and_garment(self, pants_front):
        d = piece_to_dict(pants_front)
        assert d["name"] == "FRONT"
        assert d["garment_type"] == "pants"


# ── piece_to_json ────────────────────────────────────────────────────────────


class TestPieceToJson:
    def test_returns_valid_json(self, pants_front):
        j = piece_to_json(pants_front)
        parsed = json.loads(j)
        assert parsed["name"] == "FRONT"
        assert "computed" in parsed

    def test_indent_option(self, pants_front):
        j = piece_to_json(pants_front, indent=4)
        # 4-space indent means lines start with "    "
        assert "    " in j


# ── save_json ────────────────────────────────────────────────────────────────


class TestSaveJson:
    def test_writes_file(self, pants_front):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json(pants_front, os.path.join(tmpdir, "piece.json"))
            assert path.exists()
            parsed = json.loads(path.read_text())
            assert parsed["name"] == "FRONT"

    def test_creates_parent_dir(self, pants_front):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_json(
                pants_front,
                os.path.join(tmpdir, "sub", "deep", "piece.json"),
            )
            assert path.exists()


# ── build_manifest ───────────────────────────────────────────────────────────


class TestBuildManifest:
    def test_basic_manifest(self, multi_piece_set):
        manifest = build_manifest(multi_piece_set, pattern_name="Test")
        assert manifest["pattern_name"] == "Test"
        assert manifest["piece_count"] == 3
        assert len(manifest["pieces"]) == 3

    def test_total_cut_count(self, pants_front):
        # Default cut_quantity = 1
        manifest = build_manifest([pants_front])
        assert manifest["total_cut_count"] >= 1

    def test_garment_types_collected(self, multi_piece_set):
        manifest = build_manifest(multi_piece_set)
        assert isinstance(manifest["garment_types"], list)

    def test_notes_field(self, pants_front):
        manifest = build_manifest([pants_front], notes="test note")
        assert manifest["notes"] == "test note"


# ── save_manifest ────────────────────────────────────────────────────────────


class TestSaveManifest:
    def test_writes_manifest_file(self, multi_piece_set):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_manifest(
                multi_piece_set,
                os.path.join(tmpdir, "manifest.json"),
                pattern_name="Demo",
            )
            assert path.exists()
            parsed = json.loads(path.read_text())
            assert parsed["pattern_name"] == "Demo"


# ── save_pattern_set ─────────────────────────────────────────────────────────


class TestSavePatternSet:
    def test_writes_all_files(self, multi_piece_set):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = save_pattern_set(multi_piece_set, tmpdir, pattern_name="Full")
            assert "manifest.json" in written
            # One JSON per piece + manifest
            assert len(written) == len(multi_piece_set) + 1
            for path in written.values():
                assert path.exists()

    def test_piece_files_are_valid_json(self, multi_piece_set):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = save_pattern_set(multi_piece_set, tmpdir)
            for name, path in written.items():
                parsed = json.loads(path.read_text())
                assert isinstance(parsed, dict)
