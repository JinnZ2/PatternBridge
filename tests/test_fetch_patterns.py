"""Tests for the open-source pattern fetcher."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, MagicMock

from PIL import Image

from tools.fetch_patterns import (
    PatternSource,
    FetchResult,
    classify_from_url,
    get_all_sources,
    get_sources_by_name,
    register_source,
    fetch_patterns,
    fetch_url,
    _url_hash,
    _is_valid_image,
    _save_image,
    _SOURCE_REGISTRY,
)


def _make_image_bytes(fmt: str = "PNG", size: tuple = (100, 100)) -> bytes:
    """Create minimal valid image bytes."""
    img = Image.new("RGB", size, color="blue")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_source(**overrides) -> PatternSource:
    """Create a test PatternSource with defaults."""
    defaults = dict(
        url="https://example.com/pattern.png",
        garment_type="pants",
        piece_name="front",
        source_name="test",
        license="MIT",
        attribution="Test Author",
    )
    defaults.update(overrides)
    return PatternSource(**defaults)


# ── URL classification ──────────────────────────────────────────────────────


class TestClassifyFromUrl(TestCase):

    def test_pants_front(self):
        g, p = classify_from_url("https://example.com/pants-front-pattern.png")
        assert g == "pants"
        assert p == "front"

    def test_skirt_back(self):
        g, p = classify_from_url("https://example.com/SKIRT_back_v2.jpg")
        assert g == "skirt"
        assert p == "back"

    def test_sleeve_detection(self):
        g, p = classify_from_url("https://example.com/jacket/sleeve-piece.png")
        assert g == "jacket"
        assert p == "sleeve"

    def test_hat_collar(self):
        g, p = classify_from_url("https://example.com/hat_collar.png")
        assert g == "hat"
        assert p == "collar"

    def test_unknown_url(self):
        g, p = classify_from_url("https://example.com/random_image_12345.png")
        assert g == "other"
        assert p == "other"

    def test_dress_no_piece(self):
        g, p = classify_from_url("https://example.com/dress-pattern.png")
        assert g == "dress"
        assert p == "other"

    def test_trouser_synonym(self):
        g, p = classify_from_url("https://example.com/trouser_front.png")
        assert g == "pants"
        assert p == "front"

    def test_blouse_is_top(self):
        g, p = classify_from_url("https://example.com/blouse-back.png")
        assert g == "top"
        assert p == "back"


# ── Source registry ─────────────────────────────────────────────────────────


class TestSourceRegistry(TestCase):

    def test_get_all_sources_returns_list(self):
        sources = get_all_sources()
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_all_sources_have_valid_types(self):
        from pattern_vision.classifier import GARMENT_TYPES, PIECE_NAMES
        for s in get_all_sources():
            assert s.garment_type in GARMENT_TYPES, f"{s.url}: bad garment '{s.garment_type}'"
            assert s.piece_name in PIECE_NAMES, f"{s.url}: bad piece '{s.piece_name}'"

    def test_all_sources_have_license(self):
        for s in get_all_sources():
            assert s.license, f"{s.url}: missing license"

    def test_get_sources_by_name(self):
        freesewing = get_sources_by_name("freesewing")
        assert len(freesewing) > 0
        assert all(s.source_name == "freesewing" for s in freesewing)

    def test_get_sources_unknown_name(self):
        result = get_sources_by_name("nonexistent")
        assert result == []

    def test_register_custom_source(self):
        custom = [_make_source(source_name="custom_test")]
        register_source("custom_test", custom)
        assert "custom_test" in _SOURCE_REGISTRY
        result = get_sources_by_name("custom_test")
        assert len(result) == 1
        # Cleanup
        del _SOURCE_REGISTRY["custom_test"]


# ── Helpers ─────────────────────────────────────────────────────────────────


class TestHelpers(TestCase):

    def test_url_hash_deterministic(self):
        h1 = _url_hash("https://example.com/test.png")
        h2 = _url_hash("https://example.com/test.png")
        assert h1 == h2
        assert len(h1) == 12

    def test_url_hash_different_urls(self):
        h1 = _url_hash("https://example.com/a.png")
        h2 = _url_hash("https://example.com/b.png")
        assert h1 != h2

    def test_is_valid_image_good(self):
        assert _is_valid_image(_make_image_bytes()) is True

    def test_is_valid_image_bad(self):
        assert _is_valid_image(b"not an image at all") is False

    def test_is_valid_image_empty(self):
        assert _is_valid_image(b"") is False


# ── Save image ──────────────────────────────────────────────────────────────


class TestSaveImage(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_creates_file(self):
        source = _make_source()
        data = _make_image_bytes()
        path = _save_image(data, source, self.tmpdir, 0)
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_correct_directory(self):
        source = _make_source(garment_type="dress", piece_name="back")
        data = _make_image_bytes()
        path = _save_image(data, source, self.tmpdir, 0)
        assert "dress" in str(path)
        assert "back" in str(path)

    def test_save_creates_annotation(self):
        source = _make_source(has_fold_line=True, notch_count=3)
        data = _make_image_bytes()
        path = _save_image(data, source, self.tmpdir, 0)
        ann_path = path.parent / f"{path.name}.json"
        assert ann_path.exists()
        ann = json.loads(ann_path.read_text())
        assert ann["has_fold_line"] is True
        assert ann["notch_count"] == 3
        assert ann["source_url"] == source.url
        assert ann["license"] == "MIT"

    def test_save_dedup_skips_existing(self):
        source = _make_source()
        data = _make_image_bytes()
        path1 = _save_image(data, source, self.tmpdir, 0)
        path2 = _save_image(data, source, self.tmpdir, 0)
        assert path1 is not None
        assert path2 is None  # duplicate, skipped

    def test_save_jpg_extension(self):
        source = _make_source(url="https://example.com/pattern.jpg")
        data = _make_image_bytes("JPEG")
        path = _save_image(data, source, self.tmpdir, 0)
        assert path.suffix == ".jpg"


# ── Fetch patterns (mocked) ────────────────────────────────────────────────


class TestFetchPatterns(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dry_run_downloads_nothing(self):
        result = fetch_patterns(data_dir=self.tmpdir, dry_run=True, limit=5)
        assert result.downloaded > 0
        # No actual files should exist
        image_files = list(self.tmpdir.rglob("IMG_*"))
        assert len(image_files) == 0

    @patch("tools.fetch_patterns.requests")
    def test_fetch_saves_images(self, mock_requests):
        img_bytes = _make_image_bytes()
        mock_resp = MagicMock()
        mock_resp.content = img_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        # Register a simple test source
        test_sources = [_make_source(url="https://example.com/test_pattern.png")]
        register_source("_test_fetch", test_sources)

        try:
            result = fetch_patterns(
                data_dir=self.tmpdir, source_name="_test_fetch", limit=5
            )
            assert result.downloaded == 1
            assert result.failed == 0
            images = list(self.tmpdir.rglob("IMG_*"))
            # Filter out .json sidecar files
            images = [f for f in images if not f.name.endswith(".json")]
            assert len(images) == 1
        finally:
            del _SOURCE_REGISTRY["_test_fetch"]

    @patch("tools.fetch_patterns.requests")
    def test_fetch_handles_http_error(self, mock_requests):
        mock_requests.get.side_effect = Exception("Connection refused")

        test_sources = [_make_source()]
        register_source("_test_err", test_sources)

        try:
            result = fetch_patterns(
                data_dir=self.tmpdir, source_name="_test_err", limit=5
            )
            assert result.failed == 1
            assert len(result.errors) == 1
        finally:
            del _SOURCE_REGISTRY["_test_err"]

    @patch("tools.fetch_patterns.requests")
    def test_fetch_rejects_invalid_image(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.content = b"not an image"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        test_sources = [_make_source()]
        register_source("_test_bad", test_sources)

        try:
            result = fetch_patterns(
                data_dir=self.tmpdir, source_name="_test_bad", limit=5
            )
            assert result.failed == 1
        finally:
            del _SOURCE_REGISTRY["_test_bad"]

    def test_fetch_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            fetch_patterns(data_dir=self.tmpdir, source_name="nonexistent_source")

    @patch("tools.fetch_patterns.requests")
    def test_fetch_skips_invalid_garment_type(self, mock_requests):
        test_sources = [_make_source(garment_type="spaceship")]
        register_source("_test_skip", test_sources)

        try:
            result = fetch_patterns(
                data_dir=self.tmpdir, source_name="_test_skip", limit=5
            )
            assert result.skipped == 1
            assert result.downloaded == 0
        finally:
            del _SOURCE_REGISTRY["_test_skip"]


# ── Fetch single URL (mocked) ──────────────────────────────────────────────


class TestFetchUrl(TestCase):

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("tools.fetch_patterns.requests")
    def test_fetch_url_auto_classifies(self, mock_requests):
        img_bytes = _make_image_bytes()
        mock_resp = MagicMock()
        mock_resp.content = img_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        path = fetch_url(
            "https://example.com/pants_front_pattern.png",
            data_dir=self.tmpdir,
        )
        assert path is not None
        assert "pants" in str(path)
        assert "front" in str(path)

    @patch("tools.fetch_patterns.requests")
    def test_fetch_url_explicit_classification(self, mock_requests):
        img_bytes = _make_image_bytes()
        mock_resp = MagicMock()
        mock_resp.content = img_bytes
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        path = fetch_url(
            "https://example.com/random.png",
            data_dir=self.tmpdir,
            garment_type="hat",
            piece_name="other",
            license="CC0",
        )
        assert path is not None
        assert "hat" in str(path)

    @patch("tools.fetch_patterns.requests")
    def test_fetch_url_failure_returns_none(self, mock_requests):
        mock_requests.get.side_effect = Exception("timeout")
        path = fetch_url("https://example.com/bad.png", data_dir=self.tmpdir)
        assert path is None
