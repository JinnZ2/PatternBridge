"""Tests for the mobile capture server."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest import TestCase

from PIL import Image

from tools.capture_server import app, _build_index, _count_images, DATA_DIR


def _make_test_image(fmt: str = "JPEG") -> bytes:
    """Create a minimal valid image in memory."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestCaptureIndex(TestCase):
    """Test the HTML index page."""

    def setUp(self):
        self.client = app.test_client()

    def test_index_returns_html(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"PatternBridge Capture" in resp.data

    def test_index_contains_garment_options(self):
        resp = self.client.get("/")
        assert b"pants" in resp.data
        assert b"dress" in resp.data
        assert b"skirt" in resp.data

    def test_index_contains_piece_options(self):
        resp = self.client.get("/")
        assert b"front" in resp.data
        assert b"back" in resp.data
        assert b"sleeve" in resp.data

    def test_build_index_replaces_placeholders(self):
        html = _build_index()
        assert "GARMENT_OPTIONS" not in html
        assert "PIECE_OPTIONS" not in html
        assert 'value="pants"' in html
        assert 'value="front"' in html


class TestCaptureUpload(TestCase):
    """Test photo upload endpoint."""

    def setUp(self):
        import tools.capture_server as mod
        self.tmpdir = tempfile.mkdtemp()
        self._orig_data_dir = mod.DATA_DIR
        mod.DATA_DIR = Path(self.tmpdir)
        self.client = app.test_client()

    def tearDown(self):
        import tools.capture_server as mod
        mod.DATA_DIR = self._orig_data_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_upload_valid_image(self):
        img_bytes = _make_test_image()
        resp = self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "test.jpg"),
            "garment_type": "pants",
            "piece_name": "front",
        }, content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["ok"] is True
        assert data["garment_type"] == "pants"
        assert data["piece_name"] == "front"
        assert data["filename"].startswith("IMG_")
        assert data["filename"].endswith(".jpg")
        # Verify file exists on disk
        saved = Path(self.tmpdir) / "pants" / "front" / data["filename"]
        assert saved.exists()
        assert saved.stat().st_size > 0

    def test_upload_saves_annotation(self):
        img_bytes = _make_test_image()
        resp = self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "test.jpg"),
            "garment_type": "dress",
            "piece_name": "back",
            "has_fold_line": "true",
            "has_grain_line": "false",
            "notch_count": "3",
            "dart_count": "2",
        }, content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["ok"] is True
        ann_path = Path(self.tmpdir) / "dress" / "back" / f"{data['filename']}.json"
        assert ann_path.exists()
        ann = json.loads(ann_path.read_text())
        assert ann["has_fold_line"] is True
        assert ann["has_grain_line"] is False
        assert ann["notch_count"] == 3
        assert ann["dart_count"] == 2

    def test_upload_no_photo(self):
        resp = self.client.post("/upload", data={
            "garment_type": "pants",
            "piece_name": "front",
        }, content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["ok"] is False
        assert resp.status_code == 400

    def test_upload_invalid_garment(self):
        img_bytes = _make_test_image()
        resp = self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "test.jpg"),
            "garment_type": "spaceship",
            "piece_name": "front",
        }, content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["ok"] is False
        assert resp.status_code == 400

    def test_upload_invalid_piece(self):
        img_bytes = _make_test_image()
        resp = self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "test.jpg"),
            "garment_type": "pants",
            "piece_name": "spaceship",
        }, content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["ok"] is False
        assert resp.status_code == 400

    def test_upload_png(self):
        img_bytes = _make_test_image("PNG")
        resp = self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "test.png"),
            "garment_type": "hat",
            "piece_name": "other",
        }, content_type="multipart/form-data")
        data = json.loads(resp.data)
        assert data["ok"] is True
        assert data["filename"].endswith(".png")

    def test_upload_increments_count(self):
        img_bytes = _make_test_image()
        resp1 = self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "a.jpg"),
            "garment_type": "pants",
            "piece_name": "front",
        }, content_type="multipart/form-data")
        count1 = json.loads(resp1.data)["total_images"]

        resp2 = self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "b.jpg"),
            "garment_type": "pants",
            "piece_name": "front",
        }, content_type="multipart/form-data")
        count2 = json.loads(resp2.data)["total_images"]
        assert count2 == count1 + 1


class TestCaptureStats(TestCase):
    """Test stats endpoint."""

    def setUp(self):
        import tools.capture_server as mod
        self.tmpdir = tempfile.mkdtemp()
        self._orig_data_dir = mod.DATA_DIR
        mod.DATA_DIR = Path(self.tmpdir)
        self.client = app.test_client()

    def tearDown(self):
        import tools.capture_server as mod
        mod.DATA_DIR = self._orig_data_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stats_empty(self):
        resp = self.client.get("/stats")
        data = json.loads(resp.data)
        assert data["total_images"] == 0
        assert data["breakdown"] == {}

    def test_stats_after_upload(self):
        img_bytes = _make_test_image()
        self.client.post("/upload", data={
            "photo": (io.BytesIO(img_bytes), "test.jpg"),
            "garment_type": "skirt",
            "piece_name": "front",
        }, content_type="multipart/form-data")
        resp = self.client.get("/stats")
        data = json.loads(resp.data)
        assert data["total_images"] == 1
        assert data["breakdown"]["skirt/front"] == 1
