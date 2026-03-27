"""
Mobile-friendly web server for capturing pattern piece photos.

Run on your local network, open the URL on your phone, and snap photos
of sewing pattern pieces. Images are saved directly into the directory
structure expected by PatternDataset:

    data/<garment_type>/<piece_name>/IMG_<timestamp>.jpg

Usage:
    python tools/capture_server.py                     # default: data/ on port 8000
    python tools/capture_server.py --data-dir my_data  # custom output directory
    python tools/capture_server.py --port 9000         # custom port
    python tools/capture_server.py --host 0.0.0.0      # listen on all interfaces (LAN)

Then open http://<your-ip>:8000 on your phone.

Requires: flask, Pillow
"""

from __future__ import annotations

import argparse
import io
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

try:
    from PIL import Image
except ImportError:
    Image = None

# Import label lists from the classifier module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pattern_vision.classifier import GARMENT_TYPES, PIECE_NAMES

# ── App ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)
DATA_DIR = Path("data")

# Annotation defaults for newly captured images
DEFAULT_ANNOTATIONS = {
    "has_fold_line": False,
    "has_grain_line": True,
    "notch_count": 0,
    "dart_count": 0,
}


def _ensure_dir(garment_type: str, piece_name: str) -> Path:
    """Create and return the target directory for a capture."""
    dest = DATA_DIR / garment_type / piece_name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _count_images(garment_type: str | None = None, piece_name: str | None = None) -> int:
    """Count captured images, optionally filtered by type/piece."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    count = 0
    if garment_type and piece_name:
        d = DATA_DIR / garment_type / piece_name
        if d.exists():
            count = sum(1 for f in d.iterdir() if f.suffix.lower() in extensions)
    elif garment_type:
        d = DATA_DIR / garment_type
        if d.exists():
            count = sum(
                1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in extensions
            )
    else:
        if DATA_DIR.exists():
            count = sum(
                1 for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in extensions
            )
    return count


# ── HTML ────────────────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>PatternBridge Capture</title>
<style>
  :root { --accent: #2563eb; --bg: #f8fafc; --card: #fff; --text: #1e293b; --muted: #64748b; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, system-ui, sans-serif; background: var(--bg); color: var(--text);
         padding: 16px; max-width: 480px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 20px; }
  .card { background: var(--card); border-radius: 12px; padding: 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px; }
  label { display: block; font-weight: 600; font-size: 0.9rem; margin-bottom: 6px; }
  select, input[type="file"] { width: 100%; padding: 12px; border: 2px solid #e2e8f0;
          border-radius: 8px; font-size: 1rem; margin-bottom: 16px; appearance: none;
          background: var(--bg); }
  select:focus { border-color: var(--accent); outline: none; }
  .capture-btn { display: block; width: 100%; padding: 16px; font-size: 1.1rem;
                 font-weight: 700; color: #fff; background: var(--accent); border: none;
                 border-radius: 12px; cursor: pointer; transition: transform 0.1s; }
  .capture-btn:active { transform: scale(0.97); }
  .capture-btn:disabled { background: #94a3b8; }
  .status { text-align: center; padding: 12px; border-radius: 8px; margin-top: 12px;
            font-weight: 600; font-size: 0.9rem; display: none; }
  .status.success { display: block; background: #dcfce7; color: #166534; }
  .status.error { display: block; background: #fee2e2; color: #991b1b; }
  .status.uploading { display: block; background: #dbeafe; color: #1e40af; }
  .counter { text-align: center; color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
  .preview { max-width: 100%; border-radius: 8px; margin-bottom: 12px; display: none; }
  .anno-row { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
  .anno-row label { flex: 1; margin: 0; font-weight: 500; font-size: 0.85rem; }
  .anno-row input[type="number"] { width: 60px; padding: 8px; border: 2px solid #e2e8f0;
          border-radius: 8px; font-size: 1rem; text-align: center; }
  .toggle { position: relative; width: 48px; height: 28px; flex-shrink: 0; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle .slider { position: absolute; inset: 0; background: #cbd5e1; border-radius: 14px;
                    cursor: pointer; transition: 0.2s; }
  .toggle .slider:before { content: ""; position: absolute; width: 22px; height: 22px;
                           left: 3px; bottom: 3px; background: #fff; border-radius: 50%;
                           transition: 0.2s; }
  .toggle input:checked + .slider { background: var(--accent); }
  .toggle input:checked + .slider:before { transform: translateX(20px); }
  .section-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 10px; }
  .history { max-height: 200px; overflow-y: auto; }
  .history-item { display: flex; justify-content: space-between; padding: 8px 0;
                  border-bottom: 1px solid #f1f5f9; font-size: 0.85rem; }
  .history-item .type { color: var(--accent); font-weight: 600; }
</style>
</head>
<body>

<h1>PatternBridge Capture</h1>
<p class="subtitle">Snap pattern pieces for classifier training</p>

<div class="card">
  <label for="garment">Garment Type</label>
  <select id="garment">
    GARMENT_OPTIONS
  </select>

  <label for="piece">Piece Name</label>
  <select id="piece">
    PIECE_OPTIONS
  </select>
</div>

<div class="card">
  <label>Annotations (optional)</label>
  <div class="anno-row">
    <label>Fold line</label>
    <div class="toggle"><input type="checkbox" id="foldLine"><span class="slider"></span></div>
  </div>
  <div class="anno-row">
    <label>Grain line</label>
    <div class="toggle"><input type="checkbox" id="grainLine" checked><span class="slider"></span></div>
  </div>
  <div class="anno-row">
    <label>Notches</label>
    <input type="number" id="notchCount" value="0" min="0" max="10">
  </div>
  <div class="anno-row">
    <label>Darts</label>
    <input type="number" id="dartCount" value="0" min="0" max="6">
  </div>
</div>

<div class="card">
  <img id="preview" class="preview" alt="Preview">
  <input type="file" id="photo" accept="image/*" capture="environment">
  <button class="capture-btn" id="uploadBtn" disabled>Upload Photo</button>
  <div id="status" class="status"></div>
  <div id="counter" class="counter"></div>
</div>

<div class="card">
  <div class="section-label">Recent captures</div>
  <div id="history" class="history">
    <div style="color: var(--muted); font-size: 0.85rem;">No captures yet</div>
  </div>
</div>

<script>
const photoInput = document.getElementById('photo');
const uploadBtn = document.getElementById('uploadBtn');
const preview = document.getElementById('preview');
const status = document.getElementById('status');
const counter = document.getElementById('counter');
const history = document.getElementById('history');
let captures = [];

photoInput.addEventListener('change', () => {
  const file = photoInput.files[0];
  if (file) {
    uploadBtn.disabled = false;
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
  }
});

uploadBtn.addEventListener('click', async () => {
  const file = photoInput.files[0];
  if (!file) return;

  const garment = document.getElementById('garment').value;
  const piece = document.getElementById('piece').value;

  const formData = new FormData();
  formData.append('photo', file);
  formData.append('garment_type', garment);
  formData.append('piece_name', piece);
  formData.append('has_fold_line', document.getElementById('foldLine').checked);
  formData.append('has_grain_line', document.getElementById('grainLine').checked);
  formData.append('notch_count', document.getElementById('notchCount').value);
  formData.append('dart_count', document.getElementById('dartCount').value);

  uploadBtn.disabled = true;
  status.className = 'status uploading';
  status.textContent = 'Uploading...';

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.ok) {
      status.className = 'status success';
      status.textContent = '\\u2713 Saved: ' + data.filename;
      counter.textContent = data.total_images + ' total images captured';
      captures.unshift({ garment, piece, filename: data.filename, time: new Date().toLocaleTimeString() });
      renderHistory();
      photoInput.value = '';
      preview.style.display = 'none';
    } else {
      status.className = 'status error';
      status.textContent = '\\u2717 ' + (data.error || 'Upload failed');
      uploadBtn.disabled = false;
    }
  } catch (e) {
    status.className = 'status error';
    status.textContent = '\\u2717 Network error';
    uploadBtn.disabled = false;
  }
});

function renderHistory() {
  if (captures.length === 0) return;
  history.innerHTML = captures.slice(0, 20).map(c =>
    '<div class="history-item"><span class="type">' + c.garment + '/' + c.piece +
    '</span><span>' + c.time + '</span></div>'
  ).join('');
}

// Load initial count
fetch('/stats').then(r => r.json()).then(data => {
  counter.textContent = data.total_images + ' total images captured';
});
</script>
</body>
</html>"""


def _build_index() -> str:
    """Build the HTML page with garment/piece options injected."""
    garment_opts = "\n    ".join(
        f'<option value="{g}">{g.title()}</option>' for g in GARMENT_TYPES
    )
    piece_opts = "\n    ".join(
        f'<option value="{p}">{p.title()}</option>' for p in PIECE_NAMES
    )
    html = INDEX_HTML.replace("GARMENT_OPTIONS", garment_opts)
    html = html.replace("PIECE_OPTIONS", piece_opts)
    return html


# ── Routes ──────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return _build_index(), 200, {"Content-Type": "text/html"}


@app.route("/upload", methods=["POST"])
def upload():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"ok": False, "error": "No photo provided"}), 400

    garment_type = request.form.get("garment_type", "").lower()
    piece_name = request.form.get("piece_name", "").lower()

    if garment_type not in GARMENT_TYPES:
        return jsonify({"ok": False, "error": f"Unknown garment type: {garment_type}"}), 400
    if piece_name not in PIECE_NAMES:
        return jsonify({"ok": False, "error": f"Unknown piece name: {piece_name}"}), 400

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    dest_dir = _ensure_dir(garment_type, piece_name)

    # Read and validate image
    try:
        img_bytes = photo.read()
        if Image:
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()
    except Exception:
        return jsonify({"ok": False, "error": "Invalid image file"}), 400

    # Determine extension from original filename
    original_ext = Path(photo.filename).suffix.lower() if photo.filename else ".jpg"
    if original_ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        original_ext = ".jpg"

    filename = f"IMG_{timestamp}{original_ext}"
    filepath = dest_dir / filename
    filepath.write_bytes(img_bytes)

    # Save annotation alongside image
    ann = {
        "has_fold_line": request.form.get("has_fold_line", "false").lower() == "true",
        "has_grain_line": request.form.get("has_grain_line", "true").lower() == "true",
        "notch_count": int(request.form.get("notch_count", 0)),
        "dart_count": int(request.form.get("dart_count", 0)),
    }
    ann_path = dest_dir / f"{filename}.json"
    import json
    ann_path.write_text(json.dumps(ann, indent=2))

    total = _count_images()
    print(f"  Saved: {filepath}  ({total} total)")

    return jsonify({
        "ok": True,
        "filename": filename,
        "path": str(filepath),
        "garment_type": garment_type,
        "piece_name": piece_name,
        "total_images": total,
    })


@app.route("/stats")
def stats():
    """Return image counts for the dashboard."""
    breakdown = {}
    for g in GARMENT_TYPES:
        for p in PIECE_NAMES:
            c = _count_images(g, p)
            if c > 0:
                breakdown[f"{g}/{p}"] = c

    return jsonify({
        "total_images": _count_images(),
        "breakdown": breakdown,
    })


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_local_ip() -> str:
    """Get the local network IP address for display."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Mobile capture server for pattern piece photos"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Output directory for captured images (default: data/)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for LAN access)",
    )
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    local_ip = _get_local_ip()
    print(f"\n  PatternBridge Capture Server")
    print(f"  ────────────────────────────")
    print(f"  Local:   http://127.0.0.1:{args.port}")
    print(f"  Network: http://{local_ip}:{args.port}")
    print(f"  Data:    {DATA_DIR.resolve()}")
    print(f"\n  Open the Network URL on your phone to start capturing.\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
