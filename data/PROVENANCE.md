# Training data provenance

Every image under `data/` is a pattern piece extracted from a source PDF or
downloaded from an open-licensed source. Each image has a sidecar `.json` with
its annotation labels plus `source_name`, `license`, and `attribution`.

Images are produced by:

- `tools/extract_pdf_patterns.py` — crops pieces out of pattern PDFs
- `tools/fetch_patterns.py` — downloads from open-licensed sources
- `tools/capture_server.py` — phone captures of physical patterns

## Copyright status

**None of the commercial pattern PDFs in this project are public domain.**
US works published by a corporation are protected for 95 years from
publication, so the oldest one here (2007) is covered into the 2100s. A free
download is not a license to redistribute.

The copyright line printed in each source PDF:

| Pattern | Notice in the PDF | In this repo |
|---|---|---|
| Butterick Retro Wrap | `©2008 Butterick, The McCall Pattern Company` | yes — free promotional download |
| McCall's Cosmetic Bag | `©2007 The McCall Pattern Company, All rights reserved.` | yes — free promotional download |
| Great Outdoors Fishing Vest | `© Cranston Print Works` | yes — free promotional download |
| Kwik Sew 5001 Clutch Purse | `©MMV KWIK•SEW® Pattern Co., Inc. All rights reserved.` — *"Commercial or industrial use prohibited"* | yes — bars commercial use, not redistribution |
| Fleece Socks Tutorial | none — no notice, no author, no URL | yes — no stated terms |
| Amelia Coat (Stitch Winter 2012) | `© Interweave Press LLC` — *"Not to be reprinted. All rights reserved. Please respect the copyright by not forwarding or distributing this document."* | **no** |
| Luxury Fur Coat Pattern | `Copyright 2015 Stefanie Knaus` — *"for personal use only"* | **no** |
| SoZo Undies | `© Zoe Edwards 2021` — *"licensed for individual private home use only … Pattern may not be shared, sold or re-distributed without owner's prior written consent."* | **no** |
| Tilly and the Buttons Slipper Boots | `© Tilly and the Buttons` — *"For personal use only"*, and the PDF sets permission flags denying content copying | **no** |
| OleDeMa sock pattern | `OleDeMa` — *"Only for personal using"* | **no** |

The line drawn here: a pattern is held back when its own text forbids sharing
or restricts use to the person who downloaded it. A generic "all rights
reserved" on a free promotional download is not that — those are committed,
matching what the repo already carried.

Two edge cases worth naming. **Kwik Sew 5001** reserves all rights but the only
use it actually prohibits is commercial or industrial, so redistribution is not
barred. **Fleece Socks** carries no notice at all — which is not a grant, since
copyright is automatic, but there is no restriction to honour either; treat it
as unresolved rather than free, and re-check before relying on it.

The five restricted patterns are **not committed**. `extract_pdf_patterns.py`
marks them `redistributable=False` and skips them unless `--include-restricted`
is passed. Extract them to an untracked directory instead:

```bash
python tools/extract_pdf_patterns.py --pdf-dir ~/patterns \
    --include-restricted --data-dir data_local
```

`data_local/` is in `.gitignore`. `PatternDataset` accepts any root, so
training can read both trees locally without either one being published:

```python
PatternDataset("data")        # committed, shareable
PatternDataset("data_local")  # local only
```

## Why the geometry is safer than the image

In US law a garment design is a *useful article* — its cut and shape are not
copyrightable. What the copyright covers is the PDF itself: the drawings, the
artwork, the instructions, the layout. `Star Athletica v. Varsity Brands`
(2017) protects only pictorial features separable from the garment's utility,
not the pattern's silhouette.

So the derived numbers — boundary coordinates, notch positions, grain line
angles, grade rules — sit on much firmer ground than a reproduction of the
printed page. That output is what PatternBridge is built to produce:

```python
piece.to_json()      # boundary_points, notches, darts, tokens
```

If a restricted pattern needs to live in the repo, prefer committing the
encoded geometry over the page image.

This is engineering guidance for organizing the dataset, not legal advice.
