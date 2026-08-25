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
| Mood Fabrics MDF039 Lotus Legging | none | yes — no stated terms |
| Zune's Sewing Therapy Kids Pants | none | yes — no stated terms |
| Amelia Coat (Stitch Winter 2012) | `© Interweave Press LLC` — *"Not to be reprinted. All rights reserved. Please respect the copyright by not forwarding or distributing this document."* | **no** |
| Luxury Fur Coat Pattern | `Copyright 2015 Stefanie Knaus` — *"for personal use only"* | **no** |
| SoZo Undies | `© Zoe Edwards 2021` — *"licensed for individual private home use only … Pattern may not be shared, sold or re-distributed without owner's prior written consent."* | **no** |
| Tilly and the Buttons Slipper Boots | `© Tilly and the Buttons` — *"For personal use only"*, and the PDF sets permission flags denying content copying | **no** |
| OleDeMa sock pattern | `OleDeMa` — *"Only for personal using"* | **no** |
| FleeceFun Basic Fleece Hat | *"You may not re-post the patterns or tutorials to the web or email them another person"*; pages watermarked | **no** |
| FleeceFun child's pleated skirt | same FleeceFun terms | **no** |
| make Bra sock pattern | `Copyright © Annele Salonen Tmi` — *"for your personal use only, any commercial use is prohibited"* | **no** — *removed*, see below |
| Tessuti Monroe Turtleneck | `©Tessuti Fabrics 2018` — *"Our patterns are for personal use only"* | **no** |
| Peppermint / In the Folds Playsuit | *"FOR PERSONAL USE ONLY"* | **no** |

The line drawn here: a pattern is held back when its own text forbids sharing
or restricts use to the person who downloaded it. A generic "all rights
reserved" on a free promotional download is not that — those are committed,
matching what the repo already carried.

Two edge cases worth naming. **Kwik Sew 5001** reserves all rights but the only
use it actually prohibits is commercial or industrial, so redistribution is not
barred.

And three patterns — the **Fleece Socks tutorial**, **Mood Fabrics' Lotus
Legging** and **Zune's Kids Pants** — carry no notice at all. That is not a
grant, since copyright is automatic without one, but there is no stated
restriction to honour either. Treat them as unresolved rather than free, and
re-check before relying on them.

The ten restricted patterns are **not committed**. `extract_pdf_patterns.py`
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

## Removed from the repo

`data/other/other/sock_bra_pieces_p1.png` and `sock_bra_pieces_p3.png` were
committed before this provenance policy existed, with no sidecar recording
where they came from. They are pages 1 and 3 of the **make Bra** sock pattern,
which states *"This pattern is for your personal use only, any commercial use
is prohibited."* Under the line above they should never have been published, so
they have been deleted. `git log` still has them; restore with
`git checkout f3a596b -- data/other/other/` if that call is wrong.

## Where to find more patterns

`docs/PATTERN_SOURCES.md` lists open-source, free, and vintage pattern
sources with their licensing, and flags the ones whose stated terms do not
hold up. Start there rather than hunting.

## Screening a PDF before you add it

`--check` reads a PDF's text and reports whether it restricts redistribution,
so a pattern can be triaged without opening it:

```bash
python tools/extract_pdf_patterns.py --check ~/Downloads
python tools/extract_pdf_patterns.py --check one-pattern.pdf
```

Four verdicts:

| Verdict | Meaning |
|---|---|
| `USABLE` | nothing in the text restricts sharing |
| `DO NOT USE` | a restriction was found; the matching sentence is quoted |
| `CHECK BY EYE` | scanned PDF with no searchable text — terms may be in the artwork |
| `ALREADY HAVE` | matches a registry entry by content hash, even if renamed |

`CHECK BY EYE` exists because a text scan cannot read a restriction printed
inside a scanned image. The OleDeMa sock pattern is exactly that case: its
"Only for personal using" line is part of the artwork, so reporting "no terms"
there would have been a false all-clear.

The scan deliberately does **not** treat a bare "all rights reserved" as a
block — free promotional patterns print it alongside permission to download,
and blocking on it would reject most usable patterns.

## Sources with nothing to extract

`socks_pattern_sewing_instructions.pdf` is the assembly sheet that accompanies
the OleDeMa sock pattern — construction steps only, no pattern pieces, so
there is nothing to crop out of it.

The Tessuti Monroe Turtleneck PDF is likewise an instruction booklet with no
pattern pieces. It *is* in the registry, with no pieces mapped, so that
`--check` recognises the file instead of re-deriving the same verdict.

This is engineering guidance for organizing the dataset, not legal advice.
