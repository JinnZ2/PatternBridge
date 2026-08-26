# Training data provenance

Every image under `data/` is a pattern piece extracted from a source PDF or
downloaded from an open-licensed source. Each image has a sidecar `.json` with
its annotation labels plus `source_name`, `license`, and `attribution`.

`data_geometry/` is a separate tree holding *geometry* rather than images —
see "`data_geometry/` — this argument, applied" below.

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
| Lantoki "Chaqueta Worker Unisex" | none — free download from the Domestika blog | yes — no stated terms |
| Patterns by Mood hood | none — scanned, but carries the Mood roundel | yes — no stated terms |
| Bombazine oven mitt | none | **no** — provenance unknown |
| French Navy "The Orla Dress" | none — names frenchnavy.co.za | **no** — provenance unknown |
| ThreadsMonthly wide headband | none — names ThreadsMonthly.com | **no** — provenance unknown |
| Jo-Ann Christmas tree napkins | `©2011 Jo-Ann Stores, Inc.`, sheet marked *"free"* | yes — but holds no pattern pieces |
| Bernina Men's Travel Set | `©2001 Bernina of America, Inc.` | yes — but holds no pattern pieces |
| p2designs Quick & Easy Fleece Beanie | `© 2006 Patti Pierce Stone` — *"charity or personal use only … may not copy the contents"* | **no** |
| FleeceFun Rocker trapper hat | same FleeceFun terms | **no** |
| The Crafty Kitty Bear & Fox welly liners | `Copyright 2014 The Crafty Kitty` — *"For Personal Use only"* | **no** |
| So Sew Easy fur boots | `Copyright 2017 So Sew Easy Pte Ltd` — *"do not copy, publish, sell, redistribute"*; **watermarked with the buyer's email** | **no** |
| S5474 sleeveless sundress (m-sewing.com CAD) | none — but S5474 is a Simplicity commercial pattern number | **no** — *removed*, see below |
| UAF CCM-00072 "The Cloth Parka" | none in the PDF — a free University of Alaska Fairbanks Cooperative Extension publication, with the USDA | **no** — origin known, no licence; see below |
| DRCOS Ladies' Coat | none — names dr-cos.info | **no** — provenance unknown |

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

The Lantoki worker jacket was briefly held back here on the assumption that it
was paid Domestika course material. It is not — it is a free download from the
Domestika blog (published 16 January 2024 by Mónica Martín Rivas), which puts
it in the same tier as the other no-stated-terms patterns above. The general
caution still holds and is worth keeping in mind: **provenance is invisible to
`--check`.** A scan reads what a document *says*, not where it came from, so a
paid pattern that prints no terms will read as `NO TERMS`, not as blocked.
See "When you cannot remember where a file came from" below.

## Three reasons a pattern is held back

`extract_pdf_patterns.py` records **why** each held entry is held, in
`hold_reason`:

| `hold_reason` | Meaning | Shown by `--list` as |
|---|---|---|
| `terms` | the PDF itself forbids sharing | `RESTRICTED` |
| `unknown-provenance` | it forbids nothing, but nobody can say where it came from | `LOCAL ONLY` |
| `no-grant` | the publisher **is** known and the download was free — but no licence covers this document | `LOCAL ONLY` |

The second is the ordinary case for a pattern downloaded years ago: free and
paid look identical on disk, and `--check` can only read what a document says.
Holding those costs nothing — they still train the classifier from
`data_local/` — so unknown provenance defaults to local rather than to a guess.

The third is narrower and worth separating out, because it is the one most
likely to be resolved later. Nothing about it is doubtful: you know who
published it, you know it was free, you can see that use is invited. The only
missing thing is a licence, and a licence is what grants redistribution.
Filing these as `unknown-provenance` would misreport them — it implies the file
might have been bought, which is a different and worse claim about a free
public document.

Watch for a specific trap here: **a licence on a neighbouring publication does
not travel to an unlicensed one.** A publisher may release one project under
Creative Commons and say nothing about the next. "Their other thing is CC, so
this probably is too" is inference, not permission, and inference is exactly
what this registry exists to keep out of the record.

All three are skipped unless `--include-restricted` is passed. Extract them to
an untracked directory instead:

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

### `data_geometry/` — this argument, applied

`data_geometry/` holds 124 pattern pieces imported from
Garment-Pattern-Generator (MIT, © 2021 Maria Korosteleva) by
`tools/import_garment_patterns.py`. Every piece is exact boundary geometry read
from an openly-licensed template — no page image, no vision inference, and no
licensing ambiguity.

This is the tier the section above argues for, and it is the only pattern
source in the project with no caveat attached: the licence is explicit, the
geometry is exact, and pieces feed straight into encoding, scaling, and
output.

## Removed from the repo

### S5474 sleeveless sundress — 5 images

`data/dress/*/sundress_s5474_*.png` were crops of a CAD sheet from
m-sewing.com, committed in `f3a596b` with no sidecar. The sheet states no terms
at all, so this is a weaker case than the one below — but `S5474` is a
**Simplicity commercial pattern number**, and the source file is now registered
`unknown-provenance`. Publishing crops of a file the registry holds back was
inconsistent, so they have been removed. Restore with
`git checkout f3a596b -- data/dress/` if you disagree; the evidence here is
inferred from the pattern number, not stated by the document.

### make Bra sock pattern — 2 images

`data/other/other/sock_bra_pieces_p1.png` and `sock_bra_pieces_p3.png` were
committed before this provenance policy existed, with no sidecar recording
where they came from. They are pages 1 and 3 of the **make Bra** sock pattern,
which states *"This pattern is for your personal use only, any commercial use
is prohibited."* Under the line above they should never have been published, so
they have been deleted. `git log` still has them; restore with
`git checkout f3a596b -- data/other/other/` if that call is wrong.

## A case examined and still held: the UAF cloth parka

Worth writing down, because it is the closest call in the registry and it shows
what the standard actually costs.

UAF CCM-00072, *The Cloth Parka*, is a free University of Alaska Fairbanks
Cooperative Extension publication. Its origin was checked and is not in doubt:
it was never bought, it is offered to the public as an educational resource,
and personal sewing use is plainly intended. That is more than most entries
here can say.

It is still held, for one reason: **no licence covers this document.** A
sibling UAF resource, *Patterns and Parkas*, is CC BY-NC 4.0 — which would
permit sharing. But that licence sits on that project, and a licence does not
travel to a neighbouring publication because the same institution published
both. Treating "their other thing is CC" as permission is precisely the
inference this file exists to keep out of the record.

Two things argue for patience rather than a technicality:

- The garment is a **traditional Alaska Native design**, published as an
  educational resource. That is a claim about how it should be reused which
  copyright does not measure, and which a licence scan cannot see.
- Even a clean CC BY-NC would sit awkwardly here. This repo's code is MIT and
  anyone may build commercially on it; a non-commercial image inside a public
  training set is a foreseeable trap for someone downstream who never reads
  this file.

So the parka is linked, not republished — `docs/PATTERN_SOURCES.md` points at
UAF's own page. That costs nothing and sends people to the publisher, which is
what UAF would want anyway. If UAF states redistribution terms for CCM-00072
itself, this becomes a one-line change.

**A note on what the images are.** Unlike every tiled entry here, this is not a
print-and-tape sheet. The pattern is a reduced scale drawing on a 4-inch grid —
a reader redraws it full size. The extracted crops therefore carry correct
piece *shape* but no real-world scale. They are training data for the
classifier, which learns outlines; they are not boundaries to hand to the
geometry layer expecting inches.

## Where to find more patterns

`docs/PATTERN_SOURCES.md` lists open-source, free, and vintage pattern
sources with their licensing, and flags the ones whose stated terms do not
hold up. Start there rather than hunting.

## Personalised copies

A watermark is a **personal** mailbox on a domain the document never mentions.
Two things are deliberately not treated as watermarks, because calling them
one would tell someone a free document was something they paid for:

- a **role mailbox** — `info@`, `support@`, `orders@`, `program.intake@`
- an address whose domain appears anywhere in the document as a website,
  subdomains included, so `ocio.usda.gov` in a link vouches for `@usda.gov`

The second case is not hypothetical: every federally funded US publication
carries the USDA non-discrimination statement, and an early version of this
check flagged a free University of Alaska extension booklet as a purchase
because of it.


Some paid patterns stamp every page with the buyer's email address and
purchase date. `--check` reports those as `PERSONALISED`, which outranks every
other verdict, because such a mark settles two things at once:

- the file was **bought** — the one question a licence scan cannot otherwise
  answer, and the one memory cannot answer years later
- it carries **personal data**, so those images must never be published even
  if the terms were permissive

An email address only counts as a watermark when its domain appears nowhere
else in the document as a site reference. A publisher printing its own contact
address beside its own URL is not a watermark; an address whose domain the
document never mentions is.

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
| `NO TERMS` | nothing in the text restricts sharing |
| `DO NOT USE` | a restriction was found; the matching sentence is quoted |
| `CHECK BY EYE` | scanned PDF with no searchable text — terms may be in the artwork |
| `ALREADY HAVE` | matches a registry entry by content hash, even if renamed |

Each result also prints a **"says of itself"** line — title, author, publisher,
URL, pattern number, date. That is not licensing; it is a search handle for a
file whose origin you no longer remember. A PDF saved years ago under a
meaningless filename will usually still name its publisher:

```
NO TERMS      some-old-download.pdf
              nothing restricts sharing in 6 pages
              says of itself: title: 5001.purse | author: Aimee |
                              www.kwiksew.com | ©MMV KWIK•SEW® Pattern Co., Inc.
              -> not a licence — keep local unless you know it was free
```

`CHECK BY EYE` exists because a text scan cannot read a restriction printed
inside a scanned image. The OleDeMa sock pattern is exactly that case: its
"Only for personal using" line is part of the artwork, so reporting "no terms"
there would have been a false all-clear.

The scan deliberately does **not** treat a bare "all rights reserved" as a
block — free promotional patterns print it alongside permission to download,
and blocking on it would reject most usable patterns.

### When you cannot remember where a file came from

`NO TERMS` is not `USABLE`, which is why it is not called that. A pattern
someone paid for frequently prints no terms at all, so the scan cannot
distinguish a free download from a purchase — and if the download happened
years ago, neither can memory.

The default when provenance is unrecoverable is simple and costs nothing:

```bash
python tools/extract_pdf_patterns.py --pdf-dir ~/patterns --data-dir data_local
```

`data_local/` is gitignored. `PatternDataset` reads either tree, so the pieces
still train the classifier; they just are not published. Move a pattern into
`data/` only when something positively identifies it as free — a "free
download" page, an open licence, a publisher's free-pattern section.

Nothing is lost by defaulting to local. The published dataset already has
enough openly-licensed material to stand on its own, and `data_geometry/`
adds 124 pieces with no licensing question at all.

## Sources with nothing to extract

`socks_pattern_sewing_instructions.pdf` is the assembly sheet that accompanies
the OleDeMa sock pattern — construction steps only, no pattern pieces, so
there is nothing to crop out of it.

The Tessuti Monroe Turtleneck PDF is likewise an instruction booklet with no
pattern pieces. It *is* in the registry, with no pieces mapped, so that
`--check` recognises the file instead of re-deriving the same verdict.

This is engineering guidance for organizing the dataset, not legal advice.
