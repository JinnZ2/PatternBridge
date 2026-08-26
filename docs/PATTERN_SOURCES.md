# Where to find sewing patterns

A working list of pattern sources, ordered by how freely the patterns can
actually be used. Licensing is the constraint that decides whether a pattern
can go in `data/` — see `data/PROVENANCE.md` for the rule this repo applies.

**Status column** means:

- **tested** — a PDF from this source has been through `--check` in this repo,
  and the terms below are what its own text says
- **as claimed** — taken from the source's description, not verified here
- **disputed** — the claim looks wrong; see the note

Before adding anything from any of these, run it through the checker:

```bash
python tools/extract_pdf_patterns.py --check ~/Downloads
```

---

## 1. Open source / parametric — the only truly hackable tier

These are the ones worth reaching for first: explicit licenses, no ambiguity,
and machine-readable output rather than a flattened PDF.

| Resource | What it is | License | Status |
|---|---|---|---|
| [FreeSewing](https://freesewing.org) | Web app + open-source codebase that drafts made-to-measure patterns from your measurements — menswear, womenswear, accessories, blocks/slopers | MIT (code); CC-BY (patterns/content) | **tested — SVG import verified** |
| [GitHub `sewing-patterns` topic](https://github.com/topics/sewing-patterns) | Pattern generators, PDF tilers, SVG linters, foundation-paper-piecing tools | Varies — mostly MIT / GPL / CC, check each repo | as claimed |
| [Garment Pattern Generator](https://github.com/maria-korosteleva/Garment-Pattern-Generator) | Research tool generating 3D garment datasets with sewing patterns | MIT, © 2021 Maria Korosteleva | **tested — imported** |

**Garment Pattern Generator is already imported.** Its 23 templates carry
exact panel geometry — vertices plus an edge loop with Bézier curvature — which
`tools/import_garment_patterns.py` converts into 124 PatternPiece files under
`data_geometry/`. MIT licensed, no images, no ambiguity. It is the best-licensed
pattern data in the project.

FreeSewing is the strongest fit for PatternBridge specifically. It emits
**SVG**, not a scanned raster, so its geometry goes straight into
`pattern_geometry/` with no vision layer and no tile-assembly guesswork:

```bash
python -m tools.import_svg_patterns aaron.svg --list
```

The importer is built against FreeSewing's **actual** output, read from
`packages/core/src/svg.mjs` in their source rather than guessed:

| What FreeSewing emits | How the importer reads it |
|---|---|
| `width`/`height` in mm with a matching `viewBox` | one user unit = 1 mm, so sizes land in true inches |
| groups named `fs-stack-<id>-part-<design>.<name>` | piece name taken from after the last `-part-`, design namespace dropped |
| a configurable `idPrefix` (default `fs-`) | prefix-agnostic — any prefix resolves |
| every path auto-numbered `fs-1`, `fs-2`, … | ignored; the enclosing group names the piece |
| `embed: true`, which drops `width`/`height` | still read as mm, not CSS pixels |
| helper lines drawn as open paths | skipped — only closed outlines become pieces |

**Validated against a real export.** An Aaron v4.10.1 was exported both tiled
and untiled, converted to SVG and read back. The untiled sheet puts the whole
pattern on one page with no clipping, so every piece comes out at its real
size. FreeSewing prints a calibration box and states its true size on the
sheet, which makes it self-declared ground truth for path parsing, transform
composition and unit scaling at once:

| Shape | FreeSewing says | Importer read |
|---|---|---|
| calibration box, outer | 4in x 2in | **101.6 x 50.8 mm** (4.000 x 2.000 in) |
| calibration box, inner | 10cm x 5cm | **100.0 x 50.0 mm** |
| neck binding width | 60 mm | **60.0 mm** |
| arm binding width | 60 mm | **60.0 mm** |
| front / back | — | 225.7 x 485.5 mm, correct for the stated 830 mm chest |

All four Aaron pieces came back, at true size, from a single command.

Two things that only a real file exposed, both now fixed and pinned by tests:
a PDF-derived SVG hides a page-sized rectangle in `<clipPath>` and every glyph
outline in `<defs>`, and the importer was reading those as pattern pieces; and
FreeSewing writes `V`/`H` shorthand with no separator before the digits.

**Validated against FreeSewing's own SVG, not just its PDF.** The table above
was originally read from FreeSewing's source. It has since been checked against
the renderer itself: `@freesewing/aaron@4.10.1` and `@freesewing/core` were
installed from npm and drafted directly, so the file below is what FreeSewing
writes, with no PDF round-trip in between:

```bash
python -m tools.import_svg_patterns aaron-native.svg --list
```

| Shape | True size | Importer read |
|---|---|---|
| calibration box, outer | 4in x 2in | **101.60 x 50.80 mm** |
| calibration box, inner | 10cm x 5cm | **100.00 x 50.00 mm** |
| neck binding width | 60 mm | **60.00 mm** |
| arm binding width | 60 mm | **60.00 mm** |

Rendering with `embed: true`, which drops `width` and `height` entirely, gives
byte-for-byte the same measurements — so reading an embedded sheet as
millimetres rather than CSS pixels is now confirmed against the real output
rather than inferred from source.

The real file exposed one thing the fixture could not. FreeSewing namespaces
every part by its design, so the group is
`fs-stack-aaron.back-part-aaron.back`, and pieces were coming out named
`AARON.BACK` rather than `BACK`. The design is already known from the file, so
the namespace is now dropped.

Two caveats remain. The printed **calibration box** is two closed rectangles
sitting inside a part's group, so it imports as two extra pieces named after
that part — on the Aaron above, `BACK` at 4 x 2 in and `BACK` at 3.94 x 1.97
in. The sheet **logo** behaves the same way on PDF-derived files. Nothing in
the geometry distinguishes either from a small pattern piece; raise
`--min-area` or drop them by size if it matters.

## 2. Free PDF downloads — usually personal-use

Free to download and sew. Most restrict redistribution, so expect these to
land in `data_local/` rather than `data/`.

| Resource | Highlights | Status |
|---|---|---|
| [Mood Sewciety](https://blog.moodfabrics.com/free-sewing-patterns/) | Large library — dresses, swimwear, lingerie, blazers, coats, menswear, costumes | **tested** — the Lotus Legging and hood PDFs carry no stated terms |
| [So Sew Easy](https://so-sew-easy.com/free-sewing-patterns/) | Skirts, dresses, shorts, rompers, sports bras, bags | as claimed |
| [SewCanShe](https://www.sewcanshe.com/free-patterns) | Bags, pouches, quilts, home decor, baby items, apparel | as claimed |
| [BERNINA Blog](https://blog.bernina.com/en/category/free-sewing-patterns/) | A4 PDFs — apparel, bags, baby items, patchwork; no login | as claimed |
| [Peppermint Sewing School](https://peppermintmag.com/sewing-school/) | Sustainable, beginner-friendly designs | **tested** — the Playsuit PDF states *"FOR PERSONAL USE ONLY"* |
| [Fabrics-Store](https://blog.fabrics-store.com/free-sewing-patterns/) | Linen and natural-fibre projects | as claimed |
| [ThreadsMonthly](https://threadsmonthly.com) | Tutorials with printable pattern pieces | **tested** — the headband PDF states no terms |
| [French Navy](https://frenchnavy.co.za) | South African indie label; sells patterns and gives some away | **tested** — the Orla dress PDF states no terms |
| Bombazine | Fabric shop project patterns | **tested** — the oven mitt PDF states no terms |
| Sew Mag (UK) / Love Sewing Mag | Regular free PDF drops from UK sewing magazines | as claimed |

## 3. Vintage and older catalogues

| Resource | Notes | Status |
|---|---|---|
| [Sewing.org](https://www.sewing.org) | Large free PDF library, some vintage | as claimed |
| [BurdaStyle](https://www.burdastyle.com) | "Pattern of the Day" plus rotating free vintage-style PDFs | as claimed |
| [r/freepatterns](https://www.reddit.com/r/freepatterns/) | Community-curated links, including vintage collections | as claimed |

**Do not read this section as "public domain."** Free-to-download and
public-domain are different things, and vintage *styling* says nothing about
copyright. In the US a published work enters the public domain 95 years after
publication — as of 2026 that means **published before 1931**. Everything in
the sections above is far newer than that, including patterns that reproduce a
1950s design: the Butterick Retro Wrap in `data/` is a 1952 style but a
**2008** document, and 2008 is the date that counts.

## 4. Commercial-use claims

| Resource | Claim | Status |
|---|---|---|
| Etsy — "royalty free sewing patterns" | Filter by Creative Commons Attribution to find patterns you can adapt and sell from | **disputed** |
| Paid course patterns (Domestika, Skillshare, etc.) | Bought, so usable | **disputed** |

Etsy has no Creative Commons license filter in its search, and listings are
individually copyrighted by each seller — a shop's own blurb is the only
licence statement, and it varies per listing. Treat Etsy as **not** a reliable
source of commercially-reusable patterns. If a specific seller grants those
rights in writing, that grant is between you and them; record it in the
`license` field of the registry entry.

This is the one entry on the original list that could cause real trouble if
acted on as written, which is why it is called out rather than quietly dropped.

**Paid course patterns** are a second trap, and a quieter one. They frequently
print no terms at all, so `--check` reports them `NO TERMS` — but a purchase
licenses you to *use* the pattern, not to republish it. Buying something is not
the same as being licensed to redistribute it, and no text scan can tell the
difference: provenance is exactly what a text scan cannot see.

Note the inverse trap too. Domestika publishes *free* patterns on its blog
alongside its paid courses — the Lantoki worker jacket in `data/` is one — so
"came from a course platform" is not itself a reason to reject a pattern.
Check where the specific file was published.

---

## Quick start

1. **For open, hackable patterns** → [FreeSewing](https://freesewing.org).
   Enter measurements, pick a design (Simon shirt, Tamiko zero-waste top,
   Bruce boxer briefs), export SVG or a tiled PDF.
2. **For a large free catalogue right now** →
   [Mood Sewciety](https://blog.moodfabrics.com/free-sewing-patterns/) or
   [So Sew Easy](https://so-sew-easy.com/free-sewing-patterns/).
3. **Whatever the source** → run `--check` on it before it goes near `data/`.
4. **If you cannot place a file** → extract it with `--data-dir data_local`.
   It still trains the classifier; it just is not published. That is the right
   default for anything downloaded long enough ago that its origin is gone.

## Adding a source to the fetcher

`tools/fetch_patterns.py` downloads directly from openly-licensed sources. To
add one, append a `PatternSource` to its registry with the garment type, piece
name, license, and attribution; the license string ends up in every sidecar
JSON the fetch writes.

Only add sources whose license is **explicit**. A source that is merely free
belongs in section 2 above, downloaded by hand and screened with `--check`.
