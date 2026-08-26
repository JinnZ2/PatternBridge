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
| [FreeSewing](https://freesewing.org) | Web app + open-source codebase that drafts made-to-measure patterns from your measurements — menswear, womenswear, accessories, blocks/slopers | MIT (code); CC-BY (patterns/content) | as claimed |
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

**Not yet verified against a real FreeSewing export.** The importer is
round-trip tested against this project's own SVG output and against fixtures
covering the whole path grammar, but no actual FreeSewing file has been through
it — the environment it was written in cannot reach freesewing.org. When you
run the first one, check two things: that the reported piece dimensions match
the size FreeSewing states, and that piece names come from the group ids you
expect. If the scale is out, pass `units_per_inch_override`; if names come back
generic, add the offending id to `GENERIC_LABELS`.

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
