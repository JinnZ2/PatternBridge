# Licensing of the data directories

The **MIT licence in `LICENSE` covers the code in this repository.** It does
not, and cannot, cover material this project did not create.

## `data/` — pattern piece images

Each image is a crop from a third-party sewing pattern, and **each one keeps
its own publisher's terms**. Those terms are recorded per pattern in
[`data/PROVENANCE.md`](data/PROVENANCE.md), and per image in the sidecar
`.json` beside it:

```json
{
  "source_name": "butterick",
  "license": "(c)2008 Butterick, The McCall Pattern Company - free download",
  "attribution": "Butterick, The McCall Pattern Company"
}
```

Nothing in `data/` is public domain. It is here as machine-learning training
data under the reasoning in `data/PROVENANCE.md`; anything whose terms forbid
sharing, or whose origin could not be established, is deliberately **not**
committed. If you intend to redistribute an image from `data/`, check that
image's own licence first — this repository's MIT licence does not grant it.

## `data_geometry/` — imported pattern geometry

Derived from [Garment-Pattern-Generator][gpg] templates, MIT licensed,
© 2021 Maria Korosteleva. Redistributable under MIT with that attribution.

## Why the split

A repository licence can only give away rights the author holds. Applying one
blanket licence to a tree containing other people's patterns would be a
promise this project has no standing to make — so the code is MIT, and the
pattern data carries its own provenance, one file at a time.

[gpg]: https://github.com/maria-korosteleva/Garment-Pattern-Generator
