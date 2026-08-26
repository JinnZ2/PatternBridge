# Releasing

Publishing is set up so that **every release after the first can be cut from a
phone.** There are no secrets stored in this repository and no API token to
paste. PyPI is configured to trust builds coming from this repo's `publish`
workflow, and nothing else.

---

## One-time setup

This part needs a real keyboard. It is done once, ever.

1. Create a PyPI account at <https://pypi.org/account/register/> and turn on
   two-factor authentication (PyPI requires it to publish).

2. Go to <https://pypi.org/manage/account/publishing/> and add a **pending
   publisher**. "Pending" means the project does not exist on PyPI yet — this
   is what lets the very first release be published by the workflow rather than
   by hand. Fill in exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `patternbridge` |
   | Owner | `JinnZ2` |
   | Repository name | `PatternBridge` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   All five must match or PyPI will refuse the upload. The environment name is
   the one declared in `.github/workflows/publish.yml`.

3. That is the whole setup. Nothing needs to be added to GitHub — no secret, no
   token, no settings change.

### Why no token

The workflow authenticates with PyPI using OpenID Connect: GitHub signs a
short-lived token proving "this build came from `JinnZ2/PatternBridge`, from
`publish.yml`, in the `pypi` environment", and PyPI checks it against what you
registered above. A leaked token is not a risk because there is no token. This
is PyPI's recommended way to publish, and it is what removes the keyboard from
every subsequent release.

---

## Cutting a release

All of this works in a phone browser.

1. **Bump the version** in `pyproject.toml` — this is the one text edit, and
   GitHub's web editor handles it. `0.1.0` → `0.1.1` for fixes, `0.2.0` for new
   capability. Commit to `main`.

2. Go to the repo → **Releases** → **Draft a new release**.

3. **Choose a tag** → type a new one: `v0.1.0`. It must match the version in
   `pyproject.toml`, with an optional leading `v`. The workflow checks this and
   fails the release rather than shipping a mislabelled version, because **a
   PyPI version number can never be reused.**

4. Title it the same as the tag. Write whatever notes you like — or press
   **Generate release notes** and let GitHub write them from the merged PRs.

5. **Publish release.**

The `publish` workflow then runs the full test suite, verifies the tag matches
the packaged version, builds the wheel and sdist, checks the metadata, and
uploads to PyPI. Watch it under the **Actions** tab. If anything fails, nothing
is published.

## Rehearsing without publishing

Actions → **publish** → **Run workflow**. A manual run does the tests, the
build and the metadata check, then stops — the upload step only runs for a
real published release. Use it to confirm a release would succeed before
committing to a version number.

## If something goes wrong

- **The tag check failed.** The tag and `pyproject.toml` disagree. Delete the
  release and the tag, fix one of them, and draft it again. Nothing was
  published, so nothing is lost.
- **PyPI rejected the upload.** Almost always one of the five pending-publisher
  fields not matching. Compare them against the table above.
- **A release went out with a bug.** Do not try to re-upload the same version;
  PyPI will refuse. Bump the patch version and release again. You can `yank` the
  bad version on PyPI, which hides it from new installs without breaking anyone
  who already has it.

## What ships, and what does not

The distribution contains **code only** — no pattern images and no PDFs. This
is deliberate and worth keeping that way: `data/` holds images under their
publishers' own terms, and republishing them through a package index would
distribute them far more widely than committing them ever did. See
`data/PROVENANCE.md`.

`data_geometry/` — the 124 MIT-licensed pattern pieces — is also not shipped
today. It could be, since its licence plainly allows it, but packaging it means
moving it inside a Python package and updating every path that refers to it.
That is churn better done deliberately than as part of a first release. For now
those pieces are a clone away.
