---
name: share-candidates
description: Use when sharing top-ranked Spotlights candidates with an external team as a browsable offline bundle — "share candidates", "make a zip of the top candidates", "browsable candidates". Takes a folder containing sorted_candidates.md and produces share-bundle/ + share-candidates.zip. Depends on sort-candidates: if no sorted_candidates.md exists yet, run sort-candidates first to produce it.
---

# Share Candidates

Packages the top-N candidates from a `sorted_candidates.md` ranking into a
self-contained ZIP an external team can unzip and open by double-clicking
`index.html` — no server, works offline (except outbound reference links).

## Prerequisites

This skill consumes `sorted_candidates.md`, which is produced by the
**sort-candidates** skill (`spotlights-sort-candidates`). If the source folder
does not already contain `sorted_candidates.md`, run sort-candidates first to
rank a run's `result.json` into that file, then return here. This is a soft
(documented) dependency — Claude Code does not enforce skill ordering, so it is
your responsibility to run the ranking step first.

## Procedure

1. **Ask for the source folder** — the folder that directly contains
   `sorted_candidates.md` (e.g. `.../spotlights-out/sorted/`). Do not
   guess; ask the user for the absolute path.
2. **Ask for top-N** — default is 5 if the user doesn't specify.
3. **Run the build script** from the repo root:

   ```bash
   python3 .claude/commands/spotlights-share-candidates/build_bundle.py --source "<folder>" --top-n 5
   ```

4. **Report** the printed summary: bundle title, number of candidates exported,
   the `share-bundle/` path, the `share-candidates.zip` path, and any skipped
   candidates.

## What the bundle contains

- `index.html` — ranked cards, one per candidate, linking to rendered pages.
- `candidates/modules/<module>/<file>.html` — rendered candidate page.
- `candidates/modules/<module>/<file>.md` — the original markdown, beside each page.
- The `modules/` folder structure is preserved for the exported candidates.

## Link handling (built into the script)

- External URLs (arxiv, doi, docs) → clickable, open in a new tab.
- Module breadcrumbs (`../*.md`) and source `.py` paths → plain non-clickable text.

## Notes

- The script is stdlib-only Python 3 and idempotent (re-running overwrites output).
- Output is written under the source folder you provide.
