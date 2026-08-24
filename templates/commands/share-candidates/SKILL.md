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
   the `share-bundle/` path, the `share-candidates.zip` path, how many candidates
   got evolve bundles and how many got one-shot fixes (if any), and any skipped
   candidates.

## What the bundle contains

- `index.html` — ranked cards, one per candidate, linking to rendered pages.
- `candidates/modules/<module>/<file>.html` — rendered candidate page.
- `candidates/modules/<module>/<file>.md` — the original markdown, beside each page.
- The `modules/` folder structure is preserved for the exported candidates.

## Evolve bundles (automatic when present)

If `prep-evolve` has run, its output lives in a sibling `evolve/` tree beside
`sorted/` (`<run>/evolve/<module>/<candidate>/<engine>/`). The build folds it in
automatically — no flag needed. If there is no `evolve/` tree, the build behaves
exactly as above.

For each exported candidate that has evolve data, the bundle also gets:

- `candidates/modules/<module>/<file>__evolve.html` — an "Evolve bundles" page
  with one section per engine (`coral`, `skydiscover`, `nous`): what the framework
  is, its repo, install command(s), edit scope, native config, the run command,
  the inlined README, collapsible raw views of the other files, and a per-engine
  download button.
- `candidates/modules/<module>/<file>__evolve/<engine>/…` — the complete original
  evolve files copied verbatim (including `seed.py`), plus a per-engine
  `<engine>.zip` that unzips into a tidy top-level `<engine>/` folder.
- The candidate page gains an "Evolve bundles" section, and its index card gains
  an `evolve · N` badge and a footer link to the evolve page.

> **Evaluation gap:** every evolve bundle is launchable as-is, but produces no
> meaningful score until the recipient closes its gap, which ships as a `TODO` —
> the performance measurement in the evaluator/grader for `skydiscover` and
> `coral`, and `ground_truth.pass_condition` in `campaign.yaml` for `nous`. Each
> engine's inlined README states this and the oracle Spotlights inferred.

## One-shot fix bundles (automatic when present)

If `spotlights-engine fix` (or `/spotlights-fix-candidate`) has run, its output
lives in a sibling `fix/` tree beside `sorted/`
(`<run>/fix/<module>/<candidate>/`). The build folds it in automatically — no
flag needed. If there is no `fix/` tree, the build behaves exactly as above.

For each exported candidate that has a `fix.patch`, the bundle also gets:

- `candidates/modules/<module>/<file>__fix.html` — a "One-shot fix" page: what a
  one-shot fix is, the unverified warning, an **Apply this patch** strip (real
  base commit, a `<YOUR_REPO_CHECKOUT>` placeholder, the `-3` and `patch -p1`
  fallbacks), the inlined `FIX-NOTES.md`, and the patch as a collapsible
  colorized diff.
- `candidates/modules/<module>/<file>__fix/` — `fix.patch` and `FIX-NOTES.md`
  copied verbatim, plus a `fix.zip` that unzips into a tidy top-level `fix/`
  folder.
- The candidate page gains a "One-shot fix" section, and its index card gains a
  `fix · N file(s) changed, +X/−Y` badge and a footer link to the fix page.

A candidate directory holding only `FIX-NOTES.md` and no patch — the legitimate
"the change could not be made" outcome — is skipped entirely: no page, no badge.

> **Nothing in a fix bundle was verified:** no test was run, no benchmark was
> measured, no build was attempted. The fix page states this prominently, and
> carries the candidate's recorded oracles as the verification recipe for
> whoever has the hardware.

## Link handling (built into the script)

- External URLs (arxiv, doi, docs) → clickable, open in a new tab.
- Module breadcrumbs (`../*.md`) and source `.py` paths → plain non-clickable text.

## Notes

- The script is stdlib-only Python 3 and idempotent (re-running overwrites output).
- Output is written under the source folder you provide.
