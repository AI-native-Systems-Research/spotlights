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
3. **Run the build script.** `build_bundle.py` is installed into this skill's own
   directory, alongside the `SKILL.md` you are reading. Where that is depends on
   how `spotlights-engine init` was run, so locate it with the snippet below
   rather than assuming a path — it takes the first location that exists:

   ```bash
   script=
   for d in ${CLAUDE_CONFIG_DIR+"${CLAUDE_CONFIG_DIR:-.}/commands"} \
            "$HOME/.claude/commands" \
            ".claude/commands"; do
     candidate="$d/spotlights-share-candidates/build_bundle.py"
     if [ -f "$candidate" ]; then script="$candidate"; break; fi
   done
   if [ -z "$script" ]; then
     echo "build_bundle.py not found — run 'spotlights-engine init' first" >&2
     exit 1
   fi
   python3 "$script" --source "<folder>" --top-n 5
   ```

   `$CLAUDE_CONFIG_DIR/commands/` is where a user-scope install lands whenever
   that variable is *set* — including when it is set to the empty string, which
   Claude Code and `init` both read as a cwd-relative `commands/`. Hence `+` and
   not `:+` for the test, and the `:-.` default inside it. `~/.claude/commands/`
   is the `init` default, and `./.claude/commands/` is an `init --scope project`
   install. If none of them has the script, say so rather than guessing — do not
   report the last path tried as the fault.

   The script resolves all of its paths from `--source`, so it does not matter
   which directory you run it from.

4. **Report** the printed summary: bundle title, number of candidates exported,
   the `share-bundle/` path, the `share-candidates.zip` path, how many candidates
   got evolve bundles and how many got one-shot patches (if any), and any
   skipped candidates.

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

## One-shot apply bundles (automatic when present)

If `spotlights-engine apply` (or `/spotlights-apply-candidate`) has run, its
output lives in a sibling `apply/` tree beside `sorted/`
(`<run>/apply/<module>/<candidate>/`). The build folds it in automatically — no
flag needed. If there is no `apply/` tree, the build behaves exactly as above.

For each exported candidate that has an `apply.patch`, the bundle also gets:

- `candidates/modules/<module>/<file>__apply.html` — a "One-shot apply" page:
  what a one-shot apply is, the unverified warning, an **Apply this patch**
  strip (real base commit, a checkout placeholder derived from the producer's
  repo name — `<YOUR_VLLM_CHECKOUT>`, falling back to `<YOUR_REPO_CHECKOUT>` —
  and the `-3` and `patch -p1` fallbacks), the inlined `APPLY-NOTES.md`, and the
  patch as a collapsible colorized diff.
- `candidates/modules/<module>/<file>__apply/` — `apply.patch`, `APPLY-NOTES.md`
  and `apply.prompt.txt` copied verbatim, plus an `apply.zip` carrying the same
  files, unzipping into a tidy top-level `apply/` folder. Only the patch is
  guaranteed present; the notes and the prompt each ship when the `apply/` tree
  has them, and the page names the prompt in the apply strip when it does.
- The candidate page gains a "One-shot apply" section, and its index card gains
  an `apply · +X/−Y` badge and a footer link carrying the full
  `N file(s) changed, +X/−Y` stat.

A candidate directory holding only `APPLY-NOTES.md` and no patch — the
legitimate "the change could not be made" outcome — is skipped entirely: no
page, no badge.

> **Nothing in an apply bundle was verified:** no test was run, no benchmark
> was measured, no build was attempted. The apply page states this prominently,
> and carries the candidate's recorded oracles as the verification recipe for
> whoever has the hardware.

## Link handling (built into the script)

- External URLs (arxiv, doi, docs) → clickable, open in a new tab. All three
  forms are covered: `<https://…>` autolinks, `[text](https://…)` links, and
  bare `https://…` URLs in prose. Only `http(s)` is linkified — not bare
  `www.`, not email addresses, not scheme-relative URLs.
- Trailing sentence punctuation is not part of the link: `See https://arxiv.org/abs/2309.06180.`
  links the URL and leaves the full stop outside it. Same for `,` `;` `:` and a
  closing paren the URL did not open.
- A URL inside a code span or a fenced code block stays literal text, and so
  does every URL inside a rendered patch.
- Module breadcrumbs (`../*.md`) and source `.py` paths → plain non-clickable text.

Which pages this matters for: the sorted ranking writes its references as
`<…>` autolinks, so candidate pages' reference links have always been
clickable. `APPLY-NOTES.md` writes them bare, so the bare-URL rule is what
makes an apply page's references clickable (roughly four per page).

## Notes

- The script is stdlib-only Python 3 and idempotent (re-running overwrites output).
- Output is written under the source folder you provide.
