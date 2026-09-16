# Session brief: scoping advisor follow-ups (Phase A)

Self-contained handoff. Assume no prior context.

## Where things stand

**Phase B is finished and out of the way.** The `run_config` manifest fix is
committed (`7e7d0cc`, 3 files, 183 insertions, 0 deletions), pushed, and open as
**PR #100** against `public/main` with `InbarShapira` as reviewer. Branch
`yevgeny/manifest-run-config`, 1 ahead / 0 behind. Nothing left to do there
except await review. Do not reopen it.

**Phase A — the cost advisor — is the work to resume.** It is complete as a first
cut and parked:

- Branch: **`yevgeny/scoping-advisor`**, commit **`89782aa`**, **1 ahead / 13
  behind `public/main`**. It has no upstream tracking branch and has never been
  pushed. Rebase or merge `public/main` before doing anything that depends on
  current engine code.
- 5 files, 1415 insertions:

  | file | role |
  |---|---|
  | `scripts/advisor.py` | the advisor itself — prints token range, config ladder, pricing |
  | `scripts/build_advisor_calibration.py` | builds `artifacts/advisor_calibration.json` from run dirs |
  | `scripts/advisor_backtest.py` | leave-one-repo-out backtest of the band |
  | `scripts/analyze_cost_drivers.py` | token-mix / driver analysis |
  | `artifacts/advisor_calibration.json` | committed calibration (aggregate stats, anonymized labels, tokens not dollars) |

What the advisor deliberately does **not** do: predict absolute cost. That failed
acceptance (leave-one-repo-out median error 47%, worst 345%, against a ~8%
replicate noise floor). It reports a token range, relative multipliers between
configs, and exact model pricing, with every multiplier labelled
`exact / measured / derived / unvalidated` so a reader can tell arithmetic from
evidence. Read the commit message on `89782aa` first — it carries the full
reasoning and the backtest numbers.

## Task 1 — `pick_rate()` must canonicalize, not substring-match

`scripts/advisor.py:67`:

```python
def pick_rate(table: dict[str, dict], wanted: str) -> tuple[str, dict] | None:
    """Find a rate row by loose model-name match (keys are `provider:model`)."""
    if not wanted:
        return None
    w = wanted.lower()
    for key, val in table.items():
        if w in key.lower():
            return key, val
    return None
```

Two defects:

1. **Misses legitimate ids.** Rate keys are bare (`anthropic:claude-opus-5`) but
   real model ids carry a LiteLLM route prefix and sometimes a context tag
   (`aws/claude-opus-5`, `aws/claude-opus-4-8[1m]`). `"aws/claude-opus-5" in
   "anthropic:claude-opus-5"` is false, so a correctly-spelled model silently
   gets **no rate** and drops out of the blended average at
   `scripts/advisor.py:235-240`.
2. **Order-dependent false hits.** A short `wanted` like `"opus"` matches
   whichever `opus` row dict iteration reaches first.

The engine already solved this. Use it rather than reimplementing:

```python
from spotlights_engine.costing.rates import canonical_model_id
```

`canonical_model_id` strips the route prefix (`aws/`, `azure/`, `vertex/`, …) and
the context tag (`[1m]`), and is the function the engine itself uses on both
sides of its own rate lookup. Canonicalize the model half of each table key and
the `wanted` id, then match **exactly**. Verified present and exported at
`src/spotlights_engine/costing/rates.py:195` — re-check, it may have moved.

Note there is also `display_model_id` in the same module (strips the route prefix
but keeps the context tag). That one is for reader-facing output, **not** for rate
lookup — do not use it here.

## Task 2 — re-run the advisor against the current rate tables

Once Task 1 lands, re-run the advisor and the backtest; the blended rates will
change wherever a model was previously missing its row.

## Task 3 — the rate tables are thinner than the advisor assumes

`src/spotlights_engine/costing/rates.json` currently prices **only four keys**
(verified 2026-09-02):

```
anthropic:claude-opus-4-8
anthropic:claude-opus-5
openai:codex
openai:gpt-5.5
```

Everything else on the LiteLLM proxy is **unpriced**, including every cheap model
worth using for calibration runs — `aws/claude-haiku-4-5`, `aws/claude-sonnet-4-5`,
`aws/claude-opus-4-7`, the `azure/gpt-5.*` family, the Gemini and Azure GPT rows.
An unpriced model does not error: it lands in `cost.coverage.unpriced_models`,
contributes $0, and drags `priced_token_share` below 1.0, so `amount_usd` becomes a
**partial** figure. Any advisor number derived from a run on a cheap model is
therefore a token figure only, never a dollar figure.

Decide explicitly whether to add rate rows or to keep the advisor reporting only
the four priced models. Do not invent rates — the engine's stated contract is that
it never substitutes list price for a missing contracted row.

## Task 4 — optional paid A/B run (needs explicit user go-ahead; costs money)

The only `unvalidated` row on the config ladder is the review-iteration
multiplier. Filling it means a deliberate A/B: same repo, `--review-iterations 1`
vs `3`, everything else fixed. **Do not launch without the user saying so** — they
have been consistently cost-sensitive all through this effort.

If it is authorized, make it cheap using the step-1 bypass below.

## Carry-over facts from the Phase B session (2026-09-02)

These were learned the expensive way and are not written down anywhere else.

**Skip extraction for free.** Step 1 is a single repo-wide Claude call (~$1.47 and
~6 min on RocksDB). To skip it: create the artifacts dir with a
`spotlights_manager/` subdir, copy `project_tree.json` +
`extractor_invocation.json` into it from any previous run on the same repo, and
**do not** create a `manifest.json`. With no manifest the orchestrator initializes
a fresh one using *current* fingerprints (so no resume mismatch is possible), then
finds both sidecars and takes its crash-recovery branch — logging
`extractor: recovered cached outputs from disk`. No `--resume` flag needed. See
`orchestrator.py:631-696`. `project_tree.json` holds only repo-relative paths, so
it is portable between Windows and WSL run dirs.

**`--resume` on an existing run dir is a dead end.** Step-config defaults drift
upstream, which moves `extractor_hash` / `discovery_hash` inside
`config_fingerprint`. Checked all 15 run dirs under `artifacts/` on 2026-09-02:
6 are blocked by `schema_version 1` vs current `4`, and **all 9** schema-4 dirs
have `extractor_hash` drift. `ExtractorConfig` has since gained a Stage-3 sharding
block and moved `timeout_s` 1800→5400; the hash covers the whole model dump, so no
flag can reconstruct an old fingerprint. Seed sidecars into a *fresh* dir instead.

**`--include` does not narrow extraction.** The filter is applied *after* the
extractor returns all module names (`orchestrator.py:1904`), so a narrow
`--include` still pays for full-repo extraction unless you bypass step 1.

**Usage records look like they are dropped when a step fails.** A live run on
2026-09-02 spent real money on two Claude calls, the discovery step failed
(`DiscoveryValidationError: schema parse failed twice`), and the resulting
`run_manifest.json` recorded `total_tokens: 0` with `models_used: []`. **This
matters directly to the advisor**: any calibration built from run dirs containing
failed or degraded steps will under-count tokens. Verify before trusting
`advisor_calibration.json`, and consider whether the builder should exclude or
flag non-`SUCCEEDED` modules. Not yet investigated.

**`--max-cost` is advisory only** — enforced at step boundaries, not mid-step.

## Environment

- **Work in WSL.** Both agent CLIs authenticate there through the LiteLLM proxy and
  `spotlights-engine doctor` reports 5/5 OK.
- **Source the env in the same shell**; it does not survive a `wsl -e` boundary:
  `set -a; source /home/bursh/.config/coral-env.sh; set +a`. Test the token with
  `[ -n "$ANTHROPIC_AUTH_TOKEN" ]` — never echo it.
- **Use the LiteLLM proxy only.** Never a personal `ANTHROPIC_API_KEY` — that bills
  the wrong budget.
- **`export UV_PROJECT_ENVIRONMENT=.venv-linux` before any `uv` command in WSL.**
  A plain `uv run` from WSL rebuilds `.venv` in place as a Linux venv and destroys
  the Windows one. That already happened on 2026-09-02 and is currently **not**
  repaired: `.venv` is Linux (`home = /usr/bin`, 3.12.3). Windows runs need
  `uv sync --extra dev` from PowerShell to recover.
- Tests: `uv run --extra dev pytest`. Not `python -m pytest`, not bare `uv run pytest`.
- `artifacts_dir` must not resolve **inside** `repo_path` (`orchestrator.py:538`) —
  this bit twice when the target repo was Spotlights itself.

## Standing constraints

- **Surgical changes only** — additive, no refactoring, no drive-by edits.
- **Never touch `build_config_fingerprint`** (`spotlights_manager/persistence.py`
  ~270–389). It is the `--resume` key.
- **The repo is being prepared to go public (Apache-2.0).** Never commit internal
  repo URLs, internal objectives, teammate local paths, or private commit SHAs.
  Committed calibration must stay aggregate statistics with anonymized labels,
  storing tokens rather than dollars.
- **Remote `public` only** (`AI-native-Systems-Research/spotlights`). Ignore the IBM
  `origin`. No PRs from `yevgeny/cli-permission-and-budget-fixes` — local only.
- `gh` is not on the Git Bash PATH; the user opens PRs in the web UI.

## Untracked scratch left on disk (safe to delete)

`artifacts/_probe_rocksdb-runcfg/`, `artifacts/rocksdb-runcfg-live/`, their
`spotlights-out/` siblings, `~/spotlights-self-smoke/` in WSL, and
`scripts/check_run_config_manifest.py`.
