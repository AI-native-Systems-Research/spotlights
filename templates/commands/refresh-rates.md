---
description: Use when Spotlights cost numbers need the real gateway prices — "refresh rates", "update the rate table", "price this run with our LiteLLM contract", "doctor says no rate row", "cost is PARTIAL / unpriced_models". Finds the LiteLLM endpoint and key in this environment, runs `spotlights-engine rates refresh` to build a rate table from the gateway's /model/info prices, checks that the models the CLIs really use are priced, and tells the user how to point SPOTLIGHTS_RATES_FILE at it. Never invents a price.
---

# Refresh Rates

Builds a Spotlights rate table from the prices a LiteLLM gateway publishes, so
`run_manifest.json` costs match the contract this environment is billed at.

The deterministic work is done by `spotlights-engine rates refresh`. Your job
is the part a fixed command cannot do: find the right gateway in an unknown
environment, get past auth or routing quirks, and check the result covers the
models a run will actually use.

**Hard rule:** never write a price yourself, never copy a public list price
into the table, and never edit the bundled `rates.json` in the installed
package. If the gateway does not publish a price for a model, report it and
stop there — an unpriced model shows as PARTIAL cost, which is honest; a guessed
price is not.

## Step 1 — Preview with discovery

Run a dry run first. It writes nothing:

```bash
spotlights-engine rates refresh --dry-run > /tmp/spotlights-rates-preview.json
```

The summary on stderr says which endpoint it used and where it found it
(`--endpoint`, `SPOTLIGHTS_LITELLM_URL`, `LITELLM_BASE_URL`, `ANTHROPIC_BASE_URL`
from env or `~/.claude/settings.json`, `OPENAI_BASE_URL`, or the active
provider's `base_url` in `~/.codex/config.toml`).

- **Exit 0** → go to Step 3.
- **"no LiteLLM endpoint found"** → Step 2a.
- **HTTP 401 / 403** → Step 2b.
- **HTTP 404 on both paths, or "no 'data' list"** → Step 2c.
- **"returned no priced model"** → the gateway has no prices configured.
  Tell the user, show the `skipped` lines, and stop.

Never print API keys. If you show a command or header to the user, mask the key.

## Step 2 — Fix discovery

**2a. No endpoint.** Look for one, in this order, and pass the first that
answers with `--endpoint`:

1. Env vars with `BASE_URL`, `ROUTER`, `PROXY`, or `LITELLM` in the name
   (`env | grep -iE 'base_url|router|proxy|litellm' | sed 's/=.*key.*/=***/I'`).
2. `~/.claude/settings.json` → `env.ANTHROPIC_BASE_URL`.
3. `~/.codex/config.toml` → `[model_providers.<model_provider>].base_url`.
4. A local router port in the process list or a sandbox README.

If nothing turns up, ask the user for the gateway URL with `AskUserQuestion`.

**2b. Auth error.** Find which env var holds the gateway key and pass it with
`--api-key-env VAR` (the command sends `Authorization: Bearer <value>`). Some
gateways want another header instead — pass it with `--header 'Name: value'`.
`/model/info` may need an admin or "info" key on locked-down proxies; if so,
tell the user and stop.

**2c. Not a plain LiteLLM.** Probe the paths the command tried
(`<base>/model/info`, `<base>/v1/model/info`) with `curl -sS -m 30` to see what
comes back. A front-door router that scopes the model list per harness
(each CLI sees a different list) needs per-CLI mode:

```bash
spotlights-engine rates refresh \
  --claude-header 'X-Platform-Harness: claude-code' \
  --codex-header  'X-Platform-Harness: codex' --dry-run
```

In per-CLI mode every row from the Claude fetch is keyed `anthropic:*` and every
row from the Codex fetch `openai:*`, even when the model name does not say so.

## Step 3 — Check coverage

The table only helps if it prices the model ids the CLIs **report**. Check:

1. **`openai:codex` line** in the summary. It must read `openai:codex → openai:<model>`.
   If it says "kept from base", the gateway does not price the model Codex runs.
   Find the real Codex model (`spotlights-engine doctor` prints it, as does
   `model` in `~/.codex/config.toml`) and re-run with `--codex-model <id>`.
2. **Models in use.** Collect the model ids the CLIs will report:
   - `spotlights-engine doctor` (spends a few cents on two one-word probes —
     ask first), or
   - `cost.unpriced_models` and `models_used[*].model` in a recent
     `run_manifest.json` (free).
   For each id, look up its canonical key in the preview. Keys drop the route
   prefix (`aws/`, `azure/`, `bedrock/`, …) and the `[1m]` context tag, so
   `aws/claude-opus-5[1m]` is priced by `anthropic:claude-opus-5`.
3. **Skipped lines.** "not a Claude/OpenAI model" is fine unless one of those
   models is in use. "conflicting prices" means two gateway aliases map to the
   same key with different prices — tell the user both aliases; do not pick one.

Show the user a short table: model in use → rate key → input / output per MTok
(per-token × 1,000,000) → source (gateway or base table).

## Step 4 — Write and activate

Ask the user where to write the table (default `~/.config/spotlights/rates.json`),
then run the same command without `--dry-run`, with `--out` if they chose
another path. The command validates the file with the engine's own loader before
it replaces the old one.

Tell the user to activate it in the shell that runs the engine:

```bash
export SPOTLIGHTS_RATES_FILE="$HOME/.config/spotlights/rates.json"
```

Offer to add that line to their shell profile; only do it if they say yes.
Suggest `spotlights-engine doctor` as the final check.

## Notes

- `SPOTLIGHTS_RATES_FILE` replaces the whole table. The command copies the
  base table's rows first (bundled by default, or `--base <file>` to build on
  an existing contracted table), then lets gateway prices replace matching rows.
- Gateway prices change. Re-run this skill when the contract changes; each row's
  `note` records the gateway alias and the fetch date.
- This only touches the contracted table. The public list-price table
  (`SPOTLIGHTS_EXTERNAL_RATES_FILE`) is left alone.
