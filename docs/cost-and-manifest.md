# Cost and the run manifest

Every Spotlights run writes a manifest recording provenance, token usage, timing, and cost. Cost is computed from token counts and a rate table you can override to match your contracted LiteLLM pricing. This page explains the manifest, the rate-table format, and how to price a model the bundled table doesn't know about.

← Back to [README](../README.md)

## The run manifest

Each completed engine run writes a public run manifest to:

```text
<artifacts-dir>/spotlights_manager/run_manifest.json
<output-folder>/run_manifest.json
```

The manifest records the target repo URL and commit, the Spotlights engine commit, the objective, grouped per-model token usage, wall/API timing, output counts, and cost. Usage is captured from the Claude and Codex CLI streams as durable per-invocation records under each module's artifacts directory, then aggregated from disk when the run finishes or resumes.

It also records `models_requested` beside `models_used`: the model id the engine
*asked* each CLI for (from `--claude-model` / `--codex-model` or `models.yaml`)
next to what the CLI streams actually reported back. The two differ legitimately.
An empty `models_requested` entry means the engine passed no `--model` at all and
the CLI chose for itself; and where a value was passed, a gateway alias such as
`aws/claude-opus-5` may come back reported under a different id. Use
`models_requested` to answer "what did this run intend", and `models_used` to
answer "what actually ran". See [Choosing models](agent-cli-setup.md#choosing-models).

## Cost and rate tables

Cost is computed from token counts and a rate table. The engine never uses CLI-reported `total_cost_usd` for billing, because those values can reflect provider list price rather than your LiteLLM contract. The checked-in table at `src/spotlights_engine/costing/rates.json` contains public default rates for the current Claude/Codex models. Point `SPOTLIGHTS_RATES_FILE` at your contracted LiteLLM table before running if your billing differs:

```bash
export SPOTLIGHTS_RATES_FILE="$HOME/.config/spotlights/rates.json"

spotlights-engine \
  --repo ../vllm \
  --include vllm/v1/kv_offload \
  --objective "reduce the median TTFT and median TPOT (Time Per Output Token)" \
  --output-folder ./spotlights-out \
  --artifacts-dir ./artifacts
```

Rate files are JSON keyed by `"provider:model"`:

```json
{
  "anthropic:aws/claude-opus-4-8": {
    "input": 0.0000038,
    "output": 0.000019,
    "cache_read": 0.00000038,
    "cache_create": 0.00000475,
    "unit": "per_token",
    "note": "LiteLLM (bedrock) rate as of 2026-07-07: Claude Opus 4.8 $3.80/MTok input, $19/MTok output, $0.38/MTok cache read, $4.75/MTok cache write. Override for a different contract or 1h cache writes ($10/MTok)."
  },
  "openai:codex": {
    "input": 0.0000025,
    "output": 0.000015,
    "cache_read": 0.0000005,
    "cache_create": 0.0,
    "unit": "per_token",
    "note": "LiteLLM (azure) rate as of 2026-07-07: GPT-5.5 $2.50/MTok input, $15/MTok output, $0.50/MTok cache read; Codex has no cache-create bucket. Override for a different contract."
  }
}
```

The key must be **exactly** the `provider:model` id the CLI reports — that is what the cost lookup matches against. Claude reports a bedrock-style id like `aws/claude-opus-4-8`, so its key is `anthropic:aws/claude-opus-4-8`. Codex reports no resolvable model id, so its usage falls back to the CLI-family key `openai:codex`. If you are unsure what id your CLI reports, run `spotlights-engine doctor` — it prints the exact key it looked up.

If a model is missing from the table, Spotlights still writes the manifest and prices the models it can. The manifest's `cost.rate_note` and `notes` fields list any unpriced models so partial cost is visible.

## Adding a model to the rate table

If `doctor` (or a run's `unpriced_models`) reports a model with no rate row — e.g. `[FAIL] claude: reported model 'aws/claude-opus-4-7' has no rate row (anthropic:aws/claude-opus-4-7)` — add a row for it:

```bash
# 1. Copy the bundled table (or start from your existing contracted one)
mkdir -p "$HOME/.config/spotlights"
cp src/spotlights_engine/costing/rates.json "$HOME/.config/spotlights/rates.json"

# 2. Add a row keyed by the EXACT id doctor printed inside the parentheses,
#    e.g. "anthropic:aws/claude-opus-4-7", with your per-token rates.

# 3. Point the engine at your table and re-check
export SPOTLIGHTS_RATES_FILE="$HOME/.config/spotlights/rates.json"
spotlights-engine doctor
```

`doctor` reads the same table (`SPOTLIGHTS_RATES_FILE` if set, else the bundled default), so a green `doctor` means the run will price that model.
