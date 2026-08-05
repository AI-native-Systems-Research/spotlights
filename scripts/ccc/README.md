# CCC runbook for spotlights-engine

Scripts under `~/spotlights-engine/scripts/ccc/` — checked in? Currently not; local to this CCC install.

## Layout on this cluster

- **Home (`/u/idanfr`, 25 GB, backed up)**
  - `~/spotlights-engine/` — git clone
  - `~/colpali/` — git clone (env exists but not wired for jobs)
  - `~/.spotlights.env` — API keys (chmod 600); sourced by every submit script
  - `~/.claude/settings.json`, `~/.codex/config.toml`, `~/.gemini/{settings.json,.env}` — LiteLLM proxy configs
  - `~/runs`, `~/hf-cache`, `~/logs-dccstor` — symlinks into `/dccstor/idanfr01/spotlights/`

- **GPFS (`/dccstor/idanfr01/spotlights`, ~135 GB free, NOT backed up)**
  - `envs/engine-venv/` — Python 3.11 venv for spotlights-engine (uv-managed)
  - `envs/colpali-venv/` — Python 3.11 venv for colpali (torch/CUDA)
  - `envs/npm-global/bin/{claude,codex,gemini}` — CLI binaries
  - `envs/spotlights-node/` — conda env holding node 22
  - `hf-cache/` — HuggingFace model cache
  - `runs/`, `logs/`, `artifacts/`, `models/` — outputs

`.bashrc` puts CLIs + caches on PATH and defines `engine-activate` / `colpali-activate` aliases.

## Common commands

```bash
# One-time smoke: verify env + all 3 CLIs work through LiteLLM
bash ~/spotlights-engine/scripts/ccc/smoke.sh

# Interactive shell on a compute node (4 cores, 16 GB, 4h)
~/spotlights-engine/scripts/ccc/interactive-engine.sh

# Submit a batch run (edit env vars first)
TARGET_REPO=/u/idanfr/some-clone \
INCLUDE="pkg/mod_a pkg/mod_b" \
OBJECTIVE="reduce p99 latency" \
HINT="multi-turn agentic workload" \
RUN_TAG="myrun-2026-07-08" \
bsub < ~/spotlights-engine/scripts/ccc/run-engine.lsf

# Check jobs / logs
bjobs
tail -f /dccstor/idanfr01/spotlights/logs/engine.<JOBID>.stdout
```

## Why no GPU for the engine

spotlights-engine is API-bound: it orchestrates claude/codex/gemini via LiteLLM.
A GPU node would sit at 0% util the whole time and waste the reservation.
Use `-q normal` with `-n 8 -M 32768`. If you also want colpali on GPU later,
add a separate submit script (colpali-venv is already installed).

## Rotating keys

Edit `~/.spotlights.env` and the two files that duplicate secrets:
- `~/.claude/settings.json` (`ANTHROPIC_AUTH_TOKEN`)
- `~/.codex/config.toml` (`api_key`)
- `~/.gemini/.env` (`GEMINI_API_KEY`, `LITELLM_API_KEY`)
