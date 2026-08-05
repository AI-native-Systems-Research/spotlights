#!/bin/bash
# Common env sourced by every CCC submit script.
# Activates the spotlights-engine venv, ensures CLIs are on PATH, loads secrets.

set -eo pipefail

# CLI PATH:
#   npm-global/bin — where `claude`/`codex`/`gemini` shims live
#   spotlights-node/bin — provides `node`, which the CLI shims need on PATH
#   ~/.local/bin — user-installed tools (uv, gh)
export PATH="/dccstor/idanfr01/spotlights/envs/npm-global/bin:/dccstor/idanfr01/spotlights/envs/spotlights-node/bin:$HOME/.local/bin:$PATH"

# uv + HF caches on GPFS so they don't fill $HOME (25 GB quota)
export UV_PYTHON_INSTALL_DIR=/dccstor/idanfr01/spotlights/envs/uv-pythons
export UV_CACHE_DIR=/dccstor/idanfr01/spotlights/envs/uv-cache
export HF_HOME=/dccstor/idanfr01/spotlights/hf-cache
export HUGGINGFACE_HUB_CACHE=/dccstor/idanfr01/spotlights/hf-cache/hub
export TRANSFORMERS_CACHE=/dccstor/idanfr01/spotlights/hf-cache/transformers

# LiteLLM proxy credentials (ANTHROPIC_*, GEMINI_*, etc.). File is chmod 600.
if [ -r "$HOME/.spotlights.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.spotlights.env"
else
  echo "ERROR: $HOME/.spotlights.env not found — API keys missing" >&2
  exit 1
fi

# Activate engine venv (torch not required; small deps)
VENV=/dccstor/idanfr01/spotlights/envs/engine-venv
if [ ! -x "$VENV/bin/spotlights-engine" ]; then
  echo "ERROR: engine venv missing at $VENV" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Sanity-check every CLI the engine will shell out to.
for bin in spotlights-engine claude codex gemini; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: '$bin' not found on PATH" >&2
    exit 1
  fi
done
