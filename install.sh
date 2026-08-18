#!/usr/bin/env sh
# Spotlights installer — installs the `spotlights-engine` CLI globally via uv.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/AI-native-Systems-Research/spotlights/main/install.sh | sh
#
# Pin a version (any git tag, branch, or commit):
#   curl -fsSL https://raw.githubusercontent.com/AI-native-Systems-Research/spotlights/main/install.sh | SPOTLIGHTS_VERSION=v0.1.0 sh
#
# What it does:
#   1. Installs `uv` if missing (via https://astral.sh/uv/install.sh)
#   2. Runs `uv tool install --force git+https://github.com/AI-native-Systems-Research/spotlights.git`
#      — places `spotlights-engine` (and the sibling CLIs) in ~/.local/bin
#      (an isolated venv, on PATH)
#   3. Ensures ~/.local/bin is on PATH in future shells
#
# For local development, clone the repo and run `uv sync --all-extras` instead —
# that gives you an editable install which the CLIs automatically prefer.
set -eu

REPO="https://github.com/AI-native-Systems-Research/spotlights.git"
VERSION="${SPOTLIGHTS_VERSION:-main}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$1" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$1" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is required to install from source. Install git and retry."
command -v curl >/dev/null 2>&1 || die "curl is required to bootstrap uv. Install curl and retry."

if ! command -v uv >/dev/null 2>&1; then
  say "uv not found — installing from astral.sh"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv's installer writes shell rc entries but doesn't update this shell
  PATH="$HOME/.local/bin:$PATH"
  export PATH
  command -v uv >/dev/null 2>&1 || die "uv install completed but 'uv' is still not on PATH. Open a new shell and retry."
fi

say "Installing spotlights-engine from ${REPO}@${VERSION}"
uv tool install --force "git+${REPO}@${VERSION}"

# Make sure ~/.local/bin is on PATH for future shells (idempotent)
uv tool update-shell >/dev/null 2>&1 || true

if command -v spotlights-engine >/dev/null 2>&1; then
  say "Installed: spotlights-engine ($(command -v spotlights-engine))"
  cat <<'EOF'

Next steps:
  spotlights-engine --help              List commands
  spotlights-engine init                Install the Claude Code slash commands
  spotlights-engine doctor              Preflight the agent CLIs + auth
  spotlights-engine --repo ../vllm ...  Run the engine against a target repo

Docs:  https://github.com/AI-native-Systems-Research/spotlights#readme
EOF
else
  warn "Install succeeded, but 'spotlights-engine' is not on PATH in this shell."
  warn "Open a new terminal — or run:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
