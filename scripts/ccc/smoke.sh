#!/bin/bash
# Fast end-to-end smoke test — verifies the engine setup works before you commit
# to a long batch run. Runs on the login node (cheap, < 2 min).
#
# Checks in order:
#   1) engine venv activates, spotlights-engine --help runs
#   2) claude CLI: single-turn LiteLLM round-trip
#   3) codex CLI:  single-turn LiteLLM round-trip
#   4) gemini CLI: single-turn LiteLLM round-trip (skipped if it fails — Gemini
#      often needs its web-tool aliases wired; keep going and inspect manually)
#
# Any 1–3 failure is a hard stop: the engine will fail the same way in a job.

set -eo pipefail

REPO=/u/idanfr/spotlights-engine
source "$REPO/scripts/ccc/env.sh"

echo "=== engine ==="
spotlights-engine --help | head -3
echo "OK"
echo

echo "=== claude (LiteLLM) ==="
claude -p "Reply with exactly the word: OK" 2>&1 | tail -5
echo

echo "=== codex (LiteLLM) ==="
codex exec --skip-git-repo-check "Reply with exactly the word: OK" 2>&1 | tail -5
echo

echo "=== gemini (LiteLLM) ==="
if gemini --prompt "Reply with exactly: OK" --output-format json --approval-mode yolo 2>&1 | tail -10; then
  echo "gemini OK"
else
  echo "gemini FAILED (non-blocking — check ~/.gemini/settings.json alias wiring)"
fi

echo
echo "smoke complete"
