#!/usr/bin/env bash
# Launch a coding agent backed by Qwen3.8-27B running on the VELA OpenShift cluster.
#
# The model runs remotely (vLLM pod in project wdu-research); the agent CLI runs
# locally on this Mac and reaches it through an `oc port-forward` tunnel that this
# script starts on demand.
#
#   ./scripts/qwen_agent.sh pi         [pi args...]        # earendil pi
#   ./scripts/qwen_agent.sh research   "question"          # pi in deep-research mode
#   ./scripts/qwen_agent.sh opencode   [opencode args...]  # opencode TUI
#   ./scripts/qwen_agent.sh codex      [codex args...]     # openai codex
#   ./scripts/qwen_agent.sh codex exec "one-shot prompt"
#
# The `research` preset loads the deep-research skill + web_search/web_fetch
# extension (auto-discovered from ~/.pi/agent) at xhigh reasoning effort —
# a Claude-research-style investigate/synthesize/cite loop.
#
# Note: `claude` (Claude Code) is intentionally unsupported here — it speaks the
# Anthropic /v1/messages format, while the pod serves an OpenAI-compatible API.
set -euo pipefail

MODEL="Qwen/Qwen3.8-27B"
SVC="svc/qwen38-27b-vllm-3"
PORT=18000
BASE="http://localhost:${PORT}/v1"

pf_up() { curl -sf -m3 "${BASE}/models" >/dev/null 2>&1; }

ensure_pf() {
  if pf_up; then return; fi
  command -v oc >/dev/null 2>&1 || { echo "xx  oc CLI not found" >&2; exit 1; }
  oc whoami >/dev/null 2>&1 || {
    echo "xx  not logged into OpenShift. Refresh token:" >&2
    echo "    https://oauth-openshift.apps.dmf.dipc.res.ibm.com/oauth/token/display" >&2
    echo "    then: oc login --token=sha256~... --server=https://api.dmf.dipc.res.ibm.com:6443" >&2
    exit 1
  }
  echo "==> starting port-forward ${SVC} ${PORT}:8000"
  nohup oc port-forward "$SVC" "${PORT}:8000" >/tmp/qwen_pf.log 2>&1 &
  for _ in $(seq 1 15); do pf_up && break; sleep 1; done
  pf_up || { echo "xx  port-forward failed — see /tmp/qwen_pf.log" >&2; exit 1; }
  echo "==> qwen API live at ${BASE}"
}

AGENT="${1:-pi}"; shift || true
ensure_pf

case "$AGENT" in
  pi)
    exec pi --provider vela-qwen --model "$MODEL" --thinking xhigh "$@"
    ;;
  research)
    exec pi --provider vela-qwen --model "$MODEL" --thinking xhigh \
      --append-system-prompt "Use the deep-research skill. Investigate thoroughly with web_search + web_fetch, cross-check sources, and answer with inline citations." \
      "$@"
    ;;
  opencode)
    exec opencode --model "vela-qwen/${MODEL}" "$@"
    ;;
  codex)
    export VELA_QWEN_KEY="vllm"
    exec command codex --model "$MODEL" -c model_provider=vela "$@"
    ;;
  *)
    echo "usage: $0 {pi|opencode|codex} [args...]" >&2
    exit 2
    ;;
esac
