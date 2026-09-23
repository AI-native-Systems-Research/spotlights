#!/usr/bin/env zsh
# qwen.sh — one script, two modes.
#
#   EXECUTE  ./qwen.sh {pi|research|opencode|codex} [args]   -> launch an agent on Qwen3.8-27B
#   SOURCE   source qwen.sh                                  -> alias claude/codex --model qwen* -> pi
#
# The model runs on the VELA OpenShift cluster (vLLM); the agent CLI runs locally
# and reaches it via an on-demand `oc port-forward`. If the OpenShift session is
# stale, the token page is opened in the browser.
#
# Self-bootstrapping: on launch it checks for `oc` and the chosen agent CLI
# (pi/opencode/codex) and installs whatever is missing, then prompts for an
# OpenShift login if the session is stale.

MODEL="Qwen/Qwen3.8-27B"
SVC="svc/qwen38-27b-vllm-3"
PORT=18000
BASE="http://localhost:${PORT}/v1"
OC_SERVER="https://api.dmf.dipc.res.ibm.com:6443"
TOKEN_URL="https://oauth-openshift.apps.dmf.dipc.res.ibm.com/oauth/token/display"
TOKENS_FILE="$HOME/.claude/tokens"
DOC_PROMPT="To read PDFs/office docs/scans, use bash: 'pdftotext file.pdf -' (add -layout for tables), 'pandoc f.docx -t markdown', 'tesseract img.png -' for scanned images. Convert, then read the text."

have()  { command -v "$1" >/dev/null 2>&1; }
pf_up() { curl -sf -m3 "${BASE}/models" >/dev/null 2>&1; }

# Active liveness probe: is the backend *generating*, or just accepting TCP?
# `pf_up` (the /models reachability check) can pass while vLLM is wedged and no
# completion ever returns — the state that makes a run look silently hung. This
# fires a 1-token completion with a short deadline and reports which of the
# three states we're in. Exit: 0 generating, 1 unreachable, 2 blocked.
QWEN_PROBE_TIMEOUT="${QWEN_PROBE_TIMEOUT:-20}"
qwen_probe() {
  if ! pf_up; then
    echo "xx  UNREACHABLE: ${BASE}/models not answering — tunnel down (run: qwen.sh ensure)" >&2
    return 1
  fi
  local start end body
  start=$(date +%s.%N)
  body=$(curl -sf -m"${QWEN_PROBE_TIMEOUT}" "${BASE}/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0,\"stream\":false}" 2>/dev/null)
  local rc=$?
  end=$(date +%s.%N)
  local dt; dt=$(awk "BEGIN{printf \"%.2f\", ${end}-${start}}")
  if [ $rc -ne 0 ]; then
    echo "xx  BLOCKED: reachable but no completion in ${dt}s (timeout=${QWEN_PROBE_TIMEOUT}s) — vLLM wedged/saturated" >&2
    return 2
  fi
  if print -r -- "$body" | grep -q '"choices"'; then
    echo "==> OK: backend generating (${dt}s round-trip on ${MODEL})"
    return 0
  fi
  echo "xx  BLOCKED: completion returned no choices in ${dt}s: ${body:0:200}" >&2
  return 2
}

open_url() {
  if have open; then open "$1"
  elif have xdg-open; then xdg-open "$1"
  fi
}

# --- dependency bootstrap: check, else install --------------------------------

ensure_oc() {
  have oc && return 0
  echo "==> oc (OpenShift CLI) not found — installing"
  if have brew; then brew install openshift-cli
  else echo "xx  install oc manually: https://mirror.openshift.com/pub/openshift-v4/clients/ocp/stable/" >&2; return 1; fi
  have oc || { echo "xx  oc install failed" >&2; return 1; }
}

# Doc/PDF readers pi drives via bash (pdftotext/pandoc/tesseract). Installed once.
ensure_doctools() {
  have pdftotext && return 0
  have brew || return 0   # optional — skip silently if no brew
  echo "==> installing doc readers (poppler pandoc tesseract) for PDF/office/OCR"
  brew install poppler pandoc tesseract >/dev/null 2>&1 || true
}

ensure_pi() {
  have pi || {
    echo "==> pi not found — installing @earendil-works/pi-coding-agent"
    have npm || { echo "xx  npm required to install pi (get Node.js first)" >&2; return 1; }
    npm install -g @earendil-works/pi-coding-agent
    have pi || { echo "xx  pi install failed" >&2; return 1; }
  }
  ensure_doctools
}

ensure_opencode() {
  have opencode && return 0
  echo "==> opencode not found — installing from opencode.ai"
  curl -fsSL https://opencode.ai/install | bash
  have opencode || { echo "xx  opencode install failed" >&2; return 1; }
}

ensure_codex() {
  have codex && return 0
  echo "==> codex not found — installing @openai/codex"
  have npm || { echo "xx  npm required to install codex (get Node.js first)" >&2; return 1; }
  npm install -g @openai/codex
  have codex || { echo "xx  codex install failed" >&2; return 1; }
}

# Live OpenShift session, or refresh from stored token, or open the token page.
ensure_login() {
  oc whoami >/dev/null 2>&1 && return 0
  if [ -f "$TOKENS_FILE" ]; then
    . "$TOKENS_FILE"
    if [ -n "${VELA_OC_TOKEN:-}" ]; then
      oc login --token="$VELA_OC_TOKEN" --server="${VELA_OC_SERVER:-$OC_SERVER}" >/dev/null 2>&1 \
        && oc whoami >/dev/null 2>&1 && { echo "==> OpenShift session refreshed from stored token"; return 0; }
    fi
  fi
  echo "xx  OpenShift token missing/expired — opening token page:" >&2
  echo "    ${TOKEN_URL}" >&2
  open_url "$TOKEN_URL"
  # Interactive: let the user paste the whole 'oc login --token=... --server=...'.
  if [ -t 0 ]; then
    printf "    Paste the 'oc login ...' command (or Enter to skip): " >&2
    local cmd; read -r cmd
    if [ -n "$cmd" ]; then
      eval "$cmd" >/dev/null 2>&1 && oc whoami >/dev/null 2>&1 && { echo "==> logged in"; return 0; }
      echo "xx  login command failed — check token" >&2
    fi
  else
    echo "    Run the 'oc login --token=sha256~... --server=...' it shows, then retry." >&2
  fi
  return 1
}

ensure_pf() {
  pf_up && return 0
  ensure_oc || return 1
  ensure_login || return 1
  # Retry the establish across a transient pod-connection blip. `oc
  # port-forward` dies at the data layer ("error: lost connection to pod")
  # even while the pod is healthy and Running — a momentary proxy/network hiccup.
  # A single spawn whose 15s readiness poll happens to land inside that window
  # gives up and the caller (ensure_backend) raises LocalAgentError, killing a
  # multi-hour run over a <20s blip. So we respawn up to 3 times, reaping the
  # dead forward before each try.
  #
  # Reap stale/dead forwards for this port BEFORE each spawn. A dead proc still
  # holds :PORT (pf_up false but bound); without the reap each restart stacks
  # another oc and the survivors fight over the port — the flapping that makes
  # calls hang. We only reach here when pf_up already failed, so we never kill
  # a working forward.
  local attempt
  for attempt in 1 2 3; do
    pkill -f "oc port-forward ${SVC} ${PORT}:8000" 2>/dev/null && sleep 1
    echo "==> starting port-forward ${SVC} ${PORT}:8000 (attempt ${attempt}/3)"
    nohup oc port-forward "$SVC" "${PORT}:8000" >/tmp/qwen_pf.log 2>&1 &
    local _; for _ in $(seq 1 15); do pf_up && break; sleep 1; done
    if pf_up; then echo "==> qwen API live at ${BASE}"; return 0; fi
    echo "xx  port-forward attempt ${attempt}/3 failed — see /tmp/qwen_pf.log" >&2
    sleep 2
  done
  echo "xx  port-forward failed after 3 attempts — see /tmp/qwen_pf.log" >&2
  return 1
}

# Launch an agent (no exec — safe when called from an interactive shell).
qwen_launch() {
  local agent="${1:-pi}"; shift 2>/dev/null || true
  ensure_pf || return 1
  case "$agent" in
    pi)       ensure_pi || return 1
              pi --provider vela-qwen --model "$MODEL" --thinking xhigh \
                --append-system-prompt "$DOC_PROMPT" "$@" ;;
    research) ensure_pi || return 1
              pi --provider vela-qwen --model "$MODEL" --thinking xhigh \
                --append-system-prompt "$DOC_PROMPT" \
                --append-system-prompt "Use the deep-research skill. Investigate with web_search + web_fetch, cross-check sources, answer with inline citations." "$@" ;;
    caveman)  ensure_pi || return 1
              pi --provider vela-qwen --model "$MODEL" --thinking xhigh \
                --append-system-prompt "$DOC_PROMPT" \
                --append-system-prompt "Use the caveman skill. Respond terse like smart caveman: drop articles/filler/pleasantries/hedging, fragments OK, all technical substance stays. Code/commits normal." "$@" ;;
    fast)     ensure_pi || return 1
              pi --provider vela-qwen --model "$MODEL" --thinking low \
                --append-system-prompt "$DOC_PROMPT" \
                --append-system-prompt "Use the caveman skill. Respond terse like smart caveman: drop articles/filler/pleasantries/hedging, fragments OK, all technical substance stays. Code/commits normal." "$@" ;;
    opencode) ensure_opencode || return 1
              opencode --model "vela-qwen/${MODEL}" "$@" ;;
    codex)    ensure_codex || return 1
              VELA_QWEN_KEY="vllm" command codex --model "$MODEL" -c model_provider=vela "$@" ;;
    *) echo "usage: qwen.sh {pi|research|caveman|fast|opencode|codex} [args]" >&2; return 2 ;;
  esac
}

# --- alias mode helpers ----------------------------------------------------

_qwen_has_qwen() { local a; for a in "$@"; do [[ "${a:l}" == *qwen* ]] && return 0; done; return 1; }

_qwen_pi_from_cli() {   # drop the incoming --model/-m, forward the rest to pi
  local -a rest; local a skip=0
  for a in "$@"; do
    if (( skip )); then skip=0; continue; fi
    case "$a" in --model|-m) skip=1 ;; --model=*) ;; *) rest+=("$a") ;; esac
  done
  qwen_launch pi "${rest[@]}"
}

_qwen_is_sourced() {
  # Note: evaluated inside this function, so zsh appends ":shfunc" to the
  # context — match ":file" anywhere, not just as a suffix.
  if [ -n "${ZSH_VERSION:-}" ]; then [[ "${ZSH_EVAL_CONTEXT:-}" == *file* ]]
  else [ "${BASH_SOURCE[0]:-}" != "${0}" ]; fi
}

if _qwen_is_sourced; then
  # Preserve any existing claude()/codex() (e.g. LiteLLM routing) before overriding.
  (( ${+functions[claude]} )) && ! (( ${+functions[_orig_claude]} )) && functions[_orig_claude]=$functions[claude]
  (( ${+functions[codex]} ))  && ! (( ${+functions[_orig_codex]} ))  && functions[_orig_codex]=$functions[codex]

  claude() {
    if _qwen_has_qwen "$@"; then _qwen_pi_from_cli "$@"
    elif (( ${+functions[_orig_claude]} )); then _orig_claude "$@"
    else command claude "$@"; fi
  }
  codex() {
    if _qwen_has_qwen "$@"; then _qwen_pi_from_cli "$@"
    elif (( ${+functions[_orig_codex]} )); then _orig_codex "$@"
    else command codex "$@"; fi
  }
  # Direct launcher: `qwen [pi|research|caveman|fast|opencode|codex] [args]`.
  # Auto-starts the port-forward if down (via qwen_launch -> ensure_pf).
  qwen() { qwen_launch "$@"; }
else
  # `qwen.sh ensure` just brings the backend up (oc login + port-forward) and
  # exits — used by the Python dispatch (local_agent.pi_runner.ensure_backend)
  # to make the local Qwen endpoint reachable before spawning pi.
  if [ "${1:-}" = "ensure" ]; then
    ensure_pf
  elif [ "${1:-}" = "probe" ]; then
    # `qwen.sh probe` — on-demand "is the API blocked?" check. Does NOT start
    # the tunnel (use `ensure` for that); just reports reachable/generating/
    # blocked so a hung run can be diagnosed without reading Python logs.
    qwen_probe
  else
    qwen_launch "$@"
  fi
fi
