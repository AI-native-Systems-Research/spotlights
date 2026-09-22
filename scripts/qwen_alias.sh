#!/usr/bin/env zsh
# Transparent qwen redirect for `claude` / `codex`.
#
# When you are anywhere inside a `spotlights` tree AND pass a Qwen model
# (e.g. `claude --model Qwen-3.8-27B ...`), the call is rerouted to the pi
# agent backed by Qwen3.8-27B on VELA (via scripts/qwen_agent.sh pi).
#
# Otherwise the call falls through to your normal `claude` / `codex`
# (the IBM-LiteLLM routing functions defined in ~/.zshrc), unchanged.
#
# Source this AFTER your existing claude()/codex() definitions:
#   source /Users/michaelsoloveitchik/Repositories/Projects/spotlights/open_source/dev/OpenAlex/spotlights/scripts/qwen_alias.sh

_SPOT_QWEN_LAUNCHER="/Users/michaelsoloveitchik/Repositories/Projects/spotlights/open_source/dev/OpenAlex/spotlights/scripts/qwen_agent.sh"

# Preserve the pre-existing claude()/codex() once, so fallback keeps working.
(( ${+functions[claude]} )) && ! (( ${+functions[_orig_claude]} )) && functions[_orig_claude]=$functions[claude]
(( ${+functions[codex]} ))  && ! (( ${+functions[_orig_codex]} ))  && functions[_orig_codex]=$functions[codex]

# True if any arg looks like a Qwen model (case-insensitive).
_spot_has_qwen() {
  local a
  for a in "$@"; do
    [[ "${a:l}" == *qwen* ]] && return 0
  done
  return 1
}

# Launch pi on qwen, dropping the incoming --model/-m (and its value).
_spot_run_pi_qwen() {
  local -a rest
  local a skip=0
  for a in "$@"; do
    if (( skip )); then skip=0; continue; fi
    case "$a" in
      --model|-m) skip=1 ;;
      --model=*)  ;;
      *) rest+=("$a") ;;
    esac
  done
  "$_SPOT_QWEN_LAUNCHER" pi "${rest[@]}"
}

claude() {
  if [[ "$PWD" == */spotlights* ]] && _spot_has_qwen "$@"; then
    _spot_run_pi_qwen "$@"
  elif (( ${+functions[_orig_claude]} )); then
    _orig_claude "$@"
  else
    command claude "$@"
  fi
}

codex() {
  if [[ "$PWD" == */spotlights* ]] && _spot_has_qwen "$@"; then
    _spot_run_pi_qwen "$@"
  elif (( ${+functions[_orig_codex]} )); then
    _orig_codex "$@"
  else
    command codex "$@"
  fi
}
