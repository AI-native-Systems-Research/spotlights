# local_models

Run coding agents locally, backed by **Qwen3.8-27B** on the VELA OpenShift cluster.

The model runs remotely (vLLM pod, OpenAI-compatible API). The agent CLI runs on
your Mac and reaches it through an on-demand `oc port-forward`. If your OpenShift
token is stale, the token page opens in your browser automatically.

One script, two modes.

## Execute — launch an agent

```sh
./qwen.sh pi                    # earendil pi (default)
./qwen.sh research "question"   # pi in deep-research mode (web search + cite)
./qwen.sh caveman "question"    # pi in caveman mode (terse, ~75% fewer tokens)
./qwen.sh fast "question"       # pi fastest: --thinking low + caveman
./qwen.sh opencode              # opencode TUI
./qwen.sh codex "prompt"        # openai codex
```

## Source — alias `claude`/`codex --model Qwen*` → pi

```sh
source qwen.sh

claude --model Qwen-3.8-27B "explain this repo"   # → pi on qwen
claude "normal prompt"                            # → your real claude, untouched

qwen                    # → pi on qwen (default), starts port-forward if down
qwen research "q"       # any execute-mode agent works as a `qwen` subcommand
```

Once sourced, `qwen [pi|research|caveman|fast|opencode|codex]` is available as a
shell command from anywhere. It auto-starts the `oc port-forward` if it isn't
already up, so you never manage the tunnel by hand.

Add to `~/.zshrc` to make it permanent (keep it *after* any existing
`claude()`/`codex()` definitions so fallbacks survive):

```sh
source /Users/michaelsoloveitchik/Repositories/Projects/local_models/qwen.sh
```

## Requirements — auto-installed

`qwen.sh` self-bootstraps. On launch it checks each dependency and installs
whatever is missing, then handles OpenShift login:

| Dependency | Check → auto-install |
|------------|----------------------|
| `oc` (OpenShift CLI) | `brew install openshift-cli` |
| `pi` | `npm install -g @earendil-works/pi-coding-agent` |
| `opencode` | `curl -fsSL https://opencode.ai/install \| bash` |
| `codex` | `npm install -g @openai/codex` |
| doc readers (pi) | `brew install poppler pandoc tesseract` — PDF/office/OCR, installed with pi |
| OpenShift session | refresh from `~/.claude/tokens` (`VELA_OC_TOKEN`); else open token page + prompt you to paste `oc login ...` |

Only the agent you launch is installed (e.g. `qwen.sh pi` won't pull codex).

Example first run (nothing installed yet):

```console
$ ./qwen.sh pi "hello"
==> oc (OpenShift CLI) not found — installing
==> pi not found — installing @earendil-works/pi-coding-agent
xx  OpenShift token missing/expired — opening token page:
    https://oauth-openshift.apps.dmf.dipc.res.ibm.com/oauth/token/display
    Paste the 'oc login ...' command (or Enter to skip): oc login --token=sha256~… --server=https://api.dmf.dipc.res.ibm.com:6443
==> logged in
==> starting port-forward svc/qwen38-27b-vllm-3 18000:8000
==> qwen API live at http://localhost:18000/v1
… pi starts …
```

Second run: everything cached, port-forward reused → straight into the agent.

Prereqs the script assumes present: `curl`, plus `brew` (for `oc`) and `npm`
(for pi/codex). Install those first if missing:

```sh
# Homebrew (mac)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Node/npm
brew install node
```

### OpenShift token

Stored in `~/.claude/tokens` as `VELA_OC_TOKEN` (and optional `VELA_OC_SERVER`).
When the session is stale the script opens the token page:
<https://oauth-openshift.apps.dmf.dipc.res.ibm.com/oauth/token/display>
— copy the `oc login --token=sha256~... --server=...` line and paste it when
prompted (interactive), or run it yourself and re-launch.

## Notes

- Claude Code itself **cannot** use this endpoint — it speaks Anthropic
  `/v1/messages`; the pod serves OpenAI format. Use pi/opencode/codex.
- Reasoning effort on this model: `xhigh` (default) / `medium` / `low` only.
- PDF/office/scan reading: pi has no native binary decode — it shells out via
  `bash` to `pdftotext` / `pandoc` / `tesseract` (auto-installed with pi). Just
  ask: *"read paper.pdf and summarize"*.
