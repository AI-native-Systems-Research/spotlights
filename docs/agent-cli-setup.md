# Installing the agent CLIs

Spotlights drives two external agent CLIs — `claude` (Claude Code) and `codex` (Codex). Both must be installed and authenticated before a run. This page covers installing them, optionally routing them through a LiteLLM gateway, and a quick response check.

← Back to [README](../README.md)

## Install

| Agent | Install |
| --- | --- |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `curl -fsSL https://claude.ai/install.sh \| bash` |
| [Codex](https://github.com/openai/codex) | `curl -fsSL https://chatgpt.com/codex/install.sh \| sh` |

## Route through the LiteLLM gateway (optional)

```bash
export LITELLM_API_KEY="sk-your-virtual-key"   # add to ~/.bashrc / ~/.zshrc
```

Replace the gateway URL and aliases below with your own.

**Claude Code** — `~/.claude/settings.json` (Anthropic wire format → LiteLLM's `/v1/messages`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://your-litellm-gateway.example.com",
    "ANTHROPIC_AUTH_TOKEN": "sk-your-virtual-key",
    "ANTHROPIC_MODEL": "claude-sonnet",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku"
  }
}
```

**Codex** — `~/.codex/config.toml` (OpenAI wire format → LiteLLM's `/v1`):

```toml
model = "gpt-5-codex"          # your LiteLLM alias
model_provider = "litellm"

[model_providers.litellm]
name = "LiteLLM"
base_url = "https://your-litellm-gateway.example.com/v1"   # note the /v1
env_key = "LITELLM_API_KEY"
wire_api = "chat"              # use "responses" only if the gateway is set up for it
requires_openai_auth = false  # gateway key, not an OpenAI sk- key
```

## Choosing models

A run's model has to be something you choose, not something inherited from
whoever's machine it ran on — otherwise the same command produces different
results for different people.

Out of the box the bundled `models.yaml` pins **Codex to `gpt-5.5`** (the id the
engine used to hardcode for candidate discovery, now applied to every step) and
leaves **Claude blank**, meaning it inherits `~/.claude/settings.json`. Claude is
blank on purpose: its ids are gateway-specific aliases, so a shipped default
would break anyone on a different gateway.

Two ways to pin it. Both are global: one Claude model and one Codex model for
every step of the pipeline.

**Per run** — a flag:

```bash
spotlights-engine --claude-model aws/claude-opus-4-8 --codex-model gpt-5.5 ...
```

**Durably** — edit `models.yaml` inside the installed package, or keep your own
copy anywhere and point `SPOTLIGHTS_MODELS_FILE` at it (the sane option when the
engine is installed as a tool):

```yaml
claude: aws/claude-opus-4-8
codex: gpt-5.5
```

Leave a value blank to inherit that CLI's own default. The whole file is
optional — a missing file, a missing key, and a blank value all mean "inherit".
To force inherit for one run even though the file pins something, pass the flag
empty: `--codex-model ""`.

Precedence, highest first:

1. `--claude-model` / `--codex-model`
2. `SPOTLIGHTS_MODELS_FILE`
3. the bundled `models.yaml`
4. the agent CLI's own config

Two things worth knowing:

- **A resolved model beats `ANTHROPIC_MODEL`.** Once the engine passes `--model`,
  that env var no longer has any effect. It works only while the resolved value
  is blank.
- **Model ids are gateway-specific.** `aws/claude-opus-4-8` is a LiteLLM alias,
  not a portable name. Check what your gateway exposes.

`spotlights-engine doctor` prints the effective model for each CLI and which file
it came from. Each run's `run_manifest.json` records both `models_requested`
(what you asked for) and `models_used` (what the CLIs reported back) — they can
differ when the request is blank.

## Verify

```bash
claude -p "reply with OK"       # Claude Code
codex exec "reply with OK"      # Codex
```
