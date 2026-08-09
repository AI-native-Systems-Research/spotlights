# Installing the agent CLIs

Spotlights drives two external agent CLIs — `claude` (Claude Code) and `codex` (Codex). Both must be installed and authenticated before a run. This page covers installing them, optionally routing them through a LiteLLM gateway, and a quick response check. For the engine itself, the `doctor` preflight, and the bundled slash commands, see the full [installation guide](INSTALL.md).

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

## Verify

```bash
claude -p "reply with OK"       # Claude Code
codex exec "reply with OK"      # Codex
```
