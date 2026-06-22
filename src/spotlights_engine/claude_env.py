"""Provider-neutral Claude Code subprocess environment helpers.

The engine usually scrubs inherited gateway/proxy variables before spawning
Claude so a parent agent session cannot accidentally leak credentials into a
child process. These helpers keep that safe default while allowing callers to
opt in to an explicit Claude gateway through `SPOTLIGHTS_CLAUDE_*` directives.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

DEFAULT_DROP_EXACT = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "VIRTUAL_ENV",
    }
)
DEFAULT_DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_claude_env(
    *,
    base_env: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
    drop_exact: frozenset[str] = DEFAULT_DROP_EXACT,
    drop_prefix: Sequence[str] = DEFAULT_DROP_PREFIX,
) -> dict[str, str]:
    """Return the environment for a Claude subprocess.

    Safe default: remove inherited proxy/auth variables that commonly break
    nested Claude invocations. Explicit opt-in directives are then applied from
    the *original* environment:

    - `SPOTLIGHTS_CLAUDE_BASE_URL` -> `ANTHROPIC_BASE_URL`
    - `SPOTLIGHTS_CLAUDE_AUTH_TOKEN_ENV` -> copy that env var to
      `ANTHROPIC_AUTH_TOKEN`
    - `SPOTLIGHTS_CLAUDE_AUTH_TOKEN` -> direct `ANTHROPIC_AUTH_TOKEN`
    - `SPOTLIGHTS_CLAUDE_UNSET_ENV` -> comma-separated variables to remove
    - `SPOTLIGHTS_CLAUDE_DISABLE_EXPERIMENTAL_BETAS=1` -> disable betas

    When an auth-token directive is used, `ANTHROPIC_API_KEY` is removed unless
    `SPOTLIGHTS_CLAUDE_PRESERVE_ANTHROPIC_API_KEY=1` is set. This mirrors the
    common `env -u ANTHROPIC_API_KEY ... ANTHROPIC_AUTH_TOKEN=... claude` proxy
    pattern without baking in any provider-specific endpoint.
    """
    source = dict(os.environ if base_env is None else base_env)
    env = dict(source)
    for key in list(env):
        if key in drop_exact or key.startswith(tuple(drop_prefix)):
            env.pop(key, None)

    if extra_env:
        env.update(dict(extra_env))

    unset_names = set(_csv(source.get("SPOTLIGHTS_CLAUDE_UNSET_ENV")))
    auth_token_env = source.get("SPOTLIGHTS_CLAUDE_AUTH_TOKEN_ENV")
    direct_auth_token = source.get("SPOTLIGHTS_CLAUDE_AUTH_TOKEN")
    if (auth_token_env or direct_auth_token) and not _truthy(
        source.get("SPOTLIGHTS_CLAUDE_PRESERVE_ANTHROPIC_API_KEY")
    ):
        unset_names.add("ANTHROPIC_API_KEY")
    for name in unset_names:
        env.pop(name, None)

    base_url = source.get("SPOTLIGHTS_CLAUDE_BASE_URL")
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url

    if auth_token_env and source.get(auth_token_env):
        env["ANTHROPIC_AUTH_TOKEN"] = source[auth_token_env]
    if direct_auth_token:
        env["ANTHROPIC_AUTH_TOKEN"] = direct_auth_token

    if _truthy(source.get("SPOTLIGHTS_CLAUDE_DISABLE_EXPERIMENTAL_BETAS")):
        env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"

    return env


def claude_model_args(*, base_env: Mapping[str, str] | None = None) -> list[str]:
    """Return `--model ...` args from `SPOTLIGHTS_CLAUDE_MODEL`, if set."""
    source = os.environ if base_env is None else base_env
    model = source.get("SPOTLIGHTS_CLAUDE_MODEL")
    return ["--model", model] if model else []


__all__ = ["build_claude_env", "claude_model_args"]
