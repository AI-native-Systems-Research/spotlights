"""The single definition of child-process environment scrubbing.

Historically six modules each carried a near-identical `_clean_env` /
`_DROP_EXACT` / `_DROP_PREFIX`, and two more (`module_deep_research`) did no
scrubbing at all. Five of the six copies did **not** drop
`ANTHROPIC_AUTH_TOKEN`; only `signal_pipeline/claude_subprocess.py` did. When
the engine is invoked from inside a Claude Code session, that token leaks into
the spawned CLI subprocess and overrides keychain credentials → `401 Invalid
bearer token`. This unified `clean_env` adopts the *superset* rule (drops
`ANTHROPIC_AUTH_TOKEN`) so every session gets the fix.
"""

from __future__ import annotations

import os

# Exact env-var names dropped before spawning any Claude/Codex CLI. The
# `ANTHROPIC_AUTH_TOKEN` entry is the auth-leak fix (see module docstring).
DROP_EXACT = frozenset(
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

# Env-var name prefixes dropped wholesale. `SPOTLIGHTS_` is scrubbed so the
# child CLI never inherits our own runtime knobs (e.g. the limiter budget).
DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")


def clean_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with the drop rules applied.

    Drops each name in :data:`DROP_EXACT` and any name starting with one of
    :data:`DROP_PREFIX`. Everything else (PATH, HOME, ANTHROPIC_API_KEY,
    OPENAI_API_KEY, …) is preserved.
    """
    env = os.environ.copy()
    for key in list(env):
        if key in DROP_EXACT or key.startswith(DROP_PREFIX):
            env.pop(key)
    return env


__all__ = ["DROP_EXACT", "DROP_PREFIX", "clean_env"]
