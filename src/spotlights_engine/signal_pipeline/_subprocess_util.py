"""Re-export shim: the streaming primitives now live in `llm_session.transport`.

Kept so existing imports (`event_formatter`, tests, and any external code) keep
working during and after the migration to the centralized `llm_session`
package. New code should import from `spotlights_engine.llm_session` directly.
"""

from __future__ import annotations

from spotlights_engine.llm_session.transport import (
    ClaudeResolutionError,
    StreamingResult,
    StreamingTimeout,
    default_on_event,
    kill_tree,
    resolve_claude_argv0,
    run_streaming_claude,
    summarize_event,
)

__all__ = [
    "ClaudeResolutionError",
    "StreamingResult",
    "StreamingTimeout",
    "default_on_event",
    "kill_tree",
    "resolve_claude_argv0",
    "run_streaming_claude",
    "summarize_event",
]
