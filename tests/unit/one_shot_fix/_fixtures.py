"""Fakes for one_shot_fix tests.

The `claude` runner is exercised against a fake executable on PATH rather than
a monkeypatched function, so the real subprocess/argv/stream-json path is under
test. Result payloads follow the same terminal-event shape
`claude_usage_from_stream` parses (see `costing/usage.py`).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

RESULT_EVENT = {
    "type": "result",
    "subtype": "success",
    "duration_api_ms": 1200,
    "total_cost_usd": 0.0123,
    "model": "claude-sonnet-5",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
    },
}

# Edits a file in $PWD (the worktree), then emits the terminal result event.
FAKE_CLAUDE_SUCCESS = f"""#!/bin/sh
printf '# touched by fake claude\\n' >> pkg/attn/tile.py
printf '%s\\n' '{json.dumps(RESULT_EVENT)}'
exit 0
"""

FAKE_CLAUDE_FAILURE = """#!/bin/sh
echo "boom: model unavailable" >&2
exit 3
"""

FAKE_CLAUDE_NO_RESULT_EVENT = """#!/bin/sh
printf '%s\\n' '{"type":"system","subtype":"init"}'
exit 0
"""

# Records argv so the test can assert on the flags passed to claude.
FAKE_CLAUDE_ARGV_RECORDER = f"""#!/bin/sh
printf '%s\\n' "$@" > "$TEST_ARGV_FILE"
printf '%s\\n' '{json.dumps(RESULT_EVENT)}'
exit 0
"""


def write_fake_claude(bin_dir: Path, *, script: str) -> Path:
    """Write an executable `claude` shim into `bin_dir` and return its path."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "claude"
    fake.write_text(script, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return fake


def prepend_to_path(monkeypatch, bin_dir: Path) -> None:
    """Put `bin_dir` first on PATH for the duration of a test."""
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


__all__ = [
    "FAKE_CLAUDE_ARGV_RECORDER",
    "FAKE_CLAUDE_FAILURE",
    "FAKE_CLAUDE_NO_RESULT_EVENT",
    "FAKE_CLAUDE_SUCCESS",
    "RESULT_EVENT",
    "prepend_to_path",
    "write_fake_claude",
]
