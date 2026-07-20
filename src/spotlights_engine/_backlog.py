"""Shared backlog dumper. No-op unless SPOTLIGHTS_BACKLOG_DIR is set."""
from __future__ import annotations

import os
import time
from pathlib import Path


def enabled() -> bool:
    return os.environ.get("SPOTLIGHTS_BACKLOG", "").lower() in ("1", "true", "yes", "on")


def dump(step: str, cmd, rc: int, stdout: str | None, stderr: str | None) -> None:
    if not enabled():
        return
    root = os.environ.get("SPOTLIGHTS_BACKLOG_DIR")
    if not root:
        return
    head = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else (str(cmd).split()[0] if cmd else "unknown")
    agent = Path(head).name
    d = Path(root) / step / agent
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log").write_text(
        f"=== step={step} agent={agent} rc={rc} ===\n$ {cmd}\n\n"
        f"[stdout]\n{stdout or ''}\n\n[stderr]\n{stderr or ''}\n"
    )
