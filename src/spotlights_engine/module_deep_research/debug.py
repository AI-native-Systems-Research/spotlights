"""Optional per-module debug artifacts for the deep-research step.

When a caller passes a `debug_dir`, step 3 drops the raw inputs and
intermediate outputs there so a run can be inspected after the fact:

- `prompt.md` — the exact prompt rendered for (and sent to) every runner.
- `<agent>.output.md` — each runner's raw response (Codex, Claude, Gemini),
  i.e. the text that gets parsed into findings; failed runners record their
  error instead.
- `merged_before_filter.json` — the deduped, merged finding set as it stood
  BEFORE any paper filter collapsed it to a single finding.

These are diagnostics only; nothing downstream reads them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spotlights_engine.module_deep_research.orchestration import RunnerOutcome
    from spotlights_engine.module_deep_research.validation import AgentFinding

_log = logging.getLogger(__name__)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_prompt(debug_dir: Path, prompt: str) -> None:
    """Persist the rendered prompt shared by every runner."""
    try:
        _write_text(debug_dir / "prompt.md", prompt)
    except OSError as exc:  # diagnostics must never fail the run
        _log.warning("deep_research: failed to write debug prompt: %s", exc)


def write_runner_outputs(debug_dir: Path, outcomes: Sequence[RunnerOutcome]) -> None:
    """Persist each runner's raw output (or error) as `<agent>.output.md`."""
    for outcome in outcomes:
        path = debug_dir / f"{outcome.agent_name}.output.md"
        if outcome.error is not None:
            body = f"# {outcome.agent_name}: FAILED\n\n{outcome.error}\n"
        elif outcome.result is None:
            body = f"# {outcome.agent_name}: no result\n"
        else:
            result = outcome.result
            text = result.final_message or result.stdout or ""
            header = f"# {outcome.agent_name} (exit {result.returncode})\n\n"
            stderr = result.stderr.strip()
            footer = f"\n\n---\nstderr:\n{stderr}\n" if stderr else ""
            body = f"{header}{text}{footer}"
        try:
            _write_text(path, body)
        except OSError as exc:
            _log.warning(
                "deep_research: failed to write debug output for %s: %s",
                outcome.agent_name,
                exc,
            )


def write_merged_before_filter(
    debug_dir: Path, findings: Sequence[AgentFinding]
) -> None:
    """Persist the deduped merged finding set before any paper filter ran."""
    payload = [finding.model_dump(mode="json") for finding in findings]
    try:
        _write_text(
            debug_dir / "merged_before_filter.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    except OSError as exc:
        _log.warning(
            "deep_research: failed to write debug merged findings: %s", exc
        )


__all__ = [
    "write_merged_before_filter",
    "write_prompt",
    "write_runner_outputs",
]
