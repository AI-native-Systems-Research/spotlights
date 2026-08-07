"""The single `codex exec` session (canonical `-c model=...` argv form).

This covers the agent_proposals / candidate_discovery shape:
`codex exec - --json --output-last-message <path> --output-schema <path>
--sandbox read-only -C <repo> -c model="…" -c model_reasoning_effort="…"`.

Per the design, the unified builder uses codex's `-c key="value"` config
override form (the richer of the two in-tree variants) and always wires
reasoning effort. The output JSON is read back from `output_last_message`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from spotlights_engine.costing.usage import codex_usage_from_stream
from spotlights_engine.llm_session.env import clean_env
from spotlights_engine.llm_session.resolve import CLIResolutionError, resolve_cli
from spotlights_engine.llm_session.result import SessionResult
from spotlights_engine.llm_session.transport import default_on_event, run_blocking


@dataclass
class CodexSessionOptions:
    repo_path: Path
    output_last_message: Path
    output_schema: Path
    model: str | None = None
    reasoning_effort: str | None = None
    sandbox: str = "read-only"
    schema_text: str | None = None
    timeout_s: int = 600
    codex_bin: str = "codex"
    extra_args: Sequence[str] = field(default_factory=tuple)


class CodexSession:
    name = "codex"
    cli: Literal["codex"] = "codex"

    def __init__(self, options: CodexSessionOptions) -> None:
        self.options = options

    def build_argv(self) -> list[str]:
        opt = self.options
        argv0 = resolve_cli("codex", opt.codex_bin)
        argv: list[str] = [
            argv0,
            "exec",
            "-",
            "--json",
            "--output-last-message",
            str(opt.output_last_message.resolve()),
            "--output-schema",
            str(opt.output_schema.resolve()),
            "--sandbox",
            opt.sandbox,
            "-C",
            str(opt.repo_path),
        ]
        if opt.model is not None:
            argv += ["-c", f'model="{opt.model}"']
        if opt.reasoning_effort is not None:
            argv += ["-c", f'model_reasoning_effort="{opt.reasoning_effort}"']
        argv += list(opt.extra_args)
        return argv

    def run(
        self,
        prompt: str,
        *,
        cwd: Path | str | None = None,
        on_event: Callable[[str], None] | None = default_on_event,  # noqa: ARG002
        log_dir: Path | None = None,  # noqa: ARG002
    ) -> SessionResult:
        opt = self.options
        # Prepare schema + last-message files (per-invocation isolation).
        opt.output_schema.parent.mkdir(parents=True, exist_ok=True)
        if opt.schema_text is not None:
            opt.output_schema.write_text(opt.schema_text, encoding="utf-8")
        opt.output_last_message.parent.mkdir(parents=True, exist_ok=True)
        if opt.output_last_message.exists():
            try:
                opt.output_last_message.unlink()
            except OSError:
                pass

        try:
            argv = self.build_argv()
        except CLIResolutionError as exc:
            return SessionResult(cli=self.cli, returncode=-1, duration_s=0.0, error=str(exc))

        run_cwd = cwd if cwd is not None else opt.repo_path
        res = run_blocking(
            argv=argv,
            prompt=prompt,
            env=clean_env(),
            cwd=run_cwd,
            timeout_s=opt.timeout_s,
        )
        usage = codex_usage_from_stream(res.stdout)
        if usage is not None and usage.model is None and opt.model:
            usage = usage.model_copy(update={"model": opt.model})

        if res.timed_out:
            return SessionResult(
                cli=self.cli,
                returncode=-1,
                duration_s=res.duration_s,
                error=f"codex timed out after {res.duration_s:.1f}s",
                usage=usage,
                stdout=res.stdout,
                stderr=res.stderr,
                timed_out=True,
            )
        if res.returncode != 0:
            stderr_tail = res.stderr[-500:].decode("utf-8", "replace")
            return SessionResult(
                cli=self.cli,
                returncode=res.returncode,
                duration_s=res.duration_s,
                error=f"codex exit={res.returncode}: stderr={stderr_tail!r}",
                usage=usage,
                stdout=res.stdout,
                stderr=res.stderr,
            )

        if not opt.output_last_message.exists():
            return self._err("codex output_last_message file missing", res, usage)
        text = opt.output_last_message.read_text(encoding="utf-8")
        if not text.strip():
            return self._err("codex output_last_message empty", res, usage)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return self._err(f"codex output_last_message not JSON: {exc}", res, usage)
        if not isinstance(parsed, (dict, list)):
            return self._err(
                "codex output_last_message JSON was not an object or array: "
                f"got {type(parsed).__name__}",
                res,
                usage,
            )
        return SessionResult(
            cli=self.cli,
            returncode=res.returncode,
            duration_s=res.duration_s,
            structured_output=parsed,
            final_message=text,
            usage=usage,
            model=(usage.model if usage and usage.model else None) or opt.model,
            cost_usd=usage.cli_reported_cost_usd if usage else None,
            stdout=res.stdout,
            stderr=res.stderr,
        )

    def _err(self, message: str, res, usage) -> SessionResult:
        return SessionResult(
            cli=self.cli,
            returncode=res.returncode,
            duration_s=res.duration_s,
            error=message,
            usage=usage,
            model=(usage.model if usage and usage.model else None) or self.options.model,
            stdout=res.stdout,
            stderr=res.stderr,
        )


__all__ = ["CodexSession", "CodexSessionOptions"]
