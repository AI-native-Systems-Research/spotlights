"""The single `claude -p` session: one argv builder, one spawn+parse core."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from spotlights_engine.costing.usage import (
    claude_usage_from_payload,
    claude_usage_from_stream,
)
from spotlights_engine.llm_session.env import clean_env
from spotlights_engine.llm_session.resolve import CLIResolutionError, resolve_cli
from spotlights_engine.llm_session.result import SessionResult
from spotlights_engine.llm_session.transport import (
    ResultEventError,
    default_on_event,
    extract_result_event,
    final_message_text,
    model_from_init,
    run_blocking,
    run_streaming_claude,
)


@dataclass
class ClaudeSessionOptions:
    """All behavioural axes of a Claude session, as parameters (not forks)."""

    model: str | None = None
    permission_mode: str = "plan"
    allowed_tools: Sequence[str] | None = None
    max_turns: int = 30
    json_schema: str | None = None
    output_format: Literal["stream-json", "json"] = "stream-json"
    stream_events: bool = True
    timeout_s: int = 1800
    claude_bin: str = "claude"
    extra_args: Sequence[str] = field(default_factory=tuple)
    verbose: bool = True


class ClaudeSession:
    name = "claude"
    cli: Literal["claude"] = "claude"

    def __init__(self, options: ClaudeSessionOptions | None = None) -> None:
        self.options = options or ClaudeSessionOptions()

    def build_argv(self) -> list[str]:
        opt = self.options
        argv0 = resolve_cli("claude", opt.claude_bin)
        argv: list[str] = [
            argv0,
            "-p",
            "--output-format",
            opt.output_format,
        ]
        if opt.verbose and opt.output_format == "stream-json":
            argv.append("--verbose")
        if opt.json_schema is not None:
            argv += ["--json-schema", opt.json_schema]
        argv += ["--permission-mode", opt.permission_mode]
        argv += ["--max-turns", str(opt.max_turns)]
        if opt.model is not None:
            argv += ["--model", opt.model]
        if opt.allowed_tools is not None:
            argv += ["--allowed-tools", ",".join(opt.allowed_tools)]
        argv += list(opt.extra_args)
        return argv

    def run(
        self,
        prompt: str,
        *,
        cwd: Path | str | None,
        on_event: Callable[[str], None] | None = default_on_event,
        log_dir: Path | None = None,
    ) -> SessionResult:
        opt = self.options
        try:
            # Resolve early so a missing/shim CLI fails before we spawn or
            # write log files; the streaming/blocking paths rebuild argv.
            self.build_argv()
        except CLIResolutionError as exc:
            return SessionResult(
                cli=self.cli, returncode=-1, duration_s=0.0, error=str(exc)
            )

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "prompt.md").write_text(prompt, encoding="utf-8")
            if opt.json_schema is not None:
                (log_dir / "schema.json").write_text(opt.json_schema, encoding="utf-8")

        if opt.stream_events and opt.output_format == "stream-json":
            return self._run_streaming(prompt, cwd=cwd, on_event=on_event, log_dir=log_dir)
        return self._run_blocking(prompt, cwd=cwd, log_dir=log_dir)

    # ── streaming path ────────────────────────────────────────────────
    def _run_streaming(
        self,
        prompt: str,
        *,
        cwd: Path | str | None,
        on_event: Callable[[str], None] | None,
        log_dir: Path | None,
    ) -> SessionResult:
        from spotlights_engine.llm_session.transport import StreamingTimeout

        argv = self.build_argv()
        try:
            streamed = run_streaming_claude(
                argv=argv,
                prompt=prompt,
                env=clean_env(),
                cwd=cwd,
                timeout_s=self.options.timeout_s,
                on_event=on_event,
            )
        except StreamingTimeout as exc:
            if log_dir is not None:
                _persist_streams(log_dir, exc.stdout, exc.stderr)
            return SessionResult(
                cli=self.cli,
                returncode=-1,
                duration_s=exc.duration_s,
                error=f"claude timed out after {exc.duration_s:.1f}s",
                usage=claude_usage_from_stream(exc.stdout),
                stdout=exc.stdout,
                stderr=exc.stderr,
                timed_out=True,
            )
        if log_dir is not None:
            _persist_streams(log_dir, streamed.stdout, streamed.stderr)
        return self._parse(
            stdout=streamed.stdout,
            stderr=streamed.stderr,
            returncode=streamed.returncode,
            duration_s=streamed.duration_s,
        )

    # ── blocking path ─────────────────────────────────────────────────
    def _run_blocking(
        self,
        prompt: str,
        *,
        cwd: Path | str | None,
        log_dir: Path | None,
    ) -> SessionResult:
        argv = self.build_argv()
        res = run_blocking(
            argv=argv,
            prompt=prompt,
            env=clean_env(),
            cwd=cwd,
            timeout_s=self.options.timeout_s,
        )
        if log_dir is not None:
            _persist_streams(log_dir, res.stdout, res.stderr)
        if res.timed_out:
            return SessionResult(
                cli=self.cli,
                returncode=-1,
                duration_s=res.duration_s,
                error=f"claude timed out after {res.duration_s:.1f}s",
                usage=claude_usage_from_stream(res.stdout),
                stdout=res.stdout,
                stderr=res.stderr,
                timed_out=True,
            )
        if self.options.output_format == "json":
            return self._parse_plain_json(res)
        return self._parse(
            stdout=res.stdout,
            stderr=res.stderr,
            returncode=res.returncode,
            duration_s=res.duration_s,
        )

    # ── shared stream-json parse ──────────────────────────────────────
    def _parse(
        self, *, stdout: bytes, stderr: bytes, returncode: int, duration_s: float
    ) -> SessionResult:
        usage = claude_usage_from_stream(stdout)
        model = model_from_init(stdout)
        if returncode != 0:
            stderr_tail = stderr[-500:].decode("utf-8", "replace")
            return SessionResult(
                cli=self.cli,
                returncode=returncode,
                duration_s=duration_s,
                error=f"claude exit={returncode}: stderr={stderr_tail!r}",
                usage=usage,
                model=model,
                stdout=stdout,
                stderr=stderr,
            )
        try:
            result_event = extract_result_event(stdout)
        except ResultEventError as exc:
            return SessionResult(
                cli=self.cli,
                returncode=returncode,
                duration_s=duration_s,
                error=str(exc),
                usage=usage,
                model=model,
                stdout=stdout,
                stderr=stderr,
            )
        if result_event is None:
            return SessionResult(
                cli=self.cli,
                returncode=returncode,
                duration_s=duration_s,
                error="claude stream-json had no terminal result event",
                usage=usage,
                model=model,
                stdout=stdout,
                stderr=stderr,
            )
        return self._from_result_event(
            result_event, returncode=returncode, duration_s=duration_s,
            stdout=stdout, stderr=stderr, usage=usage, model=model,
        )

    def _parse_plain_json(self, res) -> SessionResult:
        # `--output-format json`: a single top-level JSON payload with the same
        # fields as the terminal result event.
        import json

        usage_payload = claude_usage_from_stream(res.stdout)  # tolerant on non-stream
        text = res.stdout.decode("utf-8", "replace").strip()
        payload: dict | None = None
        if text:
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    payload = obj
            except json.JSONDecodeError:
                payload = None
        if res.returncode != 0:
            stderr_tail = res.stderr[-500:].decode("utf-8", "replace")
            return SessionResult(
                cli=self.cli,
                returncode=res.returncode,
                duration_s=res.duration_s,
                error=f"claude exit={res.returncode}: stderr={stderr_tail!r}",
                usage=usage_payload or (claude_usage_from_payload(payload) if payload else None),
                stdout=res.stdout,
                stderr=res.stderr,
            )
        if payload is None:
            return SessionResult(
                cli=self.cli,
                returncode=res.returncode,
                duration_s=res.duration_s,
                error="claude --output-format json produced no JSON payload",
                stdout=res.stdout,
                stderr=res.stderr,
            )
        usage = claude_usage_from_payload(payload)
        if usage is not None and usage.model is None and self.options.model:
            usage = usage.model_copy(update={"model": self.options.model})
        return self._from_result_event(
            payload, returncode=res.returncode, duration_s=res.duration_s,
            stdout=res.stdout, stderr=res.stderr, usage=usage,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        )

    def _from_result_event(
        self, ev: dict, *, returncode: int, duration_s: float,
        stdout: bytes, stderr: bytes, usage, model,
    ) -> SessionResult:
        cost = ev.get("total_cost_usd")
        cost = cost if isinstance(cost, (int, float)) else None
        num_turns = ev.get("num_turns")
        num_turns = num_turns if isinstance(num_turns, int) else None
        try:
            from spotlights_engine.llm_session.transport import (
                structured_output_from_result,
            )

            structured = structured_output_from_result(ev)
        except ResultEventError as exc:
            return SessionResult(
                cli=self.cli, returncode=returncode, duration_s=duration_s,
                error=str(exc), usage=usage, model=model,
                cost_usd=cost, num_turns=num_turns,
                stdout=stdout, stderr=stderr,
            )
        final = final_message_text(ev)
        return SessionResult(
            cli=self.cli,
            returncode=returncode,
            duration_s=duration_s,
            structured_output=structured,
            final_message=final or None,
            usage=usage,
            model=(usage.model if usage and usage.model else None) or model,
            cost_usd=cost,
            num_turns=num_turns,
            session_id=ev.get("session_id"),
            stdout=stdout,
            stderr=stderr,
        )


def _persist_streams(log_dir: Path, stdout: bytes, stderr: bytes) -> None:
    (log_dir / "raw_stdout.log").write_bytes(stdout or b"")
    (log_dir / "raw_stderr.log").write_bytes(stderr or b"")


__all__ = ["ClaudeSession", "ClaudeSessionOptions"]
