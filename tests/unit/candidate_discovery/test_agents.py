"""Unit tests for `spotlights_engine.candidate_discovery.agents`.

The runners are now thin adapters over `llm_session`: argv comes from the
centralized sessions (resolved via `llm_session.resolve.resolve_cli`) and the
spawn goes through `llm_session.transport.run_blocking`. Tests patch those
seams. Env-scrubbing is centralized and covered by `tests/unit/llm_session`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from spotlights_engine.candidate_discovery.agents import (
    ClaudeRunner,
    CodexRunner,
    _SchemaParseError,
)
from spotlights_engine.candidate_discovery.api import DiscoveryConfig
from spotlights_engine.candidate_discovery.errors import DiscoverySetupError
from spotlights_engine.llm_session.transport import BlockingResult
from spotlights_engine.schemas.project import File, Module


def _config(tmp_path: Path, **overrides) -> DiscoveryConfig:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_path": repo,
        "module_qualified_name": "v1/foo",
        "module": Module(
            name="foo",
            path="src/foo",
            description="d",
            main_files=[File(path="src/foo/x.py", role="entry")],
        ),
        "artifacts_dir": artifacts,
        "per_iteration_wallclock_s": 5,
    }
    payload.update(overrides)
    return DiscoveryConfig.model_validate(payload)


def _blocking(returncode: int, stdout: bytes, stderr: bytes, *, timed_out: bool = False):
    return BlockingResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        duration_s=1.0,
        timed_out=timed_out,
    )


def test_runner_init_raises_when_executable_missing(tmp_path):
    cfg = _config(tmp_path)
    with patch("spotlights_engine.candidate_discovery.agents.shutil.which",
               return_value=None):
        with pytest.raises(DiscoverySetupError, match="claude"):
            ClaudeRunner(cfg)
        with pytest.raises(DiscoverySetupError, match="codex"):
            CodexRunner(cfg)


def _force_present(monkeypatch):
    # AgentRunner.__init__ checks presence via agents.shutil.which; the session
    # argv builder resolves argv[0] via llm_session.resolve.resolve_cli, which
    # uses its own shutil.which. Patch both.
    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.agents.shutil.which",
        lambda _name: "/usr/local/bin/whatever",
    )
    monkeypatch.setattr(
        "spotlights_engine.llm_session.resolve.shutil.which",
        lambda _name: "/usr/local/bin/whatever",
    )


def test_claude_argv_includes_required_flags(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path, claude_max_turns=12)
    runner = ClaudeRunner(cfg)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}')
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    argv = runner._build_argv(schema_path=schema_path, iter_dir=iter_dir)
    # argv[0] is the executable resolved via the session resolver (mocked), not
    # the bare "claude" name — the cmd shim is deliberately bypassed.
    assert argv[0] == "/usr/local/bin/whatever"
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert argv[argv.index("--json-schema") + 1] == '{"type":"object"}'
    assert argv[argv.index("--permission-mode") + 1] == "plan"
    assert argv[argv.index("--max-turns") + 1] == "12"


def test_codex_argv_includes_required_flags(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path, codex_model="gpt-5.5", codex_reasoning_effort="high")
    runner = CodexRunner(cfg)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}')
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    argv = runner._build_argv(schema_path=schema_path, iter_dir=iter_dir)
    assert argv[0] == "/usr/local/bin/whatever"
    assert argv[1:3] == ["exec", "-"]
    assert "--json" in argv
    assert argv[argv.index("--output-last-message") + 1] == str(iter_dir / "last_message.json")
    assert argv[argv.index("--output-schema") + 1] == str(schema_path)
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("-C") + 1] == str(cfg.repo_path)
    # `-c model="..."` appears twice — model and reasoning_effort.
    c_pairs = [argv[i + 1] for i, x in enumerate(argv) if x == "-c"]
    assert any("model=\"gpt-5.5\"" in p for p in c_pairs)
    assert any("model_reasoning_effort=\"high\"" in p for p in c_pairs)


def test_claude_argv_omits_dash_c_flag(tmp_path, monkeypatch):
    """Plan §17 cwd asymmetry: only Codex carries `-C`; Claude relies on cwd."""
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    argv = runner._build_argv(schema_path=schema_path, iter_dir=iter_dir)
    assert "-C" not in argv


@pytest.mark.parametrize("runner_cls", [ClaudeRunner, CodexRunner])
def test_invoke_passes_cwd_repo_path(runner_cls, tmp_path, monkeypatch):
    """Plan §17 cwd asymmetry: both runners run with `cwd=repo_path`."""
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = runner_cls(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")
    # Codex normally writes last_message.json itself; pre-seed so the post-run
    # parse path is well-defined for both runners.
    (iter_dir / "last_message.json").write_text('{"x": 1}')

    captured: dict = {}

    def fake_run_blocking(**kwargs):
        captured.update(kwargs)
        return _blocking(0, b"", b"")

    with patch("spotlights_engine.llm_session.claude.run_blocking",
               side_effect=fake_run_blocking), \
         patch("spotlights_engine.llm_session.codex.run_blocking",
               side_effect=fake_run_blocking):
        # Claude's no-result-event path raises _SchemaParseError with empty
        # stdout; that's fine for this cwd assertion.
        try:
            runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)
        except _SchemaParseError:
            pass
    assert Path(captured.get("cwd")) == cfg.repo_path


def test_claude_invoke_timeout_raises_schema_parse_error(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    with patch("spotlights_engine.llm_session.claude.run_blocking",
               return_value=_blocking(-1, b"out", b"err", timed_out=True)):
        with pytest.raises(_SchemaParseError, match="timed out"):
            runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)

    assert (iter_dir / "raw_stdout.log").read_bytes() == b"out"
    assert (iter_dir / "raw_stderr.log").read_bytes() == b"err"


def test_claude_invoke_nonzero_exit_raises(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    with patch("spotlights_engine.llm_session.claude.run_blocking",
               return_value=_blocking(7, b"out-boom", b"err-boom")):
        with pytest.raises(_SchemaParseError, match="exit=7") as exc:
            runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)
    # The centralized error message keeps the stderr tail (stdout is dropped).
    assert "err-boom" in str(exc.value)


def test_claude_parse_last_message_round_trip(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    result_payload = json.dumps(
        {"module_qualified_name": "v1/foo", "candidates": [{"id": "cand-0001"}]}
    )
    stdout_lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(
            {
                "type": "result",
                "session_id": "sess-1",
                "duration_ms": 1234,
                "total_cost_usd": 0.04,
                "usage": {"input_tokens": 100, "output_tokens": 200},
                "result": result_payload,
            }
        ),
    ]
    fake_stdout = ("\n".join(stdout_lines) + "\n").encode("utf-8")
    with patch("spotlights_engine.llm_session.claude.run_blocking",
               return_value=_blocking(0, fake_stdout, b"")):
        inv = runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)
    assert inv.session_id == "sess-1"
    assert inv.duration_s == 1.234
    assert inv.cost_usd == 0.04
    assert inv.input_tokens == 100
    assert inv.output_tokens == 200
    assert runner.parse_last_message(iter_dir) == result_payload


def test_claude_invoke_uses_structured_output_when_result_empty(tmp_path, monkeypatch):
    # With --json-schema the CLI returns the parsed object on `structured_output`
    # and leaves `result` as an empty string. The runner must serialize
    # structured_output into last_message.json rather than writing the empty result.
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    structured = {"module_qualified_name": "v1/foo", "candidates": [{"id": "cand-0001"}]}
    stdout_lines = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(
            {
                "type": "result",
                "session_id": "sess-1",
                "duration_ms": 1234,
                "result": "",
                "structured_output": structured,
            }
        ),
    ]
    fake_stdout = ("\n".join(stdout_lines) + "\n").encode("utf-8")
    with patch("spotlights_engine.llm_session.claude.run_blocking",
               return_value=_blocking(0, fake_stdout, b"")):
        runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)
    assert json.loads(runner.parse_last_message(iter_dir)) == structured


def test_claude_invoke_requires_terminal_result_event(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    stdout_lines = [
        json.dumps({"type": "result", "result": "{}"}),
        json.dumps({"type": "assistant", "message": "trailing event"}),
    ]
    fake_stdout = ("\n".join(stdout_lines) + "\n").encode()
    with patch("spotlights_engine.llm_session.claude.run_blocking",
               return_value=_blocking(0, fake_stdout, b"")):
        with pytest.raises(_SchemaParseError, match="terminal result"):
            runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)
    assert (iter_dir / "last_message.json").read_text() == ""


def test_claude_invoke_rejects_non_ndjson_stdout(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    with patch("spotlights_engine.llm_session.claude.run_blocking",
               return_value=_blocking(0, b"not-json\n", b"")):
        with pytest.raises(_SchemaParseError, match="not JSON"):
            runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)


def test_claude_parse_last_message_missing_raises(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    with pytest.raises(_SchemaParseError, match="missing"):
        runner.parse_last_message(iter_dir)


def test_claude_parse_last_message_empty_raises(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    (iter_dir / "last_message.json").write_text("")
    with pytest.raises(_SchemaParseError, match="empty"):
        runner.parse_last_message(iter_dir)


def test_claude_parse_last_message_non_json_raises(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    (iter_dir / "last_message.json").write_text("not json")
    with pytest.raises(_SchemaParseError, match="not JSON"):
        runner.parse_last_message(iter_dir)


def test_codex_parse_last_message_handles_missing(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = CodexRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    with pytest.raises(_SchemaParseError, match="missing"):
        runner.parse_last_message(iter_dir)


def test_invoke_retry_appends_with_separator(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    with patch("spotlights_engine.llm_session.claude.run_blocking",
               return_value=_blocking(7, b"a", b"b")):
        with pytest.raises(_SchemaParseError):
            runner.invoke(prompt="p1", iter_dir=iter_dir, schema_path=schema_path)
        with pytest.raises(_SchemaParseError):
            runner.invoke(prompt="p2", iter_dir=iter_dir, schema_path=schema_path)

    stdout_log = (iter_dir / "raw_stdout.log").read_bytes()
    assert b"--- retry separator ---" in stdout_log
    assert stdout_log.startswith(b"a")
