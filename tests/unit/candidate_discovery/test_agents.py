"""Unit tests for `spotlights_engine.candidate_discovery.agents`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from spotlights_engine.candidate_discovery.agents import (
    ClaudeRunner,
    CodexRunner,
    _clean_env,
    _SchemaParseError,
)
from spotlights_engine.candidate_discovery.api import DiscoveryConfig
from spotlights_engine.candidate_discovery.errors import DiscoverySetupError
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


def test_clean_env_strips_exact_drops(monkeypatch):
    leaked = {
        "OPENAI_BASE_URL": "x",
        "OPENAI_API_BASE": "x",
        "ANTHROPIC_BASE_URL": "x",
        "HTTP_PROXY": "x",
        "HTTPS_PROXY": "x",
        "ALL_PROXY": "x",
        "VIRTUAL_ENV": "x",
    }
    for k, v in leaked.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/me")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    monkeypatch.setenv("OPENAI_API_KEY", "ok")

    env = _clean_env()

    for k in leaked:
        assert k not in env
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/me"
    assert env["ANTHROPIC_API_KEY"] == "ak"
    assert env["OPENAI_API_KEY"] == "ok"


def test_clean_env_strips_prefixes(monkeypatch):
    monkeypatch.setenv("VSCODE_PID", "1")
    monkeypatch.setenv("VSCODE_IPC_HOOK", "x")
    monkeypatch.setenv("OPTQUEST_RUN_ID", "x")
    monkeypatch.setenv("SPOTLIGHTS_SESSION", "x")
    monkeypatch.setenv("KEEP_ME", "y")

    env = _clean_env()
    assert "VSCODE_PID" not in env
    assert "VSCODE_IPC_HOOK" not in env
    assert "OPTQUEST_RUN_ID" not in env
    assert "SPOTLIGHTS_SESSION" not in env
    assert env["KEEP_ME"] == "y"


def test_runner_init_raises_when_executable_missing(tmp_path):
    cfg = _config(tmp_path)
    with patch("spotlights_engine.candidate_discovery.agents.shutil.which",
               return_value=None):
        with pytest.raises(DiscoverySetupError, match="claude"):
            ClaudeRunner(cfg)
        with pytest.raises(DiscoverySetupError, match="codex"):
            CodexRunner(cfg)


def _force_present(monkeypatch):
    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.agents.shutil.which",
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
    # argv[0] is the executable resolved via shutil.which (mocked above), not the
    # bare "claude" name — the cmd shim is deliberately bypassed.
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
    # argv[0] is the executable resolved via shutil.which (mocked above), not the
    # bare "codex" name.
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
    """Plan §17 cwd asymmetry: both runners receive `cwd=repo_path`."""
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

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               side_effect=fake_run):
        # Both _parse_invocation_metadata implementations tolerate empty stdout
        # by returning None-valued fields; that's fine for this assertion.
        try:
            runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)
        except _SchemaParseError:
            pass  # Claude's no-result-event path is irrelevant here.
    assert captured.get("cwd") == str(cfg.repo_path)


def test_claude_invoke_timeout_raises_schema_parse_error(tmp_path, monkeypatch):
    _force_present(monkeypatch)
    cfg = _config(tmp_path)
    runner = ClaudeRunner(cfg)
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1, output=b"out", stderr=b"err")

    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               side_effect=fake_run):
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

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=7, stdout=b"out-boom", stderr=b"err-boom"
    )
    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               return_value=fake_result):
        with pytest.raises(_SchemaParseError, match="exit=7") as exc:
            runner.invoke(prompt="hi", iter_dir=iter_dir, schema_path=schema_path)
    assert "err-boom" in str(exc.value)
    assert "out-boom" in str(exc.value)


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
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=fake_stdout, stderr=b""
    )
    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               return_value=fake_result):
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
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=fake_stdout, stderr=b""
    )
    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               return_value=fake_result):
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
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=("\n".join(stdout_lines) + "\n").encode(), stderr=b""
    )
    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               return_value=fake_result):
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

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"not-json\n", stderr=b""
    )
    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               return_value=fake_result):
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

    fake_result = subprocess.CompletedProcess(
        args=[], returncode=7, stdout=b"a", stderr=b"b"
    )
    with patch("spotlights_engine.candidate_discovery.agents.subprocess.run",
               return_value=fake_result):
        with pytest.raises(_SchemaParseError):
            runner.invoke(prompt="p1", iter_dir=iter_dir, schema_path=schema_path)
        with pytest.raises(_SchemaParseError):
            runner.invoke(prompt="p2", iter_dir=iter_dir, schema_path=schema_path)

    stdout_log = (iter_dir / "raw_stdout.log").read_bytes()
    assert b"--- retry separator ---" in stdout_log
    assert stdout_log.startswith(b"a")
