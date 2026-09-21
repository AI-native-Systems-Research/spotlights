"""Unit tests for the module deep-research API."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.module_deep_research.api import research_module, resolve_target_module
from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient
from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecResult
from spotlights_engine.module_deep_research.orchestration import select_runners
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput
from spotlights_engine.schemas.project import Module, ProjectTree, Repository


class FakeRunner:
    def __init__(self, final_message: str, returncode: int = 0, stderr: str = "") -> None:
        self.final_message = final_message
        self.returncode = returncode
        self.stderr = stderr
        self.prompts: list[str] = []

    def run(self, prompt: str, *, check: bool = True) -> CodexExecResult:
        self.prompts.append(prompt)
        return CodexExecResult(
            command=["codex", "exec"],
            returncode=self.returncode,
            stdout="",
            stderr=self.stderr,
            final_message=self.final_message,
            output_last_message=None,
        )


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="Demo repository.", source_root="src"),
        modules=[
            Module(
                name="inference",
                path="src/inference",
                submodules=[
                    Module(
                        name="attention",
                        path="src/inference/attention",
                        description="Attention implementation.",
                    )
                ],
            )
        ],
    )


def _request(module_qualified_name: str = "inference/attention") -> ModuleDeepResearchInput:
    return ModuleDeepResearchInput(
        project_tree=_tree(),
        module_qualified_name=module_qualified_name,
        context=SpotlightContext(objective="reduce latency"),
        repo_path=Path("/tmp/example-repo"),
        max_findings_per_module=5,
    )


def _request_with_cap(max_findings_per_module: int) -> ModuleDeepResearchInput:
    return _request().model_copy(
        update={"max_findings_per_module": max_findings_per_module}
    )


def _candidate(cid: str, symbol: str):
    from spotlights_engine.schemas.candidate import Candidate

    return Candidate(
        id=cid,
        module_qualified_name="inference/attention",
        origin="code_agent",
        locations=[
            {
                "file": "src/inference/attention.py",
                "spans": [
                    {"line_start": 1, "line_end": 9, "symbol": symbol, "kind": "function"}
                ],
            }
        ],
        description=f"Optimize {symbol}.",
        current_approach="Naive implementation.",
        evolve_rationale="Hot on the decode path.",
        estimated_impact="high",
        estimated_impact_explanation="Dominates latency.",
    )


def test_research_module_per_candidate_runs_one_pass_per_candidate() -> None:
    runner = FakeRunner('{"findings": [], "issues": []}')
    request = _request().model_copy(
        update={
            "per_candidate_deep_research": True,
            "candidates": [
                _candidate("cand-x-0001", "softmax"),
                _candidate("cand-x-0002", "matmul"),
            ],
        }
    )

    research_module(request, runner=runner)

    # One scoped research pass per candidate; each prompt names only its own.
    assert len(runner.prompts) == 2
    assert "softmax" in runner.prompts[0] and "matmul" not in runner.prompts[0]
    assert "matmul" in runner.prompts[1] and "softmax" not in runner.prompts[1]


def test_research_module_per_candidate_noop_without_candidates() -> None:
    runner = FakeRunner('{"findings": [], "issues": []}')
    request = _request().model_copy(update={"per_candidate_deep_research": True})

    research_module(request, runner=runner)

    assert len(runner.prompts) == 1


def test_research_module_per_candidate_stamps_and_dedups_per_candidate() -> None:
    # The same runner replies with the same paper for every prompt, so each
    # candidate researches the identical work. Per-candidate dedup keeps one
    # scoped finding per candidate (not a single global one), each tagged with
    # its own candidate_id and given a globally unique renumbered finding id.
    runner = FakeRunner(_payload("Shared paper", "https://example.com/shared"))
    request = _request().model_copy(
        update={
            "per_candidate_deep_research": True,
            "candidates": [
                _candidate("cand-x-0001", "softmax"),
                _candidate("cand-x-0002", "matmul"),
            ],
        }
    )

    output = research_module(request, runner=runner)

    assert [f.candidate_id for f in output.findings] == ["cand-x-0001", "cand-x-0002"]
    assert [f.finding_id for f in output.findings] == [
        "find-inference_attention-0001",
        "find-inference_attention-0002",
    ]
    assert all(f.title == "Shared paper" for f in output.findings)


def test_research_module_module_wide_findings_have_no_candidate_id() -> None:
    runner = FakeRunner(_payload("Module paper", "https://example.com/module"))

    output = research_module(_request(), runner=runner)

    assert output.findings[0].candidate_id is None


def test_resolve_target_module_by_slash_qualified_name() -> None:
    module = resolve_target_module(_tree(), "inference/attention")

    assert module is not None
    assert module.path == "src/inference/attention"


def test_research_module_runs_agent_and_returns_parsed_output() -> None:
    runner = FakeRunner(
        """
        {
          "findings": [
            {
              "finding_id": "find-0007",
              "title": "Flash attention tiling",
              "url": "https://example.com/flash",
              "source_type": "paper",
              "technique_summary": "Tiling attention reduces memory traffic."
            }
          ],
          "issues": []
        }
        """
    )

    output = research_module(_request(), runner=runner)

    assert len(runner.prompts) == 1
    assert "Attention implementation." in runner.prompts[0]
    # Finding ids are renumbered + prefixed with the module segment (D3); the
    # default segment is the slug of `inference/attention`.
    assert output.findings[0].finding_id == "find-inference_attention-0001"
    assert output.findings[0].title == "Flash attention tiling"
    assert output.issues == []


def test_research_module_missing_module_returns_unrecoverable_issue() -> None:
    output = research_module(_request("missing.module"), runner=FakeRunner("{}"))

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].recoverable is False
    assert "missing.module" in output.issues[0].message


def test_research_module_records_nonzero_runner_exit_as_recoverable_issue() -> None:
    runner = FakeRunner('{"findings": [], "issues": []}', returncode=2, stderr="network down")

    output = research_module(_request(), runner=runner)

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].recoverable is True
    assert "network down" in output.issues[0].message


def test_codex_command_shape_places_top_level_flags_before_exec(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecOptions

    cmd, _ = CodexExecClient(
        CodexExecOptions(
            cwd=tmp_path,
            profile="gpt55",
            approval="never",
            search=True,
            output_last_message=tmp_path / "last.md",
        )
    ).build_command("-")

    # cmd[0] is the resolved path on Windows (e.g. C:\...\codex.CMD)
    # and the literal "codex" on POSIX where shutil.which falls through.
    # Either way the basename, stripped of any extension, must be "codex".
    assert Path(cmd[0]).stem.lower() == "codex"
    assert cmd.index("--profile") < cmd.index("exec")
    assert cmd.index("--ask-for-approval") < cmd.index("exec")
    assert cmd.index("--search") < cmd.index("exec")
    assert cmd.index("--sandbox") > cmd.index("exec")
    assert cmd[-1] == "-"


def _payload(title: str, url: str) -> str:
    return f'''
    {{
      "findings": [
        {{
          "finding_id": "find-9999",
          "title": "{title}",
          "url": "{url}",
          "source_type": "paper",
          "technique_summary": "Useful transfer idea."
        }}
      ],
      "issues": []
    }}
    '''


class NamedFakeRunner(FakeRunner):
    def __init__(self, name: str, final_message: str, returncode: int = 0) -> None:
        super().__init__(final_message, returncode=returncode)
        self.name = name


def test_research_module_runs_multiple_runners_and_dedups_outputs() -> None:
    codex = NamedFakeRunner("codex", _payload("PagedAttention", "https://arxiv.org/abs/2309.06180"))
    claude = NamedFakeRunner(
        "claude", _payload("PagedAttention", "https://arxiv.org/pdf/2309.06180v2.pdf")
    )
    extra = NamedFakeRunner(
        "extra", _payload("vAttention", "https://arxiv.org/abs/2405.04437")
    )

    output = research_module(_request(), runners=[codex, claude, extra])

    assert [len(r.prompts) for r in (codex, claude, extra)] == [1, 1, 1]
    assert [finding.finding_id for finding in output.findings] == [
        "find-inference_attention-0001",
        "find-inference_attention-0002",
    ]
    assert [finding.title for finding in output.findings] == ["PagedAttention", "vAttention"]
    assert output.issues == []


def test_research_module_dedups_exact_title_matches_with_different_urls() -> None:
    first = NamedFakeRunner("codex", _payload("Cache eviction", "https://example.com/paper-a"))
    second = NamedFakeRunner("claude", _payload("Cache eviction", "https://example.com/paper-b"))

    output = research_module(_request(), runners=[first, second])

    assert [finding.url for finding in output.findings] == ["https://example.com/paper-a"]


def test_research_module_parallel_runner_pool() -> None:
    import threading

    barrier = threading.Barrier(3)

    class BarrierRunner(NamedFakeRunner):
        def run(self, prompt: str, *, check: bool = True) -> CodexExecResult:
            barrier.wait(timeout=1)
            return super().run(prompt, check=check)

    runners = [
        BarrierRunner("codex", _payload("A", "https://example.com/a")),
        BarrierRunner("claude", _payload("B", "https://example.com/b")),
        BarrierRunner("extra", _payload("C", "https://example.com/c")),
    ]

    output = research_module(_request(), runners=runners)

    assert [finding.title for finding in output.findings] == ["A", "B", "C"]
    assert output.issues == []


def test_research_module_treats_finding_cap_as_per_runner() -> None:
    runners = [
        NamedFakeRunner("codex", _payload("A", "https://example.com/a")),
        NamedFakeRunner("claude", _payload("B", "https://example.com/b")),
        NamedFakeRunner("extra", _payload("C", "https://example.com/c")),
    ]

    output = research_module(_request_with_cap(1), runners=runners)

    assert [finding.finding_id for finding in output.findings] == [
        "find-inference_attention-0001",
        "find-inference_attention-0002",
        "find-inference_attention-0003",
    ]
    assert [finding.title for finding in output.findings] == ["A", "B", "C"]


def test_claude_default_max_turns_is_40() -> None:
    from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient

    cmd = ClaudeExecClient().build_command()

    assert cmd[cmd.index("--max-turns") + 1] == "40"


def test_claude_command_shape_uses_litellm_safe_research_tools(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.claude_exec import (
        ClaudeExecClient,
        ClaudeExecOptions,
    )

    cmd = ClaudeExecClient(
        ClaudeExecOptions(cwd=tmp_path, model="claude-opus-4-7", max_turns=3)
    ).build_command()

    assert cmd[:4] == ["claude", "-p", "--output-format", "json"]
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
    assert cmd[cmd.index("--model") + 1] == "claude-opus-4-7"
    assert cmd[cmd.index("--max-turns") + 1] == "3"
    tools = cmd[cmd.index("--tools") + 1]
    allowed_tools = cmd[cmd.index("--allowedTools") + 1]
    assert "WebFetch" in tools
    assert "WebFetch" in allowed_tools
    assert "WebSearch" in tools
    assert "WebSearch" in allowed_tools
    assert "Edit" not in tools


def test_cli_resolution_preserves_posix_command_shape(monkeypatch) -> None:
    from spotlights_engine.module_deep_research import agent_exec

    monkeypatch.setattr(agent_exec, "WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION", False)

    assert agent_exec.resolve_cli_executable("claude") == "claude"


def test_cli_resolution_uses_windows_cmd_shims(monkeypatch) -> None:
    from spotlights_engine.module_deep_research import agent_exec
    from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient
    from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecOptions

    def fake_which(executable: str) -> str:
        return f"C:/Users/example/AppData/Roaming/npm/{executable}.CMD"

    monkeypatch.setattr(agent_exec, "WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION", True)
    monkeypatch.setattr(agent_exec.shutil, "which", fake_which)

    assert agent_exec.resolve_cli_executable("claude") == (
        "C:/Users/example/AppData/Roaming/npm/claude.CMD"
    )
    assert ClaudeExecClient().build_command()[0].endswith("/claude.CMD")
    codex_cmd, _ = CodexExecClient(CodexExecOptions(output_last_message="last.md")).build_command()
    assert codex_cmd[0].endswith("/codex.CMD")


def test_select_runners_default_is_codex_only(tmp_path: Path) -> None:
    runners = select_runners(
        repo_path=tmp_path,
        codex_options=None,
        runner=None,
        runners=None,
    )

    assert len(runners) == 1
    assert isinstance(runners[0], CodexExecClient)


def test_select_runners_enable_claude_search_adds_claude(tmp_path: Path) -> None:
    runners = select_runners(
        repo_path=tmp_path,
        codex_options=None,
        runner=None,
        runners=None,
        enable_claude_search=True,
    )

    assert len(runners) == 2
    assert isinstance(runners[0], CodexExecClient)
    assert isinstance(runners[1], ClaudeExecClient)


def test_select_runners_enable_openalex_replaces_codex(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.openalex_exec import OpenAlexRunner

    runners = select_runners(
        repo_path=tmp_path,
        codex_options=None,
        runner=None,
        runners=None,
        enable_openalex=True,
    )

    assert len(runners) == 1
    assert isinstance(runners[0], OpenAlexRunner)


def test_select_runners_openalex_is_exclusive_over_claude_and_codex(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.openalex_exec import OpenAlexRunner

    runners = select_runners(
        repo_path=tmp_path,
        codex_options=None,
        runner=None,
        runners=None,
        enable_openalex=True,
        enable_claude_search=True,
    )

    assert [type(r) for r in runners] == [OpenAlexRunner]


def test_select_runners_openalex_query_mode_default_is_codex(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.openalex_exec import OpenAlexRunner

    (runner,) = select_runners(
        repo_path=tmp_path,
        codex_options=None,
        runner=None,
        runners=None,
        enable_openalex=True,
    )

    assert isinstance(runner, OpenAlexRunner)
    assert runner.options.query_mode == "codex"


def test_select_runners_openalex_query_mode_regex_passthrough(tmp_path: Path) -> None:
    (runner,) = select_runners(
        repo_path=tmp_path,
        codex_options=None,
        runner=None,
        runners=None,
        enable_openalex=True,
        openalex_query_mode="regex",
    )

    assert runner.options.query_mode == "regex"
