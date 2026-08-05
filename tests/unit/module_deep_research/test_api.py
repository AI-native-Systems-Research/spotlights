"""Unit tests for the module deep-research API."""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.module_deep_research.api import (
    research_module,
    research_module_with_telemetry,
    resolve_target_module,
)
from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient
from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecResult
from spotlights_engine.module_deep_research.orchestration import select_runners
from spotlights_engine.schemas.candidate import Candidate, CodeLocation, CodeSpan
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


def _candidate(cid: str = "cand-inference_attention-0001") -> Candidate:
    return Candidate(
        id=cid,
        module_qualified_name="inference/attention",
        origin="code_agent",
        locations=[
            CodeLocation(
                file="src/inference/attention/core.py",
                spans=[
                    CodeSpan(
                        line_start=1,
                        line_end=20,
                        symbol="attend",
                        kind="function",
                    )
                ],
            )
        ],
        description="Naive attention.",
        current_approach="Dense softmax over the full sequence.",
        evolve_rationale="Quadratic cost dominates long-context latency.",
        estimated_impact="high",
        estimated_impact_explanation="Hot path.",
    )


def _request(
    module_qualified_name: str = "inference/attention",
    *,
    candidates: list[Candidate] | None = None,
) -> ModuleDeepResearchInput:
    return ModuleDeepResearchInput(
        project_tree=_tree(),
        module_qualified_name=module_qualified_name,
        context=SpotlightContext(objective="reduce latency"),
        repo_path=Path("/tmp/example-repo"),
        max_findings_per_candidate=5,
        candidates=candidates if candidates is not None else [_candidate()],
    )


def _request_with_cap(max_findings_per_candidate: int) -> ModuleDeepResearchInput:
    return _request().model_copy(
        update={"max_findings_per_candidate": max_findings_per_candidate}
    )


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

    # One prompt: one candidate.
    assert len(runner.prompts) == 1
    # The per-candidate prompt focuses on the candidate's code site.
    assert "attend" in runner.prompts[0]
    # Finding ids are renumbered + prefixed with the per-candidate segment (D3):
    # `find-<module_segment>-<candidate_counter>-NNNN`.
    assert output.findings[0].finding_id == "find-inference_attention-0001-0001"
    assert output.findings[0].candidate_id == "cand-inference_attention-0001"
    assert output.findings[0].title == "Flash attention tiling"
    assert output.issues == []


def test_research_module_loops_over_candidates_with_per_candidate_segments() -> None:
    runner = FakeRunner(
        """
        {
          "findings": [
            {
              "finding_id": "find-0001",
              "title": "Technique",
              "url": "https://example.com/x",
              "source_type": "paper",
              "technique_summary": "A transferable idea."
            }
          ],
          "issues": []
        }
        """
    )
    request = _request(
        candidates=[
            _candidate("cand-inference_attention-0001"),
            _candidate("cand-inference_attention-0002"),
        ]
    )

    output = research_module(request, runner=runner)

    # One prompt per candidate.
    assert len(runner.prompts) == 2
    assert [f.finding_id for f in output.findings] == [
        "find-inference_attention-0001-0001",
        "find-inference_attention-0002-0001",
    ]
    assert [f.candidate_id for f in output.findings] == [
        "cand-inference_attention-0001",
        "cand-inference_attention-0002",
    ]


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
        "find-inference_attention-0001-0001",
        "find-inference_attention-0001-0002",
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
        "find-inference_attention-0001-0001",
        "find-inference_attention-0001-0002",
        "find-inference_attention-0001-0003",
    ]
    assert [finding.title for finding in output.findings] == ["A", "B", "C"]


def test_telemetry_attributes_durations_per_candidate() -> None:
    """Decision D7: durations (the usage-attribution fallback) are keyed by
    candidate id, so N candidates never each claim the whole step-3 wall time."""
    request = _request(
        candidates=[
            _candidate("cand-inference_attention-0001"),
            _candidate("cand-inference_attention-0002"),
        ]
    )

    result = research_module_with_telemetry(
        request,
        runner=NamedFakeRunner("codex", '{"findings": [], "issues": []}'),
    )

    assert set(result.per_candidate_durations_s) == {
        "cand-inference_attention-0001",
        "cand-inference_attention-0002",
    }
    for durations in result.per_candidate_durations_s.values():
        assert "codex" in durations


def test_per_candidate_codex_last_message_paths_are_distinct(tmp_path: Path) -> None:
    """Decision D8: each candidate survey writes to its own codex last-message
    file `<dir>/<candidate_id>.md`, so two candidates cannot parse each other's
    response."""
    from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions

    seen_paths: list[Path | None] = []

    class RecordingCodex:
        name = "codex"

        def __init__(self, options: CodexExecOptions) -> None:
            self.options = options

        def run(self, prompt: str, *, check: bool = True) -> CodexExecResult:
            # Distinct payload per candidate, keyed off the per-candidate
            # last-message path the api derived (D8). If two candidates shared a
            # path/response, the finding titles below would cross-attribute.
            path = self.options.output_last_message
            seen_paths.append(path)
            stem = Path(path).stem  # e.g. cand-inference_attention-0001
            payload = json.dumps(
                {
                    "findings": [
                        {
                            "finding_id": "find-0001",
                            "title": f"technique for {stem}",
                            "url": "https://example.com/x",
                            "source_type": "paper",
                            "technique_summary": "A transferable idea.",
                        }
                    ],
                    "issues": [],
                }
            )
            return CodexExecResult(
                command=["codex", "exec"],
                returncode=0,
                stdout="",
                stderr="",
                final_message=payload,
                output_last_message=None,
            )

    # Emulate the manager wiring: the runner is constructed per candidate from the
    # derived options, so build them through select_runners by passing codex_options.
    request = _request(
        candidates=[
            _candidate("cand-inference_attention-0001"),
            _candidate("cand-inference_attention-0002"),
        ]
    )

    import spotlights_engine.module_deep_research.api as api_mod

    original = api_mod.select_runners

    def fake_select_runners(*, codex_options, **kwargs):
        # The api derives per-candidate codex_options; capture the file each gets.
        return (RecordingCodex(codex_options),)

    api_mod.select_runners = fake_select_runners  # type: ignore[assignment]
    try:
        result = api_mod.research_module_with_telemetry(
            request,
            codex_options=CodexExecOptions(
                cwd=tmp_path, output_last_message=tmp_path / "last_messages"
            ),
        )
    finally:
        api_mod.select_runners = original  # type: ignore[assignment]

    assert seen_paths == [
        tmp_path / "last_messages" / "cand-inference_attention-0001.md",
        tmp_path / "last_messages" / "cand-inference_attention-0002.md",
    ]
    # Cross-attribution guard: each candidate's finding must come from ITS OWN
    # codex response (whose payload was keyed off the per-candidate path), not
    # the other candidate's. A shared last-message path would swap these.
    by_candidate = {
        f.candidate_id: f.title for f in result.output.findings
    }
    assert by_candidate == {
        "cand-inference_attention-0001": "technique for cand-inference_attention-0001",
        "cand-inference_attention-0002": "technique for cand-inference_attention-0002",
    }


def test_claude_default_max_turns_is_80() -> None:
    from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient

    cmd = ClaudeExecClient().build_command()

    assert cmd[cmd.index("--max-turns") + 1] == "80"


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
