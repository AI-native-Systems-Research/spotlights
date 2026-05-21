"""Unit tests for the module deep-research API."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.module_deep_research.api import research_module, resolve_target_module
from spotlights_engine.module_deep_research.codex_exec import CodexExecResult
from spotlights_engine.schemas.deep_research import ModuleDeepResearchInput, SpotlightContext
from spotlights_engine.schemas.modules import Module, ProjectTree, Repository


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
        repository=Repository(name="demo", summary="Demo repository."),
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


def _request(module_qualified_name: str = "inference.attention") -> ModuleDeepResearchInput:
    return ModuleDeepResearchInput(
        project_tree=_tree(),
        module_qualified_name=module_qualified_name,
        context=SpotlightContext(objective="reduce latency"),
        repo_path=Path("/tmp/example-repo"),
        max_findings_per_module=5,
    )


def test_resolve_target_module_accepts_dot_qualified_name() -> None:
    module = resolve_target_module(_tree(), "inference.attention")

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
    assert output.findings[0].finding_id == "find-0001"
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

    assert cmd[0] == "codex"
    assert cmd.index("--profile") < cmd.index("exec")
    assert cmd.index("--ask-for-approval") < cmd.index("exec")
    assert cmd.index("--search") < cmd.index("exec")
    assert cmd.index("--sandbox") > cmd.index("exec")
    assert cmd[-1] == "-"
