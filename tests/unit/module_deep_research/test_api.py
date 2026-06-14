"""Unit tests for the module deep-research API."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.module_deep_research.api import research_module, resolve_target_module
from spotlights_engine.module_deep_research.codex_exec import CodexExecResult
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


def test_research_module_runs_codex_claude_gemini_and_dedups_outputs() -> None:
    codex = NamedFakeRunner("codex", _payload("PagedAttention", "https://arxiv.org/abs/2309.06180"))
    claude = NamedFakeRunner("claude", _payload("PagedAttention", "https://arxiv.org/pdf/2309.06180.pdf"))
    gemini = NamedFakeRunner("gemini", _payload("vAttention", "https://arxiv.org/abs/2405.04437"))

    output = research_module(_request(), runners=[codex, claude, gemini])

    assert [len(r.prompts) for r in (codex, claude, gemini)] == [1, 1, 1]
    assert [finding.finding_id for finding in output.findings] == ["find-0001", "find-0002"]
    assert [finding.title for finding in output.findings] == ["PagedAttention", "vAttention"]
    assert output.issues == []


def test_research_module_keeps_same_title_when_urls_differ() -> None:
    first = NamedFakeRunner("codex", _payload("Cache eviction", "https://example.com/paper-a"))
    second = NamedFakeRunner("gemini", _payload("Cache eviction", "https://example.com/paper-b"))

    output = research_module(_request(), runners=[first, second])

    assert [finding.url for finding in output.findings] == [
        "https://example.com/paper-a",
        "https://example.com/paper-b",
    ]


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
        BarrierRunner("gemini", _payload("C", "https://example.com/c")),
    ]

    output = research_module(_request(), runners=runners)

    assert [finding.title for finding in output.findings] == ["A", "B", "C"]
    assert output.issues == []


def test_claude_command_shape_uses_print_json_and_plan_mode(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.claude_exec import (
        ClaudeExecClient,
        ClaudeExecOptions,
    )

    cmd = ClaudeExecClient(
        ClaudeExecOptions(cwd=tmp_path, model="claude-opus-4-7", max_turns=3)
    ).build_command()

    assert cmd[:4] == ["claude", "-p", "--output-format", "json"]
    assert cmd[cmd.index("--permission-mode") + 1] == "plan"
    assert cmd[cmd.index("--model") + 1] == "claude-opus-4-7"
    assert cmd[cmd.index("--max-turns") + 1] == "3"


def test_gemini_default_command_is_read_only_and_does_not_expose_prompt(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.gemini_exec import (
        GeminiExecClient,
        GeminiExecOptions,
    )

    cmd = GeminiExecClient(GeminiExecOptions(cwd=tmp_path)).build_command()

    assert cmd[:3] == ["gemini", "--prompt", ""]
    assert "research prompt" not in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--approval-mode") + 1] == "plan"
    assert "--skip-trust" not in cmd
    assert "--model" not in cmd


def test_gemini_ibm_litellm_command_uses_explicit_network_config(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.gemini_exec import (
        GeminiExecClient,
        GeminiExecOptions,
    )

    cmd = GeminiExecClient(GeminiExecOptions.ibm_litellm(cwd=tmp_path)).build_command()

    assert cmd[cmd.index("--approval-mode") + 1] == "yolo"
    assert "--skip-trust" in cmd
    assert cmd[cmd.index("--model") + 1] == "gcp/gemini-3.1-pro-preview"


def test_gemini_env_maps_litellm_key_to_gemini_proxy(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.gemini_exec import (
        GeminiExecClient,
        GeminiExecOptions,
    )

    settings_path = tmp_path / "gemini-settings.json"
    client = GeminiExecClient(
        GeminiExecOptions.ibm_litellm(
            cwd=tmp_path,
            settings_path=settings_path,
            env={"LITELLM_API_KEY": "test-key"},
        )
    )

    env = client.build_env()

    assert env["GEMINI_API_KEY"] == "test-key"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "https://ete-litellm.ai-models.vpc-int.res.ibm.com"
    assert env["GEMINI_API_KEY_AUTH_MECHANISM"] == "bearer"
    assert env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] == str(settings_path)
