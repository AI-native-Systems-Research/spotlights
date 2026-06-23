"""Unit tests for the module deep-research API."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from spotlights_engine.module_deep_research.api import research_module, resolve_target_module
from spotlights_engine.module_deep_research.codex_exec import CodexExecResult
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput, ModuleDeepResearchOutput
from spotlights_engine.schemas.project import Module, ProjectTree, Repository


class FakeRunner:
    name = "fake"

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


def test_resolve_target_module_by_slash_qualified_name() -> None:
    module = resolve_target_module(_tree(), "inference/attention")

    assert module is not None
    assert module.path == "src/inference/attention"


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


def test_research_module_runs_codex_claude_antigravity_and_dedups_outputs() -> None:
    codex = NamedFakeRunner("codex", _payload("PagedAttention", "https://arxiv.org/abs/2309.06180"))
    claude = NamedFakeRunner(
        "claude", _payload("PagedAttention", "https://arxiv.org/pdf/2309.06180v2.pdf")
    )
    antigravity = NamedFakeRunner(
        "antigravity", _payload("vAttention", "https://arxiv.org/abs/2405.04437")
    )

    output = research_module(_request(), runners=[codex, claude, antigravity])

    assert [len(r.prompts) for r in (codex, claude, antigravity)] == [1, 1, 1]
    assert [finding.finding_id for finding in output.findings] == [
        "find-inference_attention-0001",
        "find-inference_attention-0002",
    ]
    assert [finding.title for finding in output.findings] == ["PagedAttention", "vAttention"]
    assert output.issues == []


def test_research_module_dedups_exact_title_matches_with_different_urls() -> None:
    first = NamedFakeRunner("codex", _payload("Cache eviction", "https://example.com/paper-a"))
    second = NamedFakeRunner("gemini", _payload("Cache eviction", "https://example.com/paper-b"))

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
        BarrierRunner("gemini", _payload("C", "https://example.com/c")),
    ]

    output = research_module(_request(), runners=runners)

    assert [finding.title for finding in output.findings] == ["A", "B", "C"]
    assert output.issues == []


def test_research_module_treats_finding_cap_as_per_runner() -> None:
    runners = [
        NamedFakeRunner("codex", _payload("A", "https://example.com/a")),
        NamedFakeRunner("claude", _payload("B", "https://example.com/b")),
        NamedFakeRunner("gemini", _payload("C", "https://example.com/c")),
    ]

    output = research_module(_request_with_cap(1), runners=runners)

    assert [finding.finding_id for finding in output.findings] == [
        "find-inference_attention-0001",
        "find-inference_attention-0002",
        "find-inference_attention-0003",
    ]
    assert [finding.title for finding in output.findings] == ["A", "B", "C"]


def test_claude_default_max_turns_is_16() -> None:
    from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient

    cmd = ClaudeExecClient().build_command()

    assert cmd[cmd.index("--max-turns") + 1] == "16"


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
    assert "WebSearch" not in tools
    assert "Edit" not in tools




def test_select_runners_defaults_to_codex_claude_antigravity(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import AntigravityExecOptions
    from spotlights_engine.module_deep_research.orchestration import select_runners

    runners = select_runners(
        repo_path=tmp_path,
        codex_options=None,
        antigravity_options=AntigravityExecOptions.litellm_proxy(
            cwd=tmp_path,
            base_url="https://litellm.example.com",
        ),
        runner=None,
        runners=None,
    )

    assert [runner.name for runner in runners] == ["codex", "claude", "antigravity"]
    assert runners[-1].options.response_schema is ModuleDeepResearchOutput


def test_antigravity_litellm_proxy_enables_sse_bytes_repr_normalizer(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import AntigravityExecOptions

    options = AntigravityExecOptions.litellm_proxy(
        cwd=tmp_path,
        base_url="https://litellm.example.com",
    )

    assert options.normalize_sse_bytes_repr is True


def test_antigravity_normalizes_litellm_bytes_repr_sse_line() -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        _forward_headers,
        _is_streaming_response,
        _normalize_sse_lines,
        _strip_proxy_placeholder_key_from_headers,
        _strip_proxy_placeholder_key_from_url,
        normalize_litellm_sse_bytes_repr_line,
        normalize_litellm_vertex_sse_line,
    )

    line = 'data: b\'data: {"ok": true}\\r\\n\\r\\n\''

    assert normalize_litellm_sse_bytes_repr_line(line) == b'data: {"ok": true}\r\n\r\n'
    assert normalize_litellm_sse_bytes_repr_line("data: b'okenCount'") is None
    assert normalize_litellm_sse_bytes_repr_line("data: b'data: [DONE]\\n\\n'") == b''
    assert normalize_litellm_sse_bytes_repr_line('data: {"ok": true}') is None
    assert normalize_litellm_vertex_sse_line('data: {"ok": true}') is None
    assert _forward_headers({"accept-encoding": "gzip", "X-Test": "yes"}) == {
        "X-Test": "yes",
        "Accept-Encoding": "identity",
    }
    headers = {
        "X-Goog-Api-Key": "proxy-placeholder-key",
        "Authorization": "Bearer real-key",
    }
    _strip_proxy_placeholder_key_from_headers(headers)
    assert headers == {"Authorization": "Bearer real-key"}
    assert (
        _strip_proxy_placeholder_key_from_url(
            "https://gemini.example.com/v1beta/models/m:streamGenerateContent"
            "?alt=sse&key=proxy-placeholder-key"
        )
        == "https://gemini.example.com/v1beta/models/m:streamGenerateContent?alt=sse"
    )
    assert (
        _strip_proxy_placeholder_key_from_url(
            "https://gemini.example.com/v1beta/models/m:generateContent?key=real"
        )
        == "https://gemini.example.com/v1beta/models/m:generateContent?key=real"
    )

    assert _is_streaming_response(
        content_type="application/json",
        request_path="/v1beta/models/gemini:streamGenerateContent?alt=sse",
    )
    assert _is_streaming_response(content_type="text/event-stream", request_path="/")
    assert not _is_streaming_response(content_type="application/json", request_path="/models")
    assert list(
        _normalize_sse_lines(
            iter(
                [
                    b"event: message\n",
                    b'data: {"ok": true}\n',
                    b"\n",
                    b": keepalive\n",
                    b"data: b'okenCount'\n",
                    b"data: b'data: {\"nested\": true}\\r\\n\\r\\n'\n",
                    b"data: [DONE]\n",
                ]
            )
        )
    ) == [
        b"event: message\n",
        b'data: {"ok": true}\n',
        b'data: {"nested": true}\r\n\r\n',
    ]


def test_antigravity_normalizer_preserves_vertex_thought_signature() -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        _normalize_sse_lines,
        normalize_litellm_sse_bytes_repr_line,
        normalize_litellm_vertex_sse_line,
    )

    line = (
        "data: b'data: {\"candidates\":[{\"content\":{\"parts\":["
        "{\"text\":\"hi\",\"thoughtSignature\":\"opaque\"}]}}],"
        "\"usageMetadata\":{\"trafficType\":\"ON_DEMAND\"},"
        "\"modelVersion\":\"gemini-3.5-flash\"}\\r\\n\\r\\n'"
    )

    normalized = normalize_litellm_sse_bytes_repr_line(line)

    assert normalized is not None
    assert b"thoughtSignature" in normalized
    assert b'"text":"hi"' in normalized
    assert b"usageMetadata" not in normalized
    assert b"modelVersion" not in normalized

    leading_newline = normalize_litellm_sse_bytes_repr_line(
        "data: b'\\ndata: "
        "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}"
        "\\r\\n\\r\\n'"
    )
    assert leading_newline is not None
    assert leading_newline.startswith(b"data: ")

    assert (
        normalize_litellm_sse_bytes_repr_line(
            "data: b'data: "
            "{\"candidates\":[{\"content\":{\"parts\":[{"
            "\"functionCall\":{\"name\":\"list_dir\"},"
            "\"thoughtSignature\":\"abc'"
        )
        is None
    )

    plain_vertex = (
        'data: {"candidates":[{"content":{"parts":[{"text":"hi",'
        '"thoughtSignature":"opaque"}]}}],"usageMetadata":{"x":1}}'
    )
    normalized_plain = normalize_litellm_vertex_sse_line(plain_vertex)
    assert normalized_plain is not None
    assert b"thoughtSignature" in normalized_plain
    assert b"usageMetadata" not in normalized_plain
    assert b'"text":"hi"' in normalized_plain

    thought_only = normalize_litellm_vertex_sse_line(
        'data: {"candidates":[{"content":{"parts":[{"text":"thinking",'
        '"thought":true}]}}],"usageMetadata":{"x":1}}'
    )
    assert thought_only == b""

    fragmented = list(
        _normalize_sse_lines(
            iter(
                [
                    b"data: b'data: {\"candidates\":[{\"content\":{\"parts\":[{"
                    b"\"functionCall\":{\"name\":\"view_file\"},"
                    b"\"thoughtSignature\":\"abc",
                    b"data: b'def\"}]}}],\"usageMetadata\":{}}\\r\\n\\r\\n'\n",
                ]
            )
        )
    )
    assert len(fragmented) == 1
    assert fragmented[0].startswith(b"data: ")
    assert b"thoughtSignature" in fragmented[0]
    assert b"usageMetadata" not in fragmented[0]
    assert b"functionCall" in fragmented[0]

    concatenated_marker = normalize_litellm_sse_bytes_repr_line(
        "data: b'data: "
        "{\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"search_web\"},"
        "\"thoughtSignature\":\"abcdata: b'def\"}]}}],"
        "\"usageMetadata\":{},\"modelVersion\":\"x\"}\\r\\n\\r\\n'"
    )
    assert concatenated_marker is not None
    assert b"data: b" not in concatenated_marker
    assert b"thoughtSignature" in concatenated_marker
    assert b"usageMetadata" not in concatenated_marker
    assert b"functionCall" in concatenated_marker

    plain_embedded_marker = normalize_litellm_vertex_sse_line(
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"search_web"},'
        '"thoughtSignature":"abcdata: b\'def"}]}}],"usageMetadata":{},'
        '"modelVersion":"x"}\\r\\n\\r\\n\''
    )
    assert plain_embedded_marker is not None
    assert b"data: b" not in plain_embedded_marker
    assert b"thoughtSignature" in plain_embedded_marker
    assert b"usageMetadata" not in plain_embedded_marker
    assert b"functionCall" in plain_embedded_marker

    finish_with_helper_metadata = normalize_litellm_vertex_sse_line(
        'data: {"candidates":[{"content":{"parts":[{"functionCall":'
        '{"name":"finish","args":{"findings":[],"issues":[],'
        '"toolAction":"done","toolSummary":"done"}}}]}}]}'
    )
    assert finish_with_helper_metadata is not None
    assert b'"name":"finish"' in finish_with_helper_metadata
    assert b"toolAction" not in finish_with_helper_metadata
    assert b"toolSummary" not in finish_with_helper_metadata

    split_plain_then_bytes = list(
        _normalize_sse_lines(
            iter(
                [
                    b'data: {"candidates":[{"content":{"parts":[{'
                    b'"functionCall":{"name":"search_web"},'
                    b'"thoughtSignature":"abc',
                    b"data: b'def\"}]}}],\"usageMetadata\":{},\"modelVersion\":\"x\"}\r\n\r\n'\n",
                ]
            )
        )
    )
    assert len(split_plain_then_bytes) == 1
    assert b"data: b" not in split_plain_then_bytes[0]
    assert b"usageMetadata" not in split_plain_then_bytes[0]
    assert b"functionCall" in split_plain_then_bytes[0]
    assert b"thoughtSignature" in split_plain_then_bytes[0]

    tool_call_with_helper_metadata = normalize_litellm_vertex_sse_line(
        'data: {"candidates":[{"content":{"parts":[{"functionCall":'
        '{"name":"list_dir","args":{"DirectoryPath":"/tmp",'
        '"toolAction":"list","toolSummary":"list"}}}]}}]}'
    )
    assert tool_call_with_helper_metadata is not None
    assert b'"name":"list_dir"' in tool_call_with_helper_metadata
    assert b"toolAction" not in tool_call_with_helper_metadata
    assert b"toolSummary" not in tool_call_with_helper_metadata


def test_antigravity_normalizer_suppresses_empty_terminal_chunk() -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        _normalize_generate_content_body,
        normalize_litellm_sse_bytes_repr_line,
        normalize_litellm_vertex_sse_line,
    )

    line = (
        "data: b'data: {\"candidates\":[{\"content\":{\"role\":\"model\","
        "\"parts\":[{\"text\":\"\"}]},\"finishReason\":\"STOP\"}],"
        "\"usageMetadata\":{\"totalTokenCount\":123}}\\r\\n\\r\\n'"
    )

    assert normalize_litellm_sse_bytes_repr_line(line) == b""

    empty_nonterminal = (
        'data: {"candidates":[{"content":{"role":"model",'
        '"parts":[{"text":""}]}}],"usageMetadata":{"totalTokenCount":123}}'
    )
    assert normalize_litellm_vertex_sse_line(empty_nonterminal) == b""

    signature_only = (
        'data: {"candidates":[{"content":{"role":"model",'
        '"parts":[{"thoughtSignature":"opaque"}]}}]}'
    )
    assert normalize_litellm_vertex_sse_line(signature_only) == b""

    nonstream = _normalize_generate_content_body(
        b'{"candidates":[{"content":{"role":"model","parts":[{"text":""}]}}]}'
    )
    assert b"No content returned." in nonstream


def test_antigravity_default_command_is_sdk_and_does_not_expose_prompt(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    cmd = AntigravityExecClient(AntigravityExecOptions(cwd=tmp_path)).build_command()

    assert cmd[0] == "antigravity-sdk"
    assert "research prompt" not in cmd
    assert "--workspace" in cmd
    assert str(tmp_path.resolve()) in cmd


def test_antigravity_litellm_proxy_defers_bearer_header_to_normalizer(
    tmp_path: Path,
) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    client = AntigravityExecClient(
        AntigravityExecOptions.litellm_proxy(
            cwd=tmp_path,
            base_url="https://litellm.example.com",
            env={"LITELLM_API_KEY": "test-key"},
        )
    )

    assert client.build_model_endpoint_kwargs() == {
        "base_url": "https://litellm.example.com",
        "api_key": "proxy-placeholder-key",
    }


def test_antigravity_litellm_proxy_without_normalizer_passes_bearer_header(
    tmp_path: Path,
) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    client = AntigravityExecClient(
        AntigravityExecOptions.litellm_proxy(
            cwd=tmp_path,
            base_url="https://litellm.example.com",
            env={"LITELLM_API_KEY": "test-key"},
            normalize_sse_bytes_repr=False,
        )
    )

    assert client.build_model_endpoint_kwargs() == {
        "base_url": "https://litellm.example.com",
        "api_key": "test-key",
        "http_headers": {"Authorization": "Bearer test-key"},
    }


def test_antigravity_x_goog_key_uses_sdk_api_key_without_duplicate_header(
    tmp_path: Path,
) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    client = AntigravityExecClient(
        AntigravityExecOptions(
            cwd=tmp_path,
            antigravity_base_url="https://gemini.example.com",
            antigravity_api_key_env="GEMINI_API_KEY",
            api_key_auth_mechanism="x-goog-api-key",
            env={"GEMINI_API_KEY": "test-key"},
        )
    )

    assert client.build_model_endpoint_kwargs() == {
        "base_url": "https://gemini.example.com",
        "api_key": "test-key",
    }


def test_antigravity_runner_enables_web_search_tool(monkeypatch, tmp_path: Path) -> None:
    import os
    import sys
    import types

    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    captured: dict[str, object] = {}

    class BuiltinTools:
        LIST_DIR = "list_directory"
        SEARCH_DIR = "search_directory"
        FIND_FILE = "find_file"
        VIEW_FILE = "view_file"
        FINISH = "finish"
        SEARCH_WEB = "search_web"

        @classmethod
        def read_only(cls):
            return [cls.LIST_DIR, cls.SEARCH_DIR, cls.FIND_FILE, cls.VIEW_FILE, cls.FINISH]

    class CapabilitiesConfig:
        def __init__(self, *, enabled_tools):
            captured["enabled_tools"] = enabled_tools

    class LocalAgentConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class Agent:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def chat(self, prompt):
            captured["env_value"] = os.environ.get("ANTIGRAVITY_TEST_ENV")

            class Response:
                async def structured_output(self):
                    return {"findings": [], "issues": []}

                async def text(self):
                    return ""

            return Response()

    fake = types.ModuleType("google.antigravity")
    fake.Agent = Agent
    fake.BuiltinTools = BuiltinTools
    fake.CapabilitiesConfig = CapabilitiesConfig
    fake.GeminiAPIEndpoint = object
    fake.LocalAgentConfig = LocalAgentConfig
    fake.ModelTarget = object
    fake.ModelType = types.SimpleNamespace(TEXT="text")

    monkeypatch.setitem(sys.modules, "google.antigravity", fake)

    result = AntigravityExecClient(
        AntigravityExecOptions(
            cwd=tmp_path,
            env={"ANTIGRAVITY_TEST_ENV": "visible"},
        )
    ).run("prompt")

    assert result.ok
    assert "search_web" in captured["enabled_tools"]
    assert captured["env_value"] == "visible"
    assert os.environ.get("ANTIGRAVITY_TEST_ENV") is None


def test_antigravity_compacts_spotlights_prompt_and_wraps_web_result() -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        _is_simple_web_grounded_spotlights_prompt,
        _maybe_compact_spotlights_prompt,
        _wrap_simple_web_result,
    )

    prompt = """You are running the Spotlights module_deep_research pipeline step.

Target module:
Qualified name: v1/sample
Path: vllm/v1/sample

Caller context:
Objective: speed up speculative decoding rejection sampling
Workload hints:
- GPU kernels

Workflow:
1. Open files.

Output rules:
- Include at most 2 findings.
"""

    compacted = _maybe_compact_spotlights_prompt(
        prompt, compact=True, enable_file_tools=False
    )

    assert _is_simple_web_grounded_spotlights_prompt(compacted)
    assert "Use web search" in compacted
    assert "up to 2" in compacted
    assert "speed up speculative decoding" in compacted

    wrapped = _wrap_simple_web_result(
        '{"findings":['
        '{"title":"Dual Pivot Rejection Sampling",'
        '"url":"https://flashinfer.ai/2025/03/10/sampling",'
        '"source_type":"blog"},'
        '{"title":"EARS",'
        '"url":"https://arxiv.org/abs/2512.13194",'
        '"source_type":"paper"}'
        '],"issues":[]}'
    )

    assert json.loads(wrapped) == {
        "findings": [
            {
                "finding_id": "find-0001",
                "title": "Dual Pivot Rejection Sampling",
                "url": "https://flashinfer.ai/2025/03/10/sampling",
                "source_type": "blog",
                "technique_summary": (
                    "External source with a transferable optimization idea "
                    "for the target module."
                ),
                "supporting_evidence": "Source identified by web search.",
            },
            {
                "finding_id": "find-0002",
                "title": "EARS",
                "url": "https://arxiv.org/abs/2512.13194",
                "source_type": "paper",
                "technique_summary": (
                    "External source with a transferable optimization idea "
                    "for the target module."
                ),
                "supporting_evidence": "Source identified by web search.",
            },
        ],
        "issues": [],
    }

    single_wrapped = _wrap_simple_web_result(
        '{"title":"Dual Pivot Rejection Sampling",'
        '"url":"https://flashinfer.ai/2025/03/10/sampling",'
        '"source_type":"blog"}'
    )
    assert len(json.loads(single_wrapped)["findings"]) == 1

    file_compacted = _maybe_compact_spotlights_prompt(
        prompt, compact=True, enable_file_tools=True
    )
    assert not _is_simple_web_grounded_spotlights_prompt(file_compacted)
    assert "SPOTLIGHTS_COMPACT_MODULE_RESEARCH" in file_compacted
    assert "Read the target module main files" in file_compacted
    assert "Use file tools" in file_compacted
    assert "Include at most 2 findings" in file_compacted


def test_antigravity_inline_context_skips_large_ignored_dirs(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        _iter_inline_context_files,
    )

    repo = tmp_path / "repo"
    module_dir = repo / "vllm" / "v1" / "sample"
    ignored_dir = repo / ".git" / "objects"
    module_dir.mkdir(parents=True)
    ignored_dir.mkdir(parents=True)
    (module_dir / "sampler.py").write_text("def sample(): return 1\n")
    (ignored_dir / "ignored.py").write_text("def ignored(): return 2\n")

    files = list(_iter_inline_context_files([repo], max_files=4, max_scanned_entries=10))

    assert files == [(module_dir / "sampler.py").resolve()]


def test_antigravity_proxy_path_keeps_file_tools_and_stages_workspace(
    monkeypatch, tmp_path: Path
) -> None:
    import contextlib
    import sys
    import types

    import spotlights_engine.module_deep_research.antigravity_exec as ag_exec
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    captured: dict[str, object] = {}
    repo = tmp_path / "repo"
    module_dir = repo / "vllm" / "v1" / "sample"
    module_dir.mkdir(parents=True)
    (module_dir / "sample.py").write_text("def sample(): return 'ok'\n")
    stage_root = tmp_path / "stage"

    class BuiltinTools:
        LIST_DIR = "list_directory"
        SEARCH_DIR = "search_directory"
        FIND_FILE = "find_file"
        VIEW_FILE = "view_file"
        FINISH = "finish"
        SEARCH_WEB = "search_web"

        @classmethod
        def read_only(cls):
            return [cls.LIST_DIR, cls.SEARCH_DIR, cls.FIND_FILE, cls.VIEW_FILE, cls.FINISH]

    class CapabilitiesConfig:
        def __init__(self, *, enabled_tools):
            captured["enabled_tools"] = enabled_tools

    class GeminiAPIEndpoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ModelTarget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs["name"]

    class LocalAgentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["local_config_kwargs"] = kwargs

    class Agent:
        def __init__(self, config):
            self.config = config
            self._config = types.SimpleNamespace(**config.kwargs)
            # Simulate Antigravity's copied config containing SDK-added
            # defaults; the wrapper should strip these before entering.
            self._config.__dict__["models"] = [
                *config.kwargs["models"],
                "sdk-added-image-default",
            ]
            self._config.__dict__["api_key"] = "sdk-placeholder"

        async def __aenter__(self):
            captured["agent_config_models"] = self._config.models
            captured["agent_config_api_key"] = self._config.api_key
            staged_workspace = Path(self.config.kwargs["workspaces"][0])
            captured["staged_sample_text"] = (
                staged_workspace / "vllm" / "v1" / "sample" / "sample.py"
            ).read_text()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def chat(self, prompt):
            captured["prompt"] = prompt

            class Response:
                async def structured_output(self):
                    return None

                async def text(self):
                    return '{"findings":[],"issues":[]}'

            return Response()

    @contextlib.contextmanager
    def fake_normalizer(options):
        yield "https://normalizer.example.com"

    fake = types.ModuleType("google.antigravity")
    fake.Agent = Agent
    fake.BuiltinTools = BuiltinTools
    fake.CapabilitiesConfig = CapabilitiesConfig
    fake.GeminiAPIEndpoint = GeminiAPIEndpoint
    fake.LocalAgentConfig = LocalAgentConfig
    fake.ModelTarget = ModelTarget
    fake.ModelType = types.SimpleNamespace(TEXT="text")

    monkeypatch.setitem(sys.modules, "google.antigravity", fake)
    monkeypatch.setattr(ag_exec, "_maybe_sse_normalizer", fake_normalizer)

    prompt = f"""You are running the Spotlights module_deep_research pipeline step.

Repository working directory (the codex sandbox is rooted here; read files
directly with relative paths from this root, e.g. the target module path
below): {repo}

Target module:
Qualified name: v1/sample
Path: vllm/v1/sample

Caller context:
Objective: speed up speculative decoding rejection sampling

Workflow:
1. Open files.

Output rules:
- Include at most 1 findings.
"""

    options = AntigravityExecOptions.litellm_proxy(
        cwd=repo,
        base_url="https://litellm.example.com",
        model="proxy/gemini",
        env={"LITELLM_API_KEY": "test-key"},
    ).model_copy(
        update={
            "workspaces": [module_dir],
            "stage_workspaces": True,
            "staged_workspaces_root": stage_root,
            "response_schema": {"type": "object"},
        }
    )
    result = AntigravityExecClient(options).run(prompt, check=False)

    staged_workspace = Path(captured["local_config_kwargs"]["workspaces"][0])

    assert result.ok
    assert captured["enabled_tools"] == [
        "list_directory",
        "search_directory",
        "find_file",
        "view_file",
        "finish",
        "search_web",
    ]
    assert staged_workspace != repo
    assert captured["staged_sample_text"] == "def sample(): return 'ok'\n"
    assert str(repo) not in captured["prompt"]
    assert str(staged_workspace) in captured["prompt"]
    assert len(captured["agent_config_models"]) == 1
    assert captured["agent_config_models"][0].name == "proxy/gemini"
    assert captured["agent_config_api_key"] is None
    assert json.loads(result.final_message) == {"findings": [], "issues": []}


def test_antigravity_web_only_proxy_path_uses_compact_wrapper(
    monkeypatch, tmp_path: Path
) -> None:
    import contextlib
    import sys
    import types

    import spotlights_engine.module_deep_research.antigravity_exec as ag_exec
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    captured: dict[str, object] = {}

    class BuiltinTools:
        LIST_DIR = "list_directory"
        SEARCH_DIR = "search_directory"
        FIND_FILE = "find_file"
        VIEW_FILE = "view_file"
        FINISH = "finish"
        SEARCH_WEB = "search_web"

        @classmethod
        def read_only(cls):
            return [cls.LIST_DIR, cls.SEARCH_DIR, cls.FIND_FILE, cls.VIEW_FILE, cls.FINISH]

    class CapabilitiesConfig:
        def __init__(self, *, enabled_tools):
            captured["enabled_tools"] = enabled_tools

    class GeminiAPIEndpoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ModelTarget:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.name = kwargs["name"]

    class LocalAgentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["local_config_kwargs"] = kwargs

    class Agent:
        def __init__(self, config):
            self.config = config
            self._config = types.SimpleNamespace(**config.kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def chat(self, prompt):
            captured["prompt"] = prompt

            class Response:
                async def structured_output(self):
                    return None

                async def text(self):
                    return (
                        '{"findings":[{"title":"Paper","url":"https://example.com",'
                        '"source_type":"paper"}],"issues":[]}'
                    )

            return Response()

    @contextlib.contextmanager
    def fake_normalizer(options):
        yield "https://normalizer.example.com"

    fake = types.ModuleType("google.antigravity")
    fake.Agent = Agent
    fake.BuiltinTools = BuiltinTools
    fake.CapabilitiesConfig = CapabilitiesConfig
    fake.GeminiAPIEndpoint = GeminiAPIEndpoint
    fake.LocalAgentConfig = LocalAgentConfig
    fake.ModelTarget = ModelTarget
    fake.ModelType = types.SimpleNamespace(TEXT="text")

    monkeypatch.setitem(sys.modules, "google.antigravity", fake)
    monkeypatch.setattr(ag_exec, "_maybe_sse_normalizer", fake_normalizer)

    prompt = """You are running the Spotlights module_deep_research pipeline step.

Target module:
Qualified name: v1/sample
Path: vllm/v1/sample

Caller context:
Objective: speed up speculative decoding rejection sampling

Workflow:
1. Open files.

Output rules:
- Include at most 1 findings.
"""

    options = AntigravityExecOptions.litellm_proxy(
        cwd=tmp_path,
        base_url="https://litellm.example.com",
        model="proxy/gemini",
        env={"LITELLM_API_KEY": "test-key"},
    ).model_copy(update={"enable_file_tools": False})
    result = AntigravityExecClient(options).run(prompt, check=False)

    assert result.ok
    assert captured["enabled_tools"] == ["search_web"]
    assert captured["local_config_kwargs"]["workspaces"] == []
    assert captured["prompt"].startswith("SPOTLIGHTS_SIMPLE_WEB_RESULT\n")
    assert json.loads(result.final_message)["findings"][0]["finding_id"] == "find-0001"

def test_antigravity_repairs_missing_structured_output(monkeypatch, tmp_path: Path) -> None:
    import sys
    import types

    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    prompts: list[str] = []

    class BuiltinTools:
        LIST_DIR = "list_directory"
        SEARCH_DIR = "search_directory"
        FIND_FILE = "find_file"
        VIEW_FILE = "view_file"
        FINISH = "finish"
        SEARCH_WEB = "search_web"

        @classmethod
        def read_only(cls):
            return [cls.LIST_DIR, cls.SEARCH_DIR, cls.FIND_FILE, cls.VIEW_FILE, cls.FINISH]

    class CapabilitiesConfig:
        def __init__(self, *, enabled_tools):
            self.enabled_tools = enabled_tools

    class LocalAgentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Agent:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def chat(self, prompt):
            prompts.append(prompt)

            class Response:
                async def structured_output(self):
                    if len(prompts) < 3:
                        return None
                    return {"findings": [], "issues": []}

                async def text(self):
                    return "I will gather more context first."

            return Response()

    fake = types.ModuleType("google.antigravity")
    fake.Agent = Agent
    fake.BuiltinTools = BuiltinTools
    fake.CapabilitiesConfig = CapabilitiesConfig
    fake.GeminiAPIEndpoint = object
    fake.LocalAgentConfig = LocalAgentConfig
    fake.ModelTarget = object
    fake.ModelType = types.SimpleNamespace(TEXT="text")

    monkeypatch.setitem(sys.modules, "google.antigravity", fake)

    result = AntigravityExecClient(
        AntigravityExecOptions(
            cwd=tmp_path,
            model="gemini-test",
            response_schema={"type": "object"},
        )
    ).run("prompt")

    assert result.ok
    assert json.loads(result.final_message) == {"findings": [], "issues": []}
    assert prompts[0] == "prompt"
    assert "previous turn did not produce" in prompts[1]
    assert "previous repair turn still" in prompts[2]


def test_antigravity_run_wraps_async_sdk_result(monkeypatch, tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    async def fake_run_async(self, prompt: str) -> str:
        assert prompt == "prompt"
        return '{"findings": [], "issues": []}'

    monkeypatch.setattr(AntigravityExecClient, "_run_async", fake_run_async)

    result = AntigravityExecClient(AntigravityExecOptions(cwd=tmp_path)).run("prompt")

    assert result.ok
    assert result.final_message == '{"findings": [], "issues": []}'
    assert result.command[0] == "antigravity-sdk"

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


def test_gemini_litellm_proxy_command_uses_explicit_network_config(tmp_path: Path) -> None:
    from spotlights_engine.module_deep_research.gemini_exec import (
        GeminiExecClient,
        GeminiExecOptions,
    )

    cmd = GeminiExecClient(
        GeminiExecOptions.litellm_proxy(cwd=tmp_path, base_url="https://litellm.example.com")
    ).build_command()

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
        GeminiExecOptions.litellm_proxy(
            cwd=tmp_path,
            base_url="https://litellm.example.com",
            settings_path=settings_path,
            env={"LITELLM_API_KEY": "test-key"},
        )
    )

    env = client.build_env()

    assert env["GEMINI_API_KEY"] == "test-key"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "https://litellm.example.com"
    assert env["GEMINI_API_KEY_AUTH_MECHANISM"] == "bearer"
    assert env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] == str(settings_path)


def test_gemini_litellm_settings_route_internal_web_tools() -> None:
    from spotlights_engine.module_deep_research.gemini_exec import litellm_settings_payload

    payload = litellm_settings_payload(
        model="gcp/gemini-3.1-pro-preview",
        web_utility_model="gcp/gemini-3.1-pro-preview",
    )

    aliases = payload["modelConfigs"]["customAliases"]
    assert aliases["web-search"]["modelConfig"]["model"] == "gcp/gemini-3.1-pro-preview"
    assert aliases["web-search"]["modelConfig"]["generateContentConfig"]["tools"] == [
        {"googleSearch": {}}
    ]
    assert aliases["web-fetch"]["modelConfig"]["model"] == "gcp/gemini-3.1-pro-preview"
    assert aliases["web-fetch"]["modelConfig"]["generateContentConfig"]["tools"] == [
        {"urlContext": {}}
    ]
    assert aliases["web-fetch-fallback"]["modelConfig"]["model"] == ("gcp/gemini-3.1-pro-preview")


def test_gemini_litellm_proxy_run_uses_managed_web_tool_settings(
    monkeypatch, tmp_path: Path
) -> None:
    from spotlights_engine.module_deep_research import gemini_exec
    from spotlights_engine.module_deep_research.gemini_exec import (
        GeminiExecClient,
        GeminiExecOptions,
    )

    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        settings_path = Path(kwargs["env"]["GEMINI_CLI_SYSTEM_SETTINGS_PATH"])
        captured["settings_path"] = settings_path
        captured["settings"] = json.loads(settings_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"response": "ok"}',
            stderr="",
        )

    monkeypatch.setattr(gemini_exec.subprocess, "run", fake_run)

    result = GeminiExecClient(
        GeminiExecOptions.litellm_proxy(
            cwd=tmp_path,
            base_url="https://litellm.example.com",
            env={"LITELLM_API_KEY": "test-key"},
        )
    ).run("prompt")

    assert result.final_message == "ok"
    settings_path = captured["settings_path"]
    assert isinstance(settings_path, Path)
    assert not settings_path.exists()
    aliases = captured["settings"]["modelConfigs"]["customAliases"]  # type: ignore[index]
    assert aliases["web-search"]["modelConfig"]["model"] == "gcp/gemini-3.1-pro-preview"


def test_cli_resolution_preserves_posix_command_shape(monkeypatch) -> None:
    from spotlights_engine.module_deep_research import agent_exec

    monkeypatch.setattr(agent_exec, "WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION", False)

    assert agent_exec.resolve_cli_executable("claude") == "claude"


def test_cli_resolution_uses_windows_cmd_shims(monkeypatch) -> None:
    from spotlights_engine.module_deep_research import agent_exec
    from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient
    from spotlights_engine.module_deep_research.codex_exec import CodexExecClient, CodexExecOptions
    from spotlights_engine.module_deep_research.gemini_exec import GeminiExecClient

    def fake_which(executable: str) -> str:
        return f"C:/Users/example/AppData/Roaming/npm/{executable}.CMD"

    monkeypatch.setattr(agent_exec, "WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION", True)
    monkeypatch.setattr(agent_exec.shutil, "which", fake_which)

    assert agent_exec.resolve_cli_executable("claude") == (
        "C:/Users/example/AppData/Roaming/npm/claude.CMD"
    )
    assert ClaudeExecClient().build_command()[0].endswith("/claude.CMD")
    assert GeminiExecClient().build_command()[0].endswith("/gemini.CMD")
    codex_cmd, _ = CodexExecClient(CodexExecOptions(output_last_message="last.md")).build_command()
    assert codex_cmd[0].endswith("/codex.CMD")


def test_antigravity_falls_back_to_inline_context_web_when_file_tools_fail(
    monkeypatch, tmp_path: Path
) -> None:
    from spotlights_engine.module_deep_research.antigravity_exec import (
        AntigravityExecClient,
        AntigravityExecOptions,
    )

    repo = tmp_path / "repo"
    module_dir = repo / "vllm" / "v1" / "sample"
    module_dir.mkdir(parents=True)
    (module_dir / "sampler.py").write_text("def sample(): return 'ok'\n")
    captured: dict[str, str] = {}

    async def fake_run_async(self, prompt: str) -> str:
        if self.options.enable_file_tools:
            raise TimeoutError("file tools hung")
        captured["prompt"] = prompt
        return (
            '{"findings":[{"finding_id":"find-0001","title":"EARS",'
            '"url":"https://arxiv.org/abs/2512.13194",'
            '"source_type":"paper","technique_summary":"Adaptive rejection '
            'thresholds improve speculative decoding acceptance.",'
            '"supporting_evidence":"paper abstract"}],"issues":[]}'
        )

    monkeypatch.setattr(AntigravityExecClient, "_run_async", fake_run_async)

    result = AntigravityExecClient(
        AntigravityExecOptions(
            cwd=repo,
            workspaces=[module_dir],
            antigravity_base_url="https://gateway.example.com",
            antigravity_api_key_env="LITELLM_API_KEY",
            api_key_auth_mechanism="bearer",
            normalize_sse_bytes_repr=True,
            timeout_seconds=20,
        )
    ).run("You are running the Spotlights module_deep_research pipeline step.")

    assert result.ok
    assert "--fallback" in result.command
    assert "primary Antigravity file-tool run failed" in result.stderr
    assert "Inline target module context" in captured["prompt"]
    assert "def sample" in captured["prompt"]
