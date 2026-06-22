"""Small wrapper around the Google Antigravity SDK agent API."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult

DEFAULT_LITELLM_ANTIGRAVITY_MODEL = "gcp/gemini-3.1-pro-preview"
_DEFAULT_PROXY_TIMEOUT_SECONDS = 900
_PROXY_PLACEHOLDER_API_KEY = "proxy-placeholder-key"


class AntigravityExecOptions(BaseModel):
    """Options for running Google Antigravity SDK in non-interactive mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    model: str | None = None
    antigravity_base_url: str | None = None
    antigravity_api_key_env: str | None = None
    api_key_auth_mechanism: Literal["x-goog-api-key", "bearer"] | None = None
    timeout_seconds: int | None = None
    env: Mapping[str, str] | None = None
    system_instructions: str | None = None
    workspaces: Sequence[Path | str] | None = None
    response_schema: dict[str, Any] | type[BaseModel] | str | None = None
    extra_http_headers: Mapping[str, str] | None = None
    normalize_sse_bytes_repr: bool = False
    structured_output_repair_attempts: int = Field(default=3, ge=0, le=8)
    enable_file_tools: bool = True
    stage_workspaces: bool | None = None
    staged_workspaces_root: Path | str | None = None

    @classmethod
    def litellm_proxy(
        cls,
        *,
        base_url: str,
        cwd: Path | str | None = None,
        model: str = DEFAULT_LITELLM_ANTIGRAVITY_MODEL,
        api_key_env: str = "LITELLM_API_KEY",
        timeout_seconds: int | None = None,
        env: Mapping[str, str] | None = None,
        normalize_sse_bytes_repr: bool = True,
    ) -> AntigravityExecOptions:
        """Return explicit Antigravity SDK settings for a Gemini-compatible gateway.

        Some LiteLLM proxy versions return Gemini streaming chunks as Python
        bytes repr strings inside SSE frames (`data: b'data: ...'`). The SDK's
        local harness expects standard Gemini SSE, so this helper enables a
        local, provider-neutral normalizer by default. Disable it when your
        gateway already emits standards-compliant Gemini SSE.
        """
        return cls(
            cwd=Path.cwd() if cwd is None else cwd,
            model=model,
            antigravity_base_url=base_url.rstrip("/"),
            antigravity_api_key_env=api_key_env,
            api_key_auth_mechanism="bearer",
            timeout_seconds=timeout_seconds,
            env=env,
            normalize_sse_bytes_repr=normalize_sse_bytes_repr,
        )


class AntigravityExecClient:
    """Run Google Antigravity SDK from Python."""

    name = "antigravity"

    def __init__(self, options: AntigravityExecOptions | None = None) -> None:
        self.options = options or AntigravityExecOptions()

    def build_command(self) -> list[str]:
        """Return a log-friendly pseudo-command for parity with CLI runners."""
        opt = self.options
        cmd = ["antigravity-sdk"]
        if opt.model:
            cmd += ["--model", opt.model]
        if opt.antigravity_base_url:
            cmd += ["--base-url", opt.antigravity_base_url]
        cmd += ["--workspace", str(Path(opt.cwd).expanduser().resolve())]
        return cmd

    def build_env(self) -> dict[str, str]:
        """Return the environment visible to SDK config construction."""
        env = os.environ.copy()
        if self.options.env:
            env.update(dict(self.options.env))
        return env

    def build_model_endpoint_kwargs(self) -> dict[str, Any]:
        """Return GeminiAPIEndpoint keyword args for the configured auth path."""
        opt = self.options
        env = self.build_env()
        key = env.get(opt.antigravity_api_key_env or "") if opt.antigravity_api_key_env else None
        headers = dict(opt.extra_http_headers or {})

        kwargs: dict[str, Any] = {}
        if opt.antigravity_base_url:
            kwargs["base_url"] = opt.antigravity_base_url
        if key and opt.api_key_auth_mechanism == "bearer":
            # The Antigravity SDK requires a non-empty Gemini API key when
            # constructing the endpoint even when a gateway authenticates with
            # a bearer header. When the local SSE normalizer is enabled, keep
            # the real secret out of GeminiAPIEndpoint entirely: current
            # Antigravity harness builds crash before sending any request with
            # some non-Gemini key formats and with custom endpoint headers.
            # Do not pass bearer headers into GeminiAPIEndpoint when the local
            # normalizer is enabled. The normalizer injects those headers
            # upstream instead.
            if opt.antigravity_base_url and opt.normalize_sse_bytes_repr:
                kwargs["api_key"] = _PROXY_PLACEHOLDER_API_KEY
            else:
                kwargs["api_key"] = key
                headers.setdefault("Authorization", f"Bearer {key}")
        elif key:
            kwargs["api_key"] = key
        if headers:
            kwargs["http_headers"] = headers
        return kwargs

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        try:
            final_message = asyncio.run(self._run_async(prompt))
            result = AgentExecResult(
                command=self.build_command(),
                returncode=0,
                stdout=final_message,
                stderr="",
                final_message=final_message,
            )
        except Exception as exc:  # noqa: BLE001 - convert SDK failures to runner output
            result = self._try_inline_context_web_fallback(prompt, exc)
            if result is None:
                result = AgentExecResult(
                    command=self.build_command(),
                    returncode=1,
                    stdout="",
                    stderr=f"{type(exc).__name__}: {exc}",
                    final_message=None,
                )
        if check:
            result.raise_for_status()
        return result

    def _try_inline_context_web_fallback(
        self, prompt: str, primary_error: Exception
    ) -> AgentExecResult | None:
        """Fallback for SDK/proxy file-tool instability.

        The preferred path keeps Antigravity file tools enabled. Some SDK +
        Gemini-compatible gateway combinations can still stall or crash while
        interleaving file tools and web search. In that case, preserve module
        grounding by embedding bounded read-only snippets from the configured
        workspaces, then retry with web search only.
        """
        opt = self.options
        if not (
            opt.enable_file_tools
            and opt.antigravity_base_url
            and opt.normalize_sse_bytes_repr
        ):
            return None
        fallback_prompt = _append_inline_workspace_context(prompt, opt)
        fallback_options = opt.model_copy(
            update={
                "enable_file_tools": False,
                "workspaces": [],
                "stage_workspaces": False,
                "timeout_seconds": max(opt.timeout_seconds or 0, 300),
            }
        )
        try:
            final_message = asyncio.run(
                AntigravityExecClient(fallback_options)._run_async(fallback_prompt)
            )
        except Exception as fallback_error:  # noqa: BLE001
            return AgentExecResult(
                command=self.build_command(),
                returncode=1,
                stdout="",
                stderr=(
                    f"{type(primary_error).__name__}: {primary_error}\n"
                    "Inline-context web fallback also failed: "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                ),
                final_message=None,
            )
        return AgentExecResult(
            command=self.build_command() + ["--fallback", "inline-context-web"],
            returncode=0,
            stdout=final_message,
            stderr=(
                "primary Antigravity file-tool run failed; used inline-context "
                f"web fallback: {type(primary_error).__name__}: {primary_error}"
            ),
            final_message=final_message,
        )

    async def _run_async(self, prompt: str) -> str:
        with _patched_environ(self.options.env):
            coro = self._run_antigravity(prompt)
            timeout_seconds = self.options.timeout_seconds
            if (
                timeout_seconds is None
                and self.options.antigravity_base_url
                and self.options.normalize_sse_bytes_repr
            ):
                timeout_seconds = _DEFAULT_PROXY_TIMEOUT_SECONDS
            if timeout_seconds is None:
                return await coro
            return await asyncio.wait_for(coro, timeout=timeout_seconds)

    async def _run_antigravity(self, prompt: str) -> str:
        try:
            from google.antigravity import (  # type: ignore[import-not-found]
                Agent,
                BuiltinTools,
                CapabilitiesConfig,
                GeminiAPIEndpoint,
                LocalAgentConfig,
                ModelTarget,
                ModelType,
            )
        except ImportError as exc:
            raise RuntimeError(
                "google-antigravity is required for AntigravityExecClient; "
                "install the package or configure custom module_deep_research runners"
            ) from exc

        opt = self.options
        prompt = _maybe_compact_spotlights_prompt(
            prompt,
            compact=bool(
                opt.antigravity_base_url
                and opt.normalize_sse_bytes_repr
            ),
            enable_file_tools=opt.enable_file_tools,
        )
        wrap_simple_web_result = _is_simple_web_grounded_spotlights_prompt(prompt)
        cwd = Path(opt.cwd).expanduser().resolve()
        effective_enable_file_tools = opt.enable_file_tools
        workspaces = (
            []
            if not effective_enable_file_tools
            else [
                str(Path(path).expanduser().resolve())
                for path in (opt.workspaces or [cwd])
            ]
        )
        endpoint_kwargs = self.build_model_endpoint_kwargs()
        model = opt.model
        endpoint_model = model or (
            DEFAULT_LITELLM_ANTIGRAVITY_MODEL if opt.antigravity_base_url else None
        )
        research_tools = _antigravity_research_tools(
            BuiltinTools, enable_file_tools=effective_enable_file_tools
        )
        response_schema = None if wrap_simple_web_result else opt.response_schema
        with _maybe_staged_workspaces(opt, prompt, cwd, workspaces) as (
            agent_prompt,
            agent_workspaces,
        ):
            config_kwargs: dict[str, Any] = {
                "system_instructions": opt.system_instructions,
                "capabilities": CapabilitiesConfig(enabled_tools=research_tools),
                "workspaces": agent_workspaces,
                "response_schema": response_schema,
            }
            if endpoint_model and endpoint_kwargs:
                if endpoint_kwargs.get("api_key"):
                    # LocalAgentConfig appends SDK defaults for model types not
                    # covered by explicit ModelTarget entries. Keep validation
                    # satisfied without hardcoding any provider-specific key name.
                    config_kwargs["api_key"] = endpoint_kwargs["api_key"]
                with _maybe_sse_normalizer(opt) as normalized_base_url:
                    if normalized_base_url:
                        endpoint_kwargs = dict(endpoint_kwargs)
                        endpoint_kwargs["base_url"] = normalized_base_url
                    model_target = ModelTarget(
                        name=endpoint_model,
                        types=[ModelType.TEXT],
                        endpoint=GeminiAPIEndpoint(**endpoint_kwargs),
                    )
                    config_kwargs["model"] = model_target
                    config_kwargs["models"] = [model_target]
                    agent_session = Agent(LocalAgentConfig(**config_kwargs))
                    _force_agent_model_targets(agent_session, [model_target])
                    async with agent_session as agent:
                        output = await _chat_text_or_structured_json(
                            agent,
                            agent_prompt,
                            structured_output_repair_attempts=(
                                opt.structured_output_repair_attempts
                                if (response_schema is not None or wrap_simple_web_result)
                                else 0
                            ),
                        )
                        if wrap_simple_web_result:
                            return _wrap_simple_web_result(output)
                        return output
            if model:
                config_kwargs["model"] = model

            async with Agent(LocalAgentConfig(**config_kwargs)) as agent:
                output = await _chat_text_or_structured_json(
                    agent,
                    agent_prompt,
                    structured_output_repair_attempts=(
                        opt.structured_output_repair_attempts
                        if (response_schema is not None or wrap_simple_web_result)
                        else 0
                    ),
                )
                if wrap_simple_web_result:
                    return _wrap_simple_web_result(output)
                return output


def _force_agent_model_targets(agent: Any, model_targets: list[Any]) -> None:
    """Keep Antigravity's copied config on the exact requested model targets.

    LocalAgentConfig currently appends SDK default models for uncovered model
    types, and Agent deep-copies that config during construction. For a
    Gemini-compatible proxy that only supports a text model, those defaults can
    make the harness crash or call the wrong upstream before user code sees a
    useful error. This targeted adjustment keeps the proxy path provider-neutral
    while avoiding SDK-added image/default targets.
    """
    config = getattr(agent, "_config", None)
    if config is None:
        return
    try:
        config.__dict__["model"] = None
        config.__dict__["models"] = list(model_targets)
        config.__dict__["api_key"] = None
    except Exception:  # noqa: BLE001 - best-effort compatibility with SDK internals
        return


@contextlib.contextmanager
def _maybe_staged_workspaces(
    options: AntigravityExecOptions,
    prompt: str,
    cwd: Path,
    workspaces: Sequence[str],
) -> Iterator[tuple[str, list[str]]]:
    """Stage fragile mounted workspaces onto a native filesystem for Antigravity.

    The Antigravity local harness can crash before the first model request when
    asked to index certain mounted workspaces. In that case, copy the requested
    read-only workspace subset into a short-lived cache directory and rewrite
    the prompt's repository root to that staged root. This keeps file tools
    enabled without provider- or organization-specific assumptions.
    """
    workspace_paths = [Path(path).expanduser().resolve() for path in workspaces]
    should_stage = options.stage_workspaces
    if should_stage is None:
        should_stage = any(_should_stage_workspace(path) for path in workspace_paths)
    if not should_stage or not workspace_paths:
        yield prompt, [str(path) for path in workspace_paths]
        return

    staging_parent = Path(
        options.staged_workspaces_root
        or Path.home() / "spotlights-engine-antigravity-workspaces"
    ).expanduser()
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="run-", dir=str(staging_parent)
    ) as staging_dir:
        staged_cwd = Path(staging_dir) / "repo"
        staged_cwd.mkdir(parents=True, exist_ok=True)
        for workspace in workspace_paths:
            _copy_workspace_into_staged_repo(workspace, cwd, staged_cwd)
        staged_prompt = prompt.replace(str(cwd), str(staged_cwd))
        staged_prompt = staged_prompt.replace(cwd.as_posix(), staged_cwd.as_posix())
        yield staged_prompt, [str(staged_cwd)]


def _should_stage_workspace(path: Path) -> bool:
    as_posix = path.as_posix()
    if os.environ.get("SPOTLIGHTS_ANTIGRAVITY_STAGE_WORKSPACES"):
        return True
    if os.environ.get("WSL_DISTRO_NAME") and as_posix.startswith("/mnt/"):
        return True
    return False


def _copy_workspace_into_staged_repo(workspace: Path, cwd: Path, staged_cwd: Path) -> None:
    if not workspace.exists():
        return
    try:
        relative = workspace.relative_to(cwd)
    except ValueError:
        relative = Path(workspace.name)
    destination = staged_cwd / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if workspace.is_dir():
        shutil.copytree(
            workspace,
            destination,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                ".venv",
                "node_modules",
            ),
        )
    else:
        shutil.copy2(workspace, destination)


def _maybe_compact_spotlights_prompt(
    prompt: str, *, compact: bool, enable_file_tools: bool
) -> str:
    if not compact or "Spotlights module_deep_research pipeline step" not in prompt:
        return prompt
    context = _extract_prompt_section(prompt, "Caller context", "Workflow")
    objective = _extract_first_group(r"Objective: (.*)", context) or context
    target_module = _extract_prompt_section(prompt, "Target module", "Caller context")
    repo_path = (
        _extract_first_group(
            r"Repository working directory.*?\): ([^\n]+)",
            prompt,
            flags=re.DOTALL,
        )
        or "(repository root)"
    )
    max_findings = _extract_first_group(r"Include at most (\d+) findings\.", prompt) or "3"
    if enable_file_tools:
        return (
            "SPOTLIGHTS_COMPACT_MODULE_RESEARCH\n"
            "Do not modify files. Use file tools only for the target module, "
            "then use web search for transferable techniques. Keep the run short.\n\n"
            f"Repository root: {repo_path}\n\n"
            f"Target module:\n{target_module}\n\n"
            f"Caller context:\n{context}\n\n"
            "Workflow:\n"
            "1. Read the target module main files listed above. Read at most three "
            "nearby files only if required to understand the sampler path.\n"
            "2. Run focused web searches for concrete algorithms, papers, PRs, or "
            "implementation notes relevant to this module and objective.\n"
            "3. Return a bare JSON object only. No markdown.\n\n"
            "JSON shape: {\"findings\":[{\"finding_id\":\"find-0001\","
            "\"title\":string,\"url\":string,\"source_type\":\"paper|blog|docs|"
            "issue|pr|talk|codebase|other\",\"technique_summary\":string,"
            "\"supporting_evidence\":string}],\"issues\":[]}.\n"
            f"Include at most {max_findings} findings. Keep only sources with a "
            "specific transferable optimization idea for speculative decoding "
            "sampling/rejection hot paths. If evidence is insufficient, return "
            "empty findings with a recoverable issue."
        )
    inline_context = _extract_prompt_section(
        prompt,
        "Inline target module context",
        "End inline target module context.",
    )
    return (
        "SPOTLIGHTS_SIMPLE_WEB_RESULT\n"
        "Return only {\"findings\": [...], \"issues\": [...]} JSON. Do not return "
        "a module summary, file inventory, or keys named module/name/path/files.\n\n"
        f"Target module:\n{target_module}\n\n"
        f"Inline module code context:\n{inline_context}\n\n"
        f"Use web search to find up to {max_findings} recent papers, blogs, "
        "docs pages, issues, or PRs with concrete techniques related to this "
        "caller objective. "
        f"Caller objective: {objective}. "
        "Return ONLY JSON: "
        "{\"findings\":[{\"title\": string, \"url\": string, "
        "\"source_type\": \"paper|blog|docs|issue|pr|talk|codebase|other\", "
        "\"technique_summary\": string, "
        "\"supporting_evidence\": string}], \"issues\": []}."
    )


def _append_inline_workspace_context(
    prompt: str,
    options: AntigravityExecOptions,
    *,
    max_total_chars: int = 12_000,
    max_file_chars: int = 3_000,
    max_files: int = 4,
) -> str:
    if "Inline target module context:" in prompt:
        return prompt
    cwd = Path(options.cwd).expanduser().resolve()
    workspace_paths = [
        Path(path).expanduser().resolve() for path in (options.workspaces or [cwd])
    ]
    chunks: list[str] = []
    remaining = max_total_chars
    for file_path in _iter_inline_context_files(workspace_paths, max_files=max_files):
        if remaining <= 0:
            break
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        try:
            label = file_path.relative_to(cwd).as_posix()
        except ValueError:
            label = file_path.as_posix()
        body = text[: min(max_file_chars, remaining)]
        chunks.append(f"--- {label} ---\n{body}")
        remaining -= len(body)
    context = "\n\n".join(chunks) if chunks else "(no readable workspace files found)"
    return (
        prompt
        + "\n\nInline target module context:\n"
        + context
        + "\nEnd inline target module context.\n"
    )


def _iter_inline_context_files(
    workspace_paths: Sequence[Path], *, max_files: int
) -> Iterator[Path]:
    suffix_priority = {
        ".py": 0,
        ".pyi": 1,
        ".cu": 2,
        ".cuh": 3,
        ".cpp": 4,
        ".cc": 5,
        ".h": 6,
        ".hpp": 7,
        ".md": 8,
        ".rst": 9,
    }
    seen: set[Path] = set()
    candidates: list[Path] = []
    for workspace in workspace_paths:
        if workspace.is_file():
            candidates.append(workspace)
            continue
        if not workspace.is_dir():
            continue
        for path in workspace.rglob("*"):
            if len(candidates) >= max_files * 4:
                break
            if not path.is_file() or path.suffix.lower() not in suffix_priority:
                continue
            if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    candidates.sort(
        key=lambda p: (
            suffix_priority.get(p.suffix.lower(), 99),
            len(p.parts),
            p.as_posix(),
        )
    )
    yield from candidates[:max_files]


def _is_simple_web_grounded_spotlights_prompt(prompt: str) -> bool:
    return prompt.startswith("SPOTLIGHTS_SIMPLE_WEB_RESULT\n")


def _wrap_simple_web_result(output: str) -> str:
    try:
        payload = json.loads(_extract_json_object_text(output))
    except (TypeError, json.JSONDecodeError):
        return output
    if not isinstance(payload, dict):
        return output
    raw_findings = payload.get("findings")
    if isinstance(raw_findings, list):
        findings = [
            _simple_web_finding(item, index)
            for index, item in enumerate(raw_findings, start=1)
            if isinstance(item, dict) and item.get("title") and item.get("url")
        ]
    elif payload.get("title") and payload.get("url"):
        findings = [_simple_web_finding(payload, 1)]
    else:
        return output
    return json.dumps({"findings": findings, "issues": payload.get("issues") or []})


def _simple_web_finding(payload: Mapping[str, Any], index: int) -> dict[str, str]:
    return {
        "finding_id": f"find-{index:04d}",
        "title": str(payload["title"]),
        "url": str(payload["url"]),
        "source_type": _normalize_source_type(payload.get("source_type")),
        "technique_summary": str(
            payload.get("technique_summary")
            or "External source with a transferable optimization idea for the target module."
        ),
        "supporting_evidence": str(
            payload.get("supporting_evidence") or "Source identified by web search."
        ),
    }


def _extract_json_object_text(output: str) -> str:
    stripped = output.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise json.JSONDecodeError("no JSON object found", output, 0)
    return stripped[start : end + 1]


def _normalize_source_type(value: Any) -> str:
    allowed = {"paper", "blog", "docs", "issue", "pr", "talk", "codebase", "other"}
    normalized = str(value or "other").strip().lower()
    return normalized if normalized in allowed else "other"


def _extract_prompt_section(prompt: str, start: str, end: str) -> str:
    pattern = rf"{re.escape(start)}:\n(?P<body>.*?)(?:\n\n{re.escape(end)}:|\Z)"
    match = re.search(pattern, prompt, flags=re.DOTALL)
    if match is None:
        return "(not provided)"
    return match.group("body").strip()


def _extract_first_group(pattern: str, value: str, *, flags: int = 0) -> str | None:
    match = re.search(pattern, value, flags=flags)
    return match.group(1) if match else None


def _antigravity_research_tools(BuiltinTools: Any, *, enable_file_tools: bool) -> list[Any]:
    if enable_file_tools:
        return list(dict.fromkeys([*BuiltinTools.read_only(), BuiltinTools.SEARCH_WEB]))
    return [BuiltinTools.SEARCH_WEB]


def _structured_output_repair_prompt(attempt: int) -> str:
    prefix = (
        "The previous turn did not produce the required structured final output. "
        if attempt == 1
        else "The previous repair turn still did not produce structured output. "
    )
    return (
        prefix
        + "Finalize now using only the context already gathered in this conversation. "
        + "Do not call search, file, shell, or planning tools. Return a raw JSON "
        + "object only, with no markdown or explanation. Required shape: "
        + '{"findings":[{"finding_id":"find-0001","title":"...","url":"...",'
        + '"source_type":"paper|blog|docs|issue|pr|talk|codebase|other",'
        + '"technique_summary":"...","supporting_evidence":"..."}],"issues":[]}. '
        + "Do not return module summaries or keys such as module, name, path, "
        + "description, depends_on, files, or key_classes. If a finish tool is the "
        + "only available finalization channel, call finish with only findings and "
        + "issues; do not include helper metadata such as toolAction or toolSummary."
    )


async def _chat_text_or_structured_json(
    agent: Any,
    prompt: str,
    *,
    structured_output_repair_attempts: int = 0,
) -> str:
    response = await agent.chat(prompt)
    text = ""
    attempts = 1 + structured_output_repair_attempts
    for attempt in range(attempts):
        if hasattr(response, "structured_output"):
            structured = await response.structured_output()
            if structured is not None:
                return json.dumps(structured, default=_json_default)
        text = await response.text()
        if _looks_like_json_object(text) and _is_module_deep_research_json(text):
            return text
        if attempt + 1 >= attempts:
            return text
        response = await agent.chat(_structured_output_repair_prompt(attempt + 1))
    return text


def _looks_like_json_object(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def _is_module_deep_research_json(value: str) -> bool:
    try:
        parsed = json.loads(_extract_json_object_text(value))
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    if isinstance(parsed.get("findings"), list):
        return True
    return bool(parsed.get("title") and parsed.get("url"))


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@contextlib.contextmanager
def _patched_environ(env: Mapping[str, str] | None) -> Iterator[None]:
    if not env:
        yield
        return

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(dict(env))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _maybe_sse_normalizer(options: AntigravityExecOptions) -> Iterator[str | None]:
    if not (options.antigravity_base_url and options.normalize_sse_bytes_repr):
        yield None
        return
    timeout_seconds = options.timeout_seconds or _DEFAULT_PROXY_TIMEOUT_SECONDS
    upstream_headers = dict(options.extra_http_headers or {})
    if options.api_key_auth_mechanism == "bearer" and options.antigravity_api_key_env:
        key = os.environ.get(options.antigravity_api_key_env)
        if key:
            upstream_headers.setdefault("Authorization", f"Bearer {key}")
    with _sse_bytes_repr_normalizer(
        options.antigravity_base_url,
        timeout_seconds=timeout_seconds,
        upstream_headers=upstream_headers,
    ) as base_url:
        yield base_url


@contextlib.contextmanager
def _sse_bytes_repr_normalizer(
    upstream_base_url: str,
    *,
    timeout_seconds: float,
    upstream_headers: Mapping[str, str] | None = None,
) -> Iterator[str]:
    """Proxy Gemini API traffic and normalize LiteLLM bytes-repr SSE frames.

    The normalizer is intentionally generic: it forwards all paths/headers to
    the configured upstream and only rewrites malformed SSE lines whose payload
    is a Python bytes literal containing a nested Gemini SSE frame.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802
            self._proxy()

        def do_PUT(self) -> None:  # noqa: N802
            self._proxy()

        def do_DELETE(self) -> None:  # noqa: N802
            self._proxy()

        def do_PATCH(self) -> None:  # noqa: N802
            self._proxy()

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            upstream_url = urljoin(upstream_base_url.rstrip("/") + "/", self.path.lstrip("/"))
            headers = _forward_headers(dict(self.headers))
            headers.update(dict(upstream_headers or {}))
            try:
                request = Request(
                    upstream_url,
                    data=body,
                    headers=headers,
                    method=self.command,
                )
                try:
                    response = urlopen(request, timeout=timeout_seconds)
                except HTTPError as error:
                    response = error
                with contextlib.closing(response):
                    self.send_response(response.status)
                    content_type = response.headers.get("content-type", "")
                    is_event_stream = _is_streaming_response(
                        content_type=content_type,
                        request_path=self.path,
                    )
                    rewrite_json = _is_json_generate_response(
                        content_type=content_type,
                        request_path=self.path,
                    )
                    for key, value in response.headers.items():
                        if _should_forward_response_header(
                            key, rewrite_body=(is_event_stream or rewrite_json)
                        ):
                            self.send_header(key, value)
                    if is_event_stream:
                        self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Connection", "close")
                    self.close_connection = True
                    self.end_headers()
                    _debug_sse_proxy(
                        f"proxy path={self.path} "
                        f"content_type={response.headers.get('content-type', '')!r} "
                        f"is_event_stream={is_event_stream}"
                    )
                    if is_event_stream:
                        for chunk in _normalize_sse_lines(response):
                            _debug_sse_proxy(f"normalized chunk={chunk[:1000]!r}")
                            _write_chunked(self.wfile, chunk)
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    else:
                        body = response.read()
                        if rewrite_json:
                            body = _normalize_generate_content_body(body)
                        self.wfile.write(body)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001
                payload = f"SSE normalizer proxy failed: {type(exc).__name__}: {exc}".encode()
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)




def _is_streaming_response(*, content_type: str, request_path: str) -> bool:
    lowered_type = content_type.lower()
    lowered_path = request_path.lower()
    return (
        "text/event-stream" in lowered_type
        or "streamgeneratecontent" in lowered_path
        or "alt=sse" in lowered_path
    )


def _is_json_generate_response(*, content_type: str, request_path: str) -> bool:
    return (
        "application/json" in content_type.lower()
        and "generatecontent" in request_path.lower()
    )


def _normalize_generate_content_body(body: bytes) -> bytes:
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(obj, dict) or "candidates" not in obj:
        return body
    _drop_inert_candidate_parts(obj)
    if _has_no_candidate_parts(obj) or _is_empty_terminal_candidate_chunk(obj):
        candidates = obj.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            candidates[0]["content"] = {
                "role": "model",
                "parts": [{"text": "No content returned."}],
            }
            candidates[0].setdefault("finishReason", "STOP")
    _normalize_tool_call_args(obj)
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")




def _debug_sse_proxy(message: str) -> None:
    path = os.environ.get("SPOTLIGHTS_ANTIGRAVITY_SSE_DEBUG_PATH")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except OSError:
        return


_HOP_BY_HOP_REQUEST_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

_HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_REWRITTEN_RESPONSE_HEADERS = {"content-encoding", "content-length"}


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    forwarded = {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_REQUEST_HEADERS | {"accept-encoding"}
    }
    forwarded.setdefault("Accept-Encoding", "identity")
    return forwarded


def _should_forward_response_header(key: str, *, rewrite_body: bool) -> bool:
    lowered = key.lower()
    if lowered in _HOP_BY_HOP_RESPONSE_HEADERS:
        return False
    return not (rewrite_body and lowered in _REWRITTEN_RESPONSE_HEADERS)


def _normalize_sse_lines(lines: Iterator[bytes]) -> Iterator[bytes]:
    pending_bytes_repr_fragments: list[str] = []
    pending_plain_json_fragments: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace")
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if stripped in {"data: [DONE]", "[DONE]"}:
            continue

        if pending_plain_json_fragments:
            fragment = _bytes_repr_fragment(stripped)
            if fragment is not None:
                body, complete = fragment
                pending_plain_json_fragments.append(body)
                if not complete:
                    continue
            else:
                pending_plain_json_fragments.append(_sse_data_payload_fragment(stripped))
            combined = "".join(pending_plain_json_fragments)
            normalized_plain = normalize_litellm_vertex_sse_line(combined)
            if normalized_plain is not None:
                pending_plain_json_fragments.clear()
                if normalized_plain:
                    yield normalized_plain
                continue
            if _is_parseable_sse_json(combined):
                pending_plain_json_fragments.clear()
                yield _ensure_sse_event_bytes(combined)
            continue

        if pending_bytes_repr_fragments:
            fragment = _bytes_repr_fragment(stripped)
            if fragment is None:
                pending_bytes_repr_fragments.clear()
                yield raw_line
                continue
            body, complete = fragment
            pending_bytes_repr_fragments.append(body)
            if not complete:
                continue
            normalized = normalize_litellm_sse_bytes_repr_line(
                "data: b'" + "".join(pending_bytes_repr_fragments) + "'"
            )
            pending_bytes_repr_fragments.clear()
            if normalized is not None:
                if normalized:
                    yield normalized
                continue
            yield raw_line
            continue

        normalized = normalize_litellm_sse_bytes_repr_line(stripped)
        if normalized is not None:
            if normalized:
                yield normalized
            continue

        fragment = _bytes_repr_fragment(stripped)
        if fragment is not None:
            body, complete = fragment
            if not complete:
                pending_bytes_repr_fragments.append(body)
                continue
            # A complete bytes-repr line that did not normalize to SSE is an
            # orphaned continuation fragment (usually a split opaque provider
            # field). Forwarding it as SSE poisons the SDK stream.
            continue

        normalized_plain = normalize_litellm_vertex_sse_line(stripped)
        if normalized_plain is not None:
            if normalized_plain:
                yield normalized_plain
            continue
        if stripped.startswith("data: {") and not _is_parseable_sse_json(stripped):
            pending_plain_json_fragments.append(stripped)
            continue
        yield raw_line




def _sse_data_payload_fragment(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("data:"):
        return stripped.removeprefix("data:").lstrip()
    return stripped


def _is_parseable_sse_json(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return False
    payload = stripped.removeprefix("data:").strip()
    if not payload.startswith("{"):
        return False
    try:
        json.loads(payload)
    except json.JSONDecodeError:
        repaired = _repair_litellm_embedded_bytes_markers(payload)
        if repaired is None:
            return False
        try:
            json.loads(repaired)
        except json.JSONDecodeError:
            return False
    return True


def _ensure_sse_event_bytes(line: str) -> bytes:
    stripped = line.strip()
    if stripped.endswith(("\r\n\r\n", "\n\n")):
        return stripped.encode()
    return f"{stripped}\r\n\r\n".encode()


def _bytes_repr_fragment(line: str) -> tuple[str, bool] | None:
    stripped = line.strip()
    if not stripped.startswith("data: b"):
        return None
    literal = stripped.removeprefix("data: ")
    if len(literal) < 3 or literal[0] != "b" or literal[1] not in {"'", '"'}:
        return None
    quote = literal[1]
    body = literal[2:]
    if body.endswith(quote) and not _ends_with_escaped_quote(body, quote):
        return body[:-1], True
    return body, False


def _ends_with_escaped_quote(body: str, quote: str) -> bool:
    if not body.endswith(quote):
        return False
    backslashes = 0
    index = len(body) - 2
    while index >= 0 and body[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def normalize_litellm_sse_bytes_repr_line(line: str) -> bytes | None:
    """Unwrap `data: b'...'` LiteLLM frames back to raw stream bytes."""
    stripped = line.strip()
    if not stripped.startswith("data: b"):
        return None
    literal = stripped.removeprefix("data: ")
    try:
        payload = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        payload = _repair_litellm_bytes_repr_payload(literal)
        if payload is None:
            return None
    if not isinstance(payload, bytes):
        return None
    decoded = payload.decode("utf-8", errors="replace")
    if decoded.strip() in {"data: [DONE]", "[DONE]"}:
        return b""
    if not decoded.lstrip().startswith(("data:", "event:", "id:", "retry:")):
        return None
    normalized_decoded = _normalize_litellm_vertex_stream_payload(decoded)
    if normalized_decoded is None:
        return b""
    if (
        normalized_decoded == decoded
        and decoded.lstrip().startswith("data: {")
        and not _is_parseable_sse_json(decoded)
    ):
        return None
    return normalized_decoded.encode()




def _repair_litellm_bytes_repr_payload(literal: str) -> bytes | None:
    """Repair LiteLLM streams that concatenate split ``data: b'...'`` frames.

    Some Gemini-compatible gateways split a long provider chunk inside an opaque
    string such as ``thoughtSignature`` and emit the next slice as another
    Python-bytes-repr SSE data field. By the time the SDK reads the stream this
    can look like ``data: b'data: {...Mdata: b'5f...'}'``. Treat those embedded
    markers as transport artifacts and reassemble the original bytes payload.
    """
    if len(literal) < 3 or literal[0] != "b" or literal[1] not in {"'", '"'}:
        return None
    quote = literal[1]
    body = literal[2:]
    if f"data: b{quote}" not in body:
        return None
    if body.endswith(quote) and not _ends_with_escaped_quote(body, quote):
        body = body[:-1]
    body = body.replace(f"data: b{quote}", "")
    try:
        repaired = ast.literal_eval(f"b{quote}{body}{quote}")
    except (SyntaxError, ValueError):
        return None
    return repaired if isinstance(repaired, bytes) else None


def normalize_litellm_vertex_sse_line(line: str) -> bytes | None:
    """Normalize Gemini SSE JSON fields that gateway streams may include."""
    stripped = line.strip()
    if not stripped.startswith("data:") or stripped.startswith("data: b"):
        return None
    normalized = _normalize_litellm_vertex_stream_payload(stripped)
    if normalized is None:
        return b""
    if normalized == stripped:
        return None
    return normalized.encode()


def _normalize_litellm_vertex_stream_payload(decoded_sse: str) -> str | None:
    """Remove provider fields/chunks that older Gemini SDK consumers reject."""
    prefix = "data:"
    stripped = decoded_sse.lstrip()
    if not stripped.startswith(prefix):
        return decoded_sse
    payload = stripped.removeprefix(prefix).strip()
    if not payload.startswith("{"):
        return decoded_sse
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        repaired_payload = _repair_litellm_embedded_bytes_markers(payload)
        if repaired_payload is None:
            return decoded_sse
        try:
            obj = json.loads(repaired_payload)
        except json.JSONDecodeError:
            return decoded_sse
    if "candidates" not in obj and "promptFeedback" not in obj:
        return decoded_sse
    _drop_inert_candidate_parts(obj)
    if _has_no_candidate_parts(obj):
        return None
    if _is_empty_terminal_candidate_chunk(obj):
        return None
    _normalize_tool_call_args(obj)
    obj = {
        key: value
        for key, value in obj.items()
        if key in {"candidates", "promptFeedback"}
    }
    return f"{prefix} {json.dumps(obj, separators=(',', ':'))}\r\n\r\n"




def _repair_litellm_embedded_bytes_markers(payload: str) -> str | None:
    if "data: b" not in payload:
        return None
    repaired = payload.replace("data: b'", "").replace('data: b"', "")
    for suffix in ("\\r\\n\\r\\n'", '\\r\\n\\r\\n"', "\\n\\n'", '\\n\\n"'):
        if repaired.endswith(suffix):
            repaired = repaired[: -len(suffix)]
            break
    for suffix in ("\r\n\r\n'", '\r\n\r\n"', "\n\n'", '\n\n"'):
        if repaired.endswith(suffix):
            repaired = repaired[: -len(suffix)]
            break
    return repaired


def _normalize_tool_call_args(obj: Any) -> None:
    """Drop gateway helper metadata from Antigravity tool-call chunks."""
    candidates = obj.get("candidates") if isinstance(obj, dict) else None
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parts = ((candidate.get("content") or {}).get("parts") or [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall")
            if not isinstance(function_call, dict):
                continue
            args = function_call.get("args")
            if isinstance(args, dict):
                args.pop("toolAction", None)
                args.pop("toolSummary", None)


def _drop_inert_candidate_parts(obj: Any) -> None:
    candidates = obj.get("candidates") if isinstance(obj, dict) else None
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        content["parts"] = [part for part in parts if not _is_inert_candidate_part(part)]


def _is_inert_candidate_part(part: Any) -> bool:
    """Drop chunks that make older Antigravity harnesses reject the stream.

    Some Gemini-compatible gateways stream bookkeeping-only parts such as
    thought-only chunks, empty text chunks, or standalone thought signatures.
    They carry no user-visible text and no tool call, and the local harness can
    surface them as "model output must contain either output text or tool
    calls". Preserve function/tool-call parts and non-empty text, including
    thought signatures attached to those meaningful parts.
    """
    if not isinstance(part, dict):
        return False
    if part.get("functionCall"):
        return False
    meaningful_payload_keys = {
        "functionResponse",
        "executableCode",
        "codeExecutionResult",
        "inlineData",
        "fileData",
    }
    if any(part.get(key) for key in meaningful_payload_keys):
        return False
    text = part.get("text")
    if isinstance(text, str) and text.strip() and part.get("thought") is not True:
        return False
    return True


def _has_no_candidate_parts(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    candidates = obj.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False
        parts = ((candidate.get("content") or {}).get("parts") or [])
        if parts:
            return False
        if candidate.get("finishReason"):
            return False
    return "promptFeedback" not in obj


def _is_empty_terminal_candidate_chunk(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    candidates = obj.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    has_finish = False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False
        has_finish = has_finish or bool(candidate.get("finishReason"))
        parts = ((candidate.get("content") or {}).get("parts") or [])
        if not isinstance(parts, list):
            return False
        for part in parts:
            if not isinstance(part, dict):
                return False
            if part.get("text"):
                return False
            if part.get("functionCall"):
                return False
    return has_finish


def _drop_keys_recursive(value: Any, keys: set[str]) -> None:
    if isinstance(value, dict):
        for key in keys:
            value.pop(key, None)
        for child in value.values():
            _drop_keys_recursive(child, keys)
    elif isinstance(value, list):
        for child in value:
            _drop_keys_recursive(child, keys)


def _write_chunked(wfile: Any, chunk: bytes) -> None:
    if not chunk:
        return
    wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
    wfile.write(chunk)
    wfile.write(b"\r\n")
    wfile.flush()


__all__ = [
    "AntigravityExecClient",
    "AntigravityExecOptions",
    "DEFAULT_LITELLM_ANTIGRAVITY_MODEL",
    "normalize_litellm_sse_bytes_repr_line",
    "normalize_litellm_vertex_sse_line",
]
