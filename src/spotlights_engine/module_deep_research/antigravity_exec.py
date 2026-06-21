"""Small wrapper around the Google Antigravity SDK agent API."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult

DEFAULT_LITELLM_ANTIGRAVITY_MODEL = "gcp/gemini-3.1-pro-preview"


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

        if key and opt.api_key_auth_mechanism == "bearer":
            headers.setdefault("Authorization", f"Bearer {key}")
        elif key and opt.api_key_auth_mechanism == "x-goog-api-key":
            headers.setdefault("x-goog-api-key", key)

        kwargs: dict[str, Any] = {}
        if opt.antigravity_base_url:
            kwargs["base_url"] = opt.antigravity_base_url
        if headers:
            kwargs["http_headers"] = headers
        if key:
            kwargs["api_key"] = key
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

    async def _run_async(self, prompt: str) -> str:
        coro = self._run_antigravity(prompt)
        if self.options.timeout_seconds is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=self.options.timeout_seconds)

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
        cwd = Path(opt.cwd).expanduser().resolve()
        workspaces = [str(Path(path).expanduser().resolve()) for path in (opt.workspaces or [cwd])]
        endpoint_kwargs = self.build_model_endpoint_kwargs()
        model = opt.model
        endpoint_model = model or (
            DEFAULT_LITELLM_ANTIGRAVITY_MODEL if opt.antigravity_base_url else None
        )
        research_tools = [*BuiltinTools.read_only(), BuiltinTools.SEARCH_WEB]
        config_kwargs: dict[str, Any] = {
            "system_instructions": opt.system_instructions,
            "capabilities": CapabilitiesConfig(enabled_tools=research_tools),
            "workspaces": workspaces,
            "response_schema": opt.response_schema,
        }
        if endpoint_model and endpoint_kwargs:
            with _maybe_sse_normalizer(opt) as normalized_base_url:
                if normalized_base_url:
                    endpoint_kwargs = dict(endpoint_kwargs)
                    endpoint_kwargs["base_url"] = normalized_base_url
                config_kwargs["models"] = [
                    ModelTarget(
                        name=endpoint_model,
                        types=[ModelType.TEXT, ModelType.IMAGE],
                        endpoint=GeminiAPIEndpoint(**endpoint_kwargs),
                    )
                ]
                async with Agent(LocalAgentConfig(**config_kwargs)) as agent:
                    return await _chat_text_or_structured_json(agent, prompt)
        if model:
            config_kwargs["model"] = model

        async with Agent(LocalAgentConfig(**config_kwargs)) as agent:
            return await _chat_text_or_structured_json(agent, prompt)


async def _chat_text_or_structured_json(agent: Any, prompt: str) -> str:
    response = await agent.chat(prompt)
    structured_output = await response.structured_output()
    if structured_output is not None:
        return json.dumps(structured_output, indent=2, sort_keys=True)
    return await response.text()


@contextlib.contextmanager
def _maybe_sse_normalizer(options: AntigravityExecOptions) -> Iterator[str | None]:
    if not (options.antigravity_base_url and options.normalize_sse_bytes_repr):
        yield None
        return
    with _sse_bytes_repr_normalizer(options.antigravity_base_url) as base_url:
        yield base_url


@contextlib.contextmanager
def _sse_bytes_repr_normalizer(upstream_base_url: str) -> Iterator[str]:
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

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            upstream_url = urljoin(upstream_base_url.rstrip("/") + "/", self.path.lstrip("/"))
            headers = _forward_headers(dict(self.headers))
            try:
                with httpx.Client(timeout=None) as client:
                    with client.stream(
                        self.command,
                        upstream_url,
                        headers=headers,
                        content=body,
                    ) as response:
                        self.send_response(response.status_code)
                        for key, value in response.headers.items():
                            if key.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS:
                                self.send_header(key, value)
                        self.send_header("Connection", "close")
                        self.close_connection = True
                        self.end_headers()
                        is_event_stream = "text/event-stream" in response.headers.get(
                            "content-type", ""
                        ).lower()
                        if is_event_stream:
                            for chunk in _normalize_sse_lines(response.iter_lines()):
                                self.wfile.write(chunk)
                                self.wfile.flush()
                        else:
                            for chunk in response.iter_bytes():
                                self.wfile.write(chunk)
                                self.wfile.flush()
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

_HOP_BY_HOP_RESPONSE_HEADERS = _HOP_BY_HOP_REQUEST_HEADERS | {
    "content-encoding",
    "content-length",
}


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_REQUEST_HEADERS
    }


def _normalize_sse_lines(lines: Iterator[str]) -> Iterator[bytes]:
    for line in lines:
        if not line or line.strip() in {"data: [DONE]", "[DONE]"}:
            continue
        normalized = normalize_litellm_sse_bytes_repr_line(line)
        if normalized is not None:
            if normalized:
                yield normalized
        else:
            yield f"{line}\n\n".encode()


def normalize_litellm_sse_bytes_repr_line(line: str) -> bytes | None:
    """Unwrap `data: b'...'` LiteLLM frames back to raw stream bytes."""
    stripped = line.strip()
    if not stripped.startswith("data: b"):
        return None
    literal = stripped.removeprefix("data: ")
    try:
        payload = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(payload, bytes):
        return None
    decoded = payload.decode("utf-8", errors="replace")
    if decoded.strip() in {"data: [DONE]", "[DONE]"}:
        return b""
    return decoded.encode()


__all__ = [
    "AntigravityExecClient",
    "AntigravityExecOptions",
    "DEFAULT_LITELLM_ANTIGRAVITY_MODEL",
    "normalize_litellm_sse_bytes_repr_line",
]
