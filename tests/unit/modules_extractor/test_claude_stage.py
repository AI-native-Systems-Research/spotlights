"""Unit tests for `claude_stage.api_failure_reason`.

The classifier decides which CLI failures the orchestrator may retry: only
API-transport deaths (rate limit, request timeout, server error), judged from
the *last* stream event so a recovered early `api_retry` never marks a later
unrelated failure as retryable.
"""

from __future__ import annotations

import json

from spotlights_engine.modules_extractor.claude_stage import api_failure_reason


def _stream(*events: dict) -> bytes:
    return "".join(json.dumps(e) + "\n" for e in events).encode("utf-8")


def _error_result(text: str) -> dict:
    return {"type": "result", "subtype": "success", "is_error": True, "result": text}


_RETRY_429 = {
    "type": "system",
    "subtype": "api_retry",
    "attempt": 10,
    "max_retries": 10,
    "error_status": 429,
    "error": "rate_limit",
}


def test_429_result_text_is_rate_limit() -> None:
    stream = _stream(
        _RETRY_429,
        _error_result(
            "API Error: Request rejected (429) · 429: Rate limit exceeded "
            "for api_key: abc. Limit type: tokens."
        ),
    )
    assert api_failure_reason(stream) == "rate_limit (429)"


def test_request_timed_out_result_text() -> None:
    stream = _stream(_RETRY_429, _error_result("Request timed out"))
    assert api_failure_reason(stream) == "request_timeout"


def test_overloaded_result_text() -> None:
    stream = _stream(_error_result("API Error (overloaded_error): Overloaded"))
    assert api_failure_reason(stream) == "overloaded"


def test_server_error_status_in_result_text() -> None:
    stream = _stream(_error_result("API Error: Request rejected (529)"))
    assert api_failure_reason(stream) == "api_error (529)"


def test_stream_cut_off_mid_api_retry() -> None:
    stream = _stream({"type": "system", "subtype": "init"}, _RETRY_429)
    assert api_failure_reason(stream) == "rate_limit (429)"


def test_stream_cut_off_after_terminal_assistant_transport_error() -> None:
    stream = _stream(
        _RETRY_429,
        {"type": "assistant", "error": "server_error", "message": {}},
    )
    assert api_failure_reason(stream) == "server_error"


def test_recovered_retry_then_unrelated_error_is_not_api_failure() -> None:
    stream = _stream(_RETRY_429, _error_result("Reached max turns"))
    assert api_failure_reason(stream) is None


def test_client_error_status_is_not_retryable() -> None:
    stream = _stream(
        _error_result("API Error: Request rejected (401) · Invalid API key")
    )
    assert api_failure_reason(stream) is None


def test_successful_result_is_not_api_failure() -> None:
    stream = _stream(
        {"type": "result", "subtype": "success", "is_error": False, "result": "{}"}
    )
    assert api_failure_reason(stream) is None


def test_empty_and_garbage_streams() -> None:
    assert api_failure_reason(b"") is None
    assert api_failure_reason(b"not json\n[1, 2]\n") is None
