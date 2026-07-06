import json

import pytest

from spotlights_engine.costing.manifest import build_run_manifest
from spotlights_engine.costing.rates import ModelRate, compute_cost
from spotlights_engine.costing.records import UsageRecord
from spotlights_engine.costing.usage import (
    AgentUsage,
    claude_usage_from_payload,
    claude_usage_from_stream,
    codex_usage_from_stream,
)


def test_claude_usage_from_payload_and_stream_captures_cache_and_cost() -> None:
    payload = {
        "type": "result",
        "duration_api_ms": 2500,
        "total_cost_usd": 9.99,
        "model": "claude-opus-4-8",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 40,
        },
    }

    usage = claude_usage_from_payload(payload)
    assert usage is not None
    assert usage.input == 10
    assert usage.output == 20
    assert usage.cache_read == 30
    assert usage.cache_create == 40
    assert usage.model == "claude-opus-4-8"
    assert usage.api_time_s == 2.5
    assert usage.cli_reported_cost_usd == 9.99

    stream_usage = claude_usage_from_stream(
        json.dumps({"type": "message"}) + "\n" + json.dumps(payload) + "\n"
    )
    assert stream_usage == usage


def test_codex_usage_keeps_latest_cumulative_payload_and_splits_cached_input() -> None:
    first = {"usage": {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 25}}
    latest = {
        "model": "gpt-5.5",
        "usage": {
            "input_tokens": 200,
            "output_tokens": 30,
            "cached_input_tokens": 50,
        },
    }

    usage = codex_usage_from_stream(json.dumps(first) + "\n" + json.dumps(latest))

    assert usage is not None
    assert usage.input == 150
    assert usage.output == 30
    assert usage.cache_read == 50
    assert usage.cache_create == 0
    assert usage.model == "gpt-5.5"


def test_compute_cost_uses_rates_and_reports_missing_models() -> None:
    records = [
        UsageRecord.from_usage(
            AgentUsage(input=10, output=5, cache_read=3, cache_create=2, model="m1"),
            step="candidate_discovery",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=0,
            invocation_id="i0",
            cli="codex",
            role="candidate_discovery",
        ),
        UsageRecord.from_usage(
            AgentUsage(input=100, model="missing"),
            step="module_deep_research",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=1,
            invocation_id="i1",
            cli="claude",
            role="deep_research",
        ),
    ]

    summary = compute_cost(
        records,
        {
            "openai:m1": ModelRate(
                input=0.01,
                output=0.02,
                cache_read=0.001,
                cache_create=0.005,
                note="test",
            )
        },
    )

    assert summary.amount_usd == pytest.approx(0.213)
    assert summary.unpriced_models == ["anthropic:missing"]
    assert "PARTIAL cost" in summary.rate_note


def test_run_manifest_groups_models_and_totals_tokens() -> None:
    records = [
        UsageRecord.from_usage(
            AgentUsage(input=10, output=5, cache_read=3, cache_create=2, model="m1"),
            step="candidate_discovery",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=0,
            invocation_id="i0",
            cli="codex",
            role="candidate_discovery",
            fallback_api_time_s=1.25,
        ),
        UsageRecord.from_usage(
            AgentUsage(input=7, output=8, model="m1", api_time_s=2.0),
            step="candidate_discovery",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=1,
            invocation_id="i1",
            cli="codex",
            role="candidate_discovery",
        ),
    ]
    summary = compute_cost(
        records,
        {
            "openai:m1": ModelRate(
                input=1.0,
                output=1.0,
                cache_read=1.0,
                cache_create=1.0,
            )
        },
    )

    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-06T00:00:00Z",
        objective="find spots",
        provenance={
            "repo_url": "https://example.test/repo.git",
            "target_commit_sha": "abc",
            "spotlights_commit_sha": "def",
        },
        config_fingerprint={"x": 1},
        records=records,
        cost=summary,
        wall_clock_s=12.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=4,
        module_status={"SUCCEEDED": 1},
        notes=[],
    )

    assert manifest.total_tokens == 35
    assert manifest.timing.api_time_s == 3.25
    assert manifest.models_used[0].usage.input == 17
    assert manifest.models_used[0].usage.output == 13
    assert manifest.cost.amount_usd == 35.0
