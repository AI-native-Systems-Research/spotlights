import json
from pathlib import Path

import pytest

from spotlights_engine.costing.manifest import build_run_manifest
from spotlights_engine.costing.rates import (
    EXTERNAL_RATES_ENV_VAR,
    ModelRate,
    compute_cost,
    load_external_rates,
    load_rates,
)
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
        accumulated_duration_s=99.5,
        candidates_path="/tmp/out/index.md",
        num_candidates=4,
        module_status={"SUCCEEDED": 1},
        notes=[],
    )

    assert manifest.total_tokens == 35
    assert manifest.timing.api_time_s == 3.25
    assert manifest.timing.wall_clock_s == 12.0
    assert manifest.timing.accumulated_duration_s == 99.5
    assert manifest.models_used[0].usage.input == 17
    assert manifest.models_used[0].usage.output == 13
    assert manifest.cost.amount_usd == 35.0


def _opus_record() -> UsageRecord:
    """One Opus 4.8 record priced by both bundled tables."""
    return UsageRecord.from_usage(
        AgentUsage(
            input=1000,
            output=1000,
            cache_read=1000,
            cache_create=1000,
            model="aws/claude-opus-4-8",
        ),
        step="module_deep_research",
        module_qualified_name="pkg/a",
        session_index=1,
        invocation_index=0,
        invocation_id="i0",
        cli="claude",
        role="deep_research",
    )


def test_load_external_rates_returns_bundled_table() -> None:
    external = load_external_rates()
    assert "anthropic:aws/claude-opus-4-8" in external
    assert "openai:codex" in external
    # Public Opus list price ($5/MTok input) exceeds the contracted rate.
    assert external["anthropic:aws/claude-opus-4-8"].input == pytest.approx(
        0.000005
    )


def test_load_external_rates_env_var_and_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "ext.json"
    custom.write_text(
        json.dumps(
            {
                "anthropic:aws/claude-opus-4-8": {
                    "input": 0.1,
                    "output": 0.2,
                    "cache_read": 0.0,
                    "cache_create": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    # Env-var precedence over the bundled default.
    monkeypatch.setenv(EXTERNAL_RATES_ENV_VAR, str(custom))
    from_env = load_external_rates()
    assert from_env["anthropic:aws/claude-opus-4-8"].input == pytest.approx(0.1)

    # Explicit path wins over the env var.
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {
                "anthropic:aws/claude-opus-4-8": {
                    "input": 0.9,
                    "output": 0.2,
                    "cache_read": 0.0,
                    "cache_create": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    from_path = load_external_rates(other)
    assert from_path["anthropic:aws/claude-opus-4-8"].input == pytest.approx(0.9)


def test_compute_cost_source_label_and_table_divergence() -> None:
    records = [_opus_record()]
    contracted = compute_cost(records, load_rates())
    external = compute_cost(
        records, load_external_rates(), source="public-api-rate-table"
    )

    assert contracted.source == "contracted-rate-table"
    assert external.source == "public-api-rate-table"
    # Same records, different tables -> different totals. For an Opus-only
    # record, public list price is strictly above the contracted rate.
    assert external.amount_usd > contracted.amount_usd


def test_run_manifest_carries_external_cost() -> None:
    records = [_opus_record()]
    contracted = compute_cost(records, load_rates())
    external = compute_cost(
        records, load_external_rates(), source="public-api-rate-table"
    )

    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-08T00:00:00Z",
        objective="find spots",
        provenance={},
        config_fingerprint={},
        records=records,
        cost=contracted,
        external_cost=external,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=0,
        module_status={},
        notes=[],
    )

    assert manifest.external_cost is not None
    assert manifest.external_cost.source == "public-api-rate-table"
    assert manifest.external_cost.amount_usd == pytest.approx(external.amount_usd)
    assert manifest.external_cost.amount_usd > manifest.cost.amount_usd


def test_run_manifest_external_cost_defaults_to_none() -> None:
    summary = compute_cost([], {})
    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-08T00:00:00Z",
        objective="find spots",
        provenance={},
        config_fingerprint={},
        records=[],
        cost=summary,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=0,
        module_status={},
        notes=[],
    )

    assert manifest.external_cost is None


def test_old_manifest_dict_without_external_cost_validates() -> None:
    from spotlights_engine.costing.manifest import RunManifest

    summary = compute_cost([], {})
    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-08T00:00:00Z",
        objective="find spots",
        provenance={},
        config_fingerprint={},
        records=[],
        cost=summary,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=0,
        module_status={},
        notes=[],
    )
    payload = manifest.model_dump(mode="json")
    payload.pop("external_cost", None)

    # A persisted manifest predating external_cost still validates.
    reloaded = RunManifest.model_validate(payload)
    assert reloaded.external_cost is None


def test_manifest_external_unpriced_note() -> None:
    records = [_opus_record()]
    contracted = compute_cost(records, load_rates())
    # An external table that prices nothing -> external unpriced note.
    external = compute_cost(records, {}, source="public-api-rate-table")

    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-08T00:00:00Z",
        objective="find spots",
        provenance={
            "repo_url": "r",
            "target_commit_sha": "a",
            "spotlights_commit_sha": "b",
        },
        config_fingerprint={},
        records=records,
        cost=contracted,
        external_cost=external,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=0,
        module_status={},
        notes=[],
    )

    assert "external: unpriced models excluded from cost:" in manifest.notes


def test_run_manifest_accumulated_duration_defaults_to_zero() -> None:
    summary = compute_cost([], {})
    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-06T00:00:00Z",
        objective="find spots",
        provenance={},
        config_fingerprint={},
        records=[],
        cost=summary,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=0,
        module_status={},
        notes=[],
    )

    assert manifest.timing.accumulated_duration_s == 0.0
