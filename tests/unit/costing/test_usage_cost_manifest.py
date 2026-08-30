import json
from pathlib import Path

import pytest

from spotlights_engine.costing.manifest import (
    RunManifestModelsRequested,
    build_run_manifest,
)
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


def test_compute_cost_strips_context_window_tag_from_model_id() -> None:
    """A "[1m]" context-variant id prices off its base model's rate row."""
    records = [
        UsageRecord.from_usage(
            AgentUsage(input=100, output=10, model="claude-opus-4-8[1m]"),
            step="module_deep_research",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=0,
            invocation_id="i0",
            cli="claude",
            role="deep_research",
        ),
    ]

    summary = compute_cost(
        records,
        {
            "anthropic:claude-opus-4-8": ModelRate(
                input=0.01,
                output=0.02,
                cache_read=0.001,
                cache_create=0.005,
                note="test",
            )
        },
    )

    assert summary.amount_usd == pytest.approx(100 * 0.01 + 10 * 0.02)
    assert summary.unpriced_models == []
    assert "PARTIAL cost" not in summary.rate_note


def test_compute_cost_strips_litellm_route_prefix_from_model_id() -> None:
    """A CLI reporting a plain model id prices against a table keyed with the
    LiteLLM route prefix — and vice versa. Pricing is per model, not per route
    (bedrock vs direct-API), so `aws/claude-opus-5` and `claude-opus-5` must
    hit the same row from either direction."""

    def _record(model: str, idx: int) -> UsageRecord:
        return UsageRecord.from_usage(
            AgentUsage(input=100, output=10, model=model),
            step="module_deep_research",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=idx,
            invocation_id=f"i{idx}",
            cli="claude",
            role="deep_research",
        )

    rate = ModelRate(input=0.01, output=0.02, cache_read=0.0, cache_create=0.0)

    # CLI reports the unprefixed form (the user's DAM-image bug); table has the
    # route-prefixed key. Both sides canonicalize on lookup, so it hits.
    unprefixed_hit = compute_cost(
        [_record("claude-opus-5", 0)],
        {"anthropic:aws/claude-opus-5": rate},
    )
    assert unprefixed_hit.amount_usd == pytest.approx(100 * 0.01 + 10 * 0.02)
    assert unprefixed_hit.unpriced_models == []

    # Reverse: CLI reports the prefixed form, table has the unprefixed key
    # (what the loader normalizes to). Also hits.
    prefixed_hit = compute_cost(
        [_record("aws/claude-opus-5", 0)],
        {"anthropic:claude-opus-5": rate},
    )
    assert prefixed_hit.amount_usd == pytest.approx(100 * 0.01 + 10 * 0.02)
    assert prefixed_hit.unpriced_models == []

    # Context tag + route prefix together also collapse to the canonical row.
    both_tags = compute_cost(
        [_record("aws/claude-opus-5[1m]", 0)],
        {"anthropic:claude-opus-5": rate},
    )
    assert both_tags.amount_usd == pytest.approx(100 * 0.01 + 10 * 0.02)

    # An unknown-prefixed model id (not in the allowlist) is left alone — the
    # loader canonicalizes both sides, so a table keyed `hf-org/model-x` and a
    # CLI reporting `hf-org/model-x` still match without eating the slash.
    hf = compute_cost(
        [_record("hf-org/model-x", 0)],
        {"anthropic:hf-org/model-x": rate},
    )
    assert hf.amount_usd == pytest.approx(100 * 0.01 + 10 * 0.02)

    # Stacked LiteLLM route prefixes collapse in one pass. LiteLLM's OpenRouter
    # provider emits `openrouter/anthropic/claude-opus-5`; both segments are
    # LiteLLM route labels and must strip together, or the model silently drops
    # into `unpriced_models` and the cost total goes partial with no error.
    stacked = compute_cost(
        [_record("openrouter/anthropic/claude-opus-5", 0)],
        {"anthropic:claude-opus-5": rate},
    )
    assert stacked.amount_usd == pytest.approx(100 * 0.01 + 10 * 0.02)
    assert stacked.unpriced_models == []


def test_canonical_model_id_preserves_degenerate_inputs() -> None:
    """An id that strips to empty is returned unchanged so a malformed id lands
    in `unpriced_models` under its own visibly-broken key rather than every bad
    record silently colliding on the empty-model rate key `"provider:"`."""
    from spotlights_engine.costing.rates import canonical_model_id

    # Pure context tag: stripping the "[1m]" leaves nothing behind.
    assert canonical_model_id("[1m]") == "[1m]"
    # Pure route prefix: stripping the "aws/" leaves nothing behind.
    assert canonical_model_id("aws/") == "aws/"
    # An id that legitimately reduces to nothing (empty in, empty out) is fine.
    assert canonical_model_id("") == ""


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


def test_run_manifest_models_used_matches_by_model_for_route_prefixed_id() -> None:
    """`models_used` and `cost.by_model` must carry the same model string for
    the same record — the docstring on `_group_key` promises 1:1 lockstep. A
    route-prefixed or context-tagged id from the CLI (`aws/claude-opus-5[1m]`
    in the DAM sandbox) is canonicalized on the cost side; the models_used
    side must apply the same canonicalization or the two sections of the run
    manifest disagree for the same underlying model.
    """
    record = UsageRecord.from_usage(
        AgentUsage(input=100, output=10, model="aws/claude-opus-5[1m]"),
        step="module_deep_research",
        module_qualified_name="pkg/a",
        session_index=1,
        invocation_index=0,
        invocation_id="i0",
        cli="claude",
        role="deep_research",
    )
    summary = compute_cost(
        [record],
        {
            "anthropic:claude-opus-5": ModelRate(
                input=0.01, output=0.02, cache_read=0.0, cache_create=0.0
            )
        },
    )

    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-06T00:00:00Z",
        objective="find spots",
        provenance={},
        config_fingerprint={},
        records=[record],
        cost=summary,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=0,
        module_status={"SUCCEEDED": 1},
        notes=[],
    )

    assert len(manifest.models_used) == 1
    assert len(manifest.cost.by_model) == 1
    assert manifest.models_used[0].model == manifest.cost.by_model[0].model
    assert manifest.models_used[0].model == "claude-opus-5"


def _opus_record() -> UsageRecord:
    """One Opus 4.8 record priced by both bundled tables. The bedrock `aws/`
    prefix here exercises the loader's route-prefix canonicalization — the
    bundled tables key their rows without the prefix."""
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
    # Bundled keys are canonical (no LiteLLM route prefix); the loader would
    # strip a prefix if one were present.
    assert "anthropic:claude-opus-4-8" in external
    assert "openai:codex" in external
    # Public Opus list price ($5/MTok input) exceeds the contracted rate.
    assert external["anthropic:claude-opus-4-8"].input == pytest.approx(0.000005)


def test_load_external_rates_env_var_and_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A user-supplied table keyed with the LiteLLM `aws/` prefix must still be
    # reachable — the loader normalizes both sides of the lookup, so writing
    # `anthropic:aws/claude-opus-4-8` in the file works even though runtime
    # lookups produce `anthropic:claude-opus-4-8`.
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
    monkeypatch.setenv(EXTERNAL_RATES_ENV_VAR, str(custom))
    from_env = load_external_rates()
    assert from_env["anthropic:claude-opus-4-8"].input == pytest.approx(0.1)

    # Explicit path wins over the env var; canonical key also works directly.
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {
                "anthropic:claude-opus-4-8": {
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
    assert from_path["anthropic:claude-opus-4-8"].input == pytest.approx(0.9)


def test_load_rates_rejects_duplicate_canonical_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two raw keys that collapse to the same canonical key must fail loudly.

    A plain dict comprehension would keep only whichever entry JSON iteration
    happened to visit last, silently mis-pricing the collided row.
    """
    custom = tmp_path / "rates.json"
    custom.write_text(
        json.dumps(
            {
                "anthropic:aws/claude-opus-5": {
                    "input": 0.1,
                    "output": 0.2,
                    "cache_read": 0.0,
                    "cache_create": 0.0,
                },
                "anthropic:claude-opus-5": {
                    "input": 0.9,
                    "output": 0.9,
                    "cache_read": 0.0,
                    "cache_create": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTERNAL_RATES_ENV_VAR, str(custom))

    with pytest.raises(ValueError) as excinfo:
        load_external_rates()

    msg = str(excinfo.value)
    assert "anthropic:aws/claude-opus-5" in msg
    assert "anthropic:claude-opus-5" in msg
    assert str(custom) in msg


def test_compute_cost_rejects_duplicate_canonical_keys() -> None:
    """A caller-supplied rates dict with a canonical-key collision is rejected."""
    rate_a = ModelRate(input=0.1, output=0.2, cache_read=0.0, cache_create=0.0)
    rate_b = ModelRate(input=0.9, output=0.9, cache_read=0.0, cache_create=0.0)

    with pytest.raises(ValueError) as excinfo:
        compute_cost(
            [],
            {
                "anthropic:aws/claude-opus-5": rate_a,
                "anthropic:claude-opus-5": rate_b,
            },
        )
    msg = str(excinfo.value)
    assert "anthropic:aws/claude-opus-5" in msg
    assert "anthropic:claude-opus-5" in msg


def test_compute_cost_source_label_and_table_divergence() -> None:
    records = [_opus_record()]
    contracted = compute_cost(records, load_rates())
    external = compute_cost(records, load_external_rates(), source="public-api-rate-table")

    assert contracted.source == "contracted-rate-table"
    assert external.source == "public-api-rate-table"
    # Same records, different tables -> different totals. For an Opus-only
    # record, public list price is strictly above the contracted rate.
    assert external.amount_usd > contracted.amount_usd


def test_run_manifest_carries_external_cost() -> None:
    records = [_opus_record()]
    contracted = compute_cost(records, load_rates())
    external = compute_cost(records, load_external_rates(), source="public-api-rate-table")

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


def test_compute_cost_reports_coverage_and_by_model_on_partial_run() -> None:
    """Partial-priced run: top-level priced_token_share + per-row shares
    together tell the reader what fraction of the run got dollarized and
    where each `(provider, model, role)` group sits relative to the total."""
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
            )
        },
    )

    # 20 priced tokens vs 100 unpriced -> 20/120 share.
    assert summary.priced_token_share == pytest.approx(20 / 120)
    assert summary.coverage.priced_models == ["openai:m1"]
    assert summary.coverage.unpriced_models == ["anthropic:missing"]
    assert summary.unpriced_models == ["anthropic:missing"]

    # One row per (provider, model, role) — same grouping as models_used.
    by_model = {(r.provider, r.model, r.role): r for r in summary.by_model}
    priced = by_model[("openai", "m1", "candidate_discovery")]
    unpriced = by_model[("anthropic", "missing", "deep_research")]

    assert priced.priced is True
    assert priced.rate_key == "openai:m1"
    assert priced.amount_usd == pytest.approx(0.213)
    # 20 tokens on this row / 120 total across all rows.
    assert priced.priced_token_share == pytest.approx(20 / 120)

    assert unpriced.priced is False
    assert unpriced.rate_key == "anthropic:missing"
    assert unpriced.amount_usd == 0.0
    assert unpriced.priced_token_share == pytest.approx(100 / 120)

    # Per-row shares must sum to 1.0 (they partition the run's tokens).
    assert sum(r.priced_token_share for r in summary.by_model) == pytest.approx(1.0)
    # by_model dollar sum invariant: matches amount_usd.
    assert sum(r.amount_usd for r in summary.by_model) == pytest.approx(summary.amount_usd)


def test_compute_cost_full_coverage_when_every_model_has_a_rate() -> None:
    records = [_opus_record()]
    summary = compute_cost(records, load_rates())

    assert summary.priced_token_share == 1.0
    assert summary.coverage.unpriced_models == []
    assert summary.unpriced_models == []
    assert all(row.priced for row in summary.by_model)


def test_compute_cost_empty_records_reports_zero_share() -> None:
    summary = compute_cost([], {})

    assert summary.amount_usd == 0.0
    assert summary.priced_token_share == 0.0
    assert summary.coverage.priced_models == []
    assert summary.coverage.unpriced_models == []
    assert summary.by_model == []


def test_run_manifest_exposes_coverage_and_by_model() -> None:
    """The manifest's cost block carries priced_token_share, coverage, and
    by_model through verbatim so downstream readers get the same structural
    signal `compute_cost` produced."""
    records = [
        UsageRecord.from_usage(
            AgentUsage(input=10, output=5, model="m1"),
            step="candidate_discovery",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=0,
            invocation_id="i0",
            cli="codex",
            role="candidate_discovery",
        ),
    ]
    summary = compute_cost(
        records,
        {"openai:m1": ModelRate(input=0.01, output=0.02, cache_read=0.0, cache_create=0.0)},
    )

    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-07-06T00:00:00Z",
        objective="find spots",
        provenance={},
        config_fingerprint={},
        records=records,
        cost=summary,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=0,
        module_status={},
        notes=[],
    )

    assert manifest.cost.priced_token_share == 1.0
    assert manifest.cost.coverage.priced_models == ["openai:m1"]
    assert manifest.cost.by_model[0].rate_key == "openai:m1"
    assert manifest.cost.by_model[0].priced is True
    assert manifest.cost.by_model[0].amount_usd == pytest.approx(summary.amount_usd)
    assert manifest.cost.by_model[0].priced_token_share == pytest.approx(1.0)


def test_old_manifest_dict_without_coverage_validates() -> None:
    """A persisted manifest predating the priced_token_share / coverage /
    by_model additions still validates — the new fields default to
    0.0 / empty coverage / [] by_model."""
    from spotlights_engine.costing.manifest import RunManifest

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
    payload = manifest.model_dump(mode="json")
    payload["cost"].pop("priced_token_share", None)
    payload["cost"].pop("coverage", None)
    payload["cost"].pop("by_model", None)

    reloaded = RunManifest.model_validate(payload)
    assert reloaded.cost.priced_token_share == 0.0
    assert reloaded.cost.coverage.priced_models == []
    assert reloaded.cost.by_model == []


def test_bundled_rates_price_gpt5_and_codex_at_identical_rates() -> None:
    """openai:gpt-5.5 (resolved) and openai:codex (CLI-family fallback) must
    price the same tokens to the same dollars. Same-cli, same-model reality;
    the split is a stream-emission quirk, not a billing difference."""
    rates = load_rates()
    ext_rates = load_external_rates()

    assert "openai:gpt-5.5" in rates
    assert "openai:codex" in rates
    for table in (rates, ext_rates):
        gpt = table["openai:gpt-5.5"]
        codex = table["openai:codex"]
        assert gpt.input == codex.input
        assert gpt.output == codex.output
        assert gpt.cache_read == codex.cache_read
        assert gpt.cache_create == codex.cache_create


def test_run_manifest_records_requested_models_separately_from_used() -> None:
    """`models_requested` is intent; `models_used` is what the CLIs reported.

    They differ on purpose: a blank request means the engine passed no `--model`
    and the CLI chose, so the manifest has to record both to be reproducible.
    """
    records = [
        UsageRecord.from_usage(
            AgentUsage(input=10, output=5, model="claude-opus-5[1m]"),
            step="candidate_discovery",
            module_qualified_name="pkg/a",
            session_index=1,
            invocation_index=0,
            invocation_id="i0",
            cli="claude",
            role="candidate_discovery",
        ),
    ]
    summary = compute_cost(records, {})

    manifest = build_run_manifest(
        run_id="run-1",
        date="2026-08-26T00:00:00Z",
        objective="find spots",
        provenance={
            "repo_url": "https://example.test/repo.git",
            "target_commit_sha": "abc",
            "spotlights_commit_sha": "def",
        },
        config_fingerprint={},
        records=records,
        cost=summary,
        wall_clock_s=1.0,
        candidates_path="/tmp/out/index.md",
        num_candidates=1,
        module_status={"SUCCEEDED": 1},
        notes=[],
        models_requested=RunManifestModelsRequested(claude="aws/claude-opus-4-8"),
    )

    assert manifest.models_requested.claude == "aws/claude-opus-4-8"
    # Codex was never asked for one — empty, not None, so the JSON stays flat.
    assert manifest.models_requested.codex == ""
    # What actually ran is unchanged and still recorded independently.
    assert [m.model for m in manifest.models_used] == ["claude-opus-5[1m]"]


def test_run_manifest_requested_models_default_to_blank() -> None:
    """Omitting the argument must not break older callers or the schema."""
    summary = compute_cost([], {})
    manifest = build_run_manifest(
        run_id="run-2",
        date="2026-08-26T00:00:00Z",
        objective="o",
        provenance={
            "repo_url": "u",
            "target_commit_sha": "a",
            "spotlights_commit_sha": "b",
        },
        config_fingerprint={},
        records=[],
        cost=summary,
        wall_clock_s=1.0,
        candidates_path="p",
        num_candidates=0,
        module_status={},
        notes=[],
    )
    assert manifest.models_requested.claude == ""
    assert manifest.models_requested.codex == ""


def test_models_requested_reports_step_2s_default_not_a_blank() -> None:
    """A blank config still results in the engine passing a Codex model.

    Step 2 carries a built-in `gpt-5.5` default, so recording "" would claim the
    CLI chose when it did not — exactly the provenance question this field exists
    to answer.
    """
    from pathlib import Path

    from spotlights_engine.spotlights_manager.api import SpotlightsManagerConfig
    from spotlights_engine.spotlights_manager.orchestrator import _models_requested

    cfg = SpotlightsManagerConfig(
        artifacts_dir=Path("/tmp/artifacts"), output_folder=Path("/tmp/out")
    )
    requested = _models_requested(cfg)
    assert requested.codex == "gpt-5.5"
    assert requested.claude == ""  # no built-in default; genuinely inherits


def test_models_requested_finds_a_model_pinned_on_any_single_step() -> None:
    """A library caller may configure one step and not the others.

    Reading only `models` (or only the first two step configs) recorded such a
    run as "the engine passed no model", the opposite of the truth. An explicitly
    set value also has to win over another step's *field default*, which is why
    the lookup asks pydantic which fields were actually supplied.
    """
    from pathlib import Path

    from spotlights_engine.agent_proposals import AgentProposalsConfig
    from spotlights_engine.spotlights_manager.api import SpotlightsManagerConfig
    from spotlights_engine.spotlights_manager.orchestrator import _models_requested

    cfg = SpotlightsManagerConfig(
        artifacts_dir=Path("/tmp/artifacts"),
        output_folder=Path("/tmp/out"),
        agent_proposals=AgentProposalsConfig(
            claude_model="pinned-claude", codex_model="pinned-codex"
        ),
    )
    requested = _models_requested(cfg)
    assert requested.claude == "pinned-claude"
    # Not step 2's `gpt-5.5` default, which would otherwise shadow this.
    assert requested.codex == "pinned-codex"
