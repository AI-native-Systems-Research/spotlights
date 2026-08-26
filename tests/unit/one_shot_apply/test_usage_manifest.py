"""The per-candidate usage manifest apply writes beside apply.patch.

Rate tables are injected rather than loaded, so a rate change in the bundled
`rates.json` cannot turn a dollar assertion red. The loader path has its own
test (see the rate-table failure case).
"""

from __future__ import annotations

import json

import pytest

from spotlights_engine.costing.rates import ModelRate
from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.one_shot_apply.usage_manifest import (
    MANIFEST_NAME,
    ApplyUsageManifest,
    build_apply_usage_manifest,
)

# Round rates so every expected dollar figure is readable by hand.
RATES = {
    "anthropic:aws/claude-opus-5": ModelRate(
        input=1e-6, output=2e-6, cache_read=1e-7, cache_create=5e-7
    )
}
# Exactly double, so external > contracted the way list price exceeds a contract.
EXTERNAL_RATES = {
    "anthropic:aws/claude-opus-5": ModelRate(
        input=2e-6, output=4e-6, cache_read=2e-7, cache_create=1e-6
    )
}

USAGE = AgentUsage(
    input=1000,
    output=2000,
    cache_read=30000,
    cache_create=4000,
    model="aws/claude-opus-5",
    api_time_s=42.5,
    cli_reported_cost_usd=0.99,
)


def _build(**overrides):
    kwargs = dict(
        candidate_id="cand-pkg_a-0001",
        module_qualified_name="pkg/a",
        date="2026-08-26T12:00:00+00:00",
        objective="reduce end-to-end page latency",
        target_commit_sha="641b9806",
        repo_url="git@github.com:example/repo.git",
        spotlights_commit_sha="e1c787c9",
        usage=USAGE,
        duration_s=99.0,
        agent_error=None,
        rates=RATES,
        external_rates=EXTERNAL_RATES,
    )
    kwargs.update(overrides)
    return build_apply_usage_manifest(**kwargs)


def test_the_filename_is_unqualified_because_the_directory_qualifies_it() -> None:
    assert MANIFEST_NAME == "manifest.json"


def test_a_complete_capture_prices_one_record_against_both_tables() -> None:
    """One `claude -p` call per candidate means one record and one models_used row.

    The four token buckets are disjoint by construction, so the cost is
    `sum(bucket * rate)` with no double-charging:
      1000*1e-6 + 2000*2e-6 + 30000*1e-7 + 4000*5e-7 = 0.010
    """
    m = _build()

    assert m.candidate_id == "cand-pkg_a-0001"
    assert m.module_qualified_name == "pkg/a"
    assert m.date == "2026-08-26T12:00:00+00:00"
    assert m.target.commit_sha == "641b9806"
    assert m.target.repo_url == "git@github.com:example/repo.git"
    assert m.target.objective == "reduce end-to-end page latency"
    assert m.spotlights.commit_sha == "e1c787c9"

    assert len(m.models_used) == 1
    row = m.models_used[0]
    assert row.model == "aws/claude-opus-5"
    assert row.provider == "anthropic"
    assert row.role == "one_shot_apply"
    assert row.usage.input == 1000
    assert row.usage.output == 2000
    assert row.usage.cache_read == 30000
    assert row.usage.cache_create == 4000

    assert m.total_tokens == 37000
    assert m.cost.amount_usd == pytest.approx(0.010)
    assert m.cost.source == "contracted-rate-table"
    assert m.cost.priced_token_share == 1.0
    assert m.external_cost is not None
    assert m.external_cost.amount_usd == pytest.approx(0.020)
    assert m.external_cost.source == "public-api-rate-table"

    assert m.timing.wall_clock_s == 99.0
    assert m.timing.api_time_s == 42.5
    assert m.notes == ""


def test_no_usage_keeps_every_field_and_puts_the_reason_in_notes() -> None:
    """A `--wallclock` kill severs the CLI's exit handshake after the work is done.

    `claude_usage_from_stream` returns None with no terminal `result` event, so
    there is no record to price. Handling matches `run_manifest.json`: stable
    field names, zeroed cost, reason in `notes` — no new `capture` enum and no
    per-turn salvage parser, whose numbers would not reconcile with the CLI's.
    """
    m = _build(usage=None, duration_s=1800.0, agent_error="claude timed out after 1800.0s")

    assert m.models_used == []
    assert m.total_tokens == 0
    assert m.cost.amount_usd == 0.0
    assert m.cost.priced_token_share == 0.0
    assert m.cost.by_model == []
    assert m.cost.coverage.priced_models == []
    assert m.cost.coverage.unpriced_models == []
    assert m.cost.source == "contracted-rate-table"
    assert m.external_cost is not None
    assert m.external_cost.source == "public-api-rate-table"

    # Still real: a degraded manifest must still say 1800 seconds were burned.
    assert m.timing.wall_clock_s == 1800.0
    assert m.timing.api_time_s == 0.0

    assert m.notes == "no usage records found: claude timed out after 1800.0s"


def test_no_usage_from_a_clean_session_says_so_differently() -> None:
    """A clean session with no totals is a different fault from a timeout, and a
    reader deciding whether to re-run needs to tell them apart."""
    m = _build(usage=None, agent_error=None)
    assert m.notes == (
        "no usage records found; the agent session reported no usage totals"
    )


def test_the_manifest_round_trips_through_json() -> None:
    """Guards the `extra="forbid"` models against a field added on one side only."""
    m = _build()
    dumped = json.dumps(m.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    assert ApplyUsageManifest.model_validate(json.loads(dumped)) == m


def test_an_unpriced_model_is_named_in_both_notes_and_in_coverage() -> None:
    """`amount_usd` stays a number when a model has no rate row, so the note and
    `coverage.unpriced_models` are the only things telling a reader the figure
    is partial. Both tables get their own line, because a model can be priced
    contractually and not publicly."""
    m = _build(usage=USAGE.model_copy(update={"model": "aws/claude-opus-9"}))

    assert m.cost.amount_usd == 0.0
    assert m.cost.priced_token_share == 0.0
    assert m.cost.coverage.unpriced_models == ["anthropic:aws/claude-opus-9"]
    assert m.notes == (
        "unpriced models excluded from cost: anthropic:aws/claude-opus-9; "
        "external: unpriced models excluded from cost: anthropic:aws/claude-opus-9"
    )
    # The tokens are still counted — only the dollars are missing.
    assert m.total_tokens == 37000
    assert len(m.models_used) == 1


def test_an_unresolved_model_falls_back_to_the_cli_family_rather_than_inventing_one() -> None:
    """No `fallback_model` is passed: a made-up name would silently price at
    another model's rate. Unresolved becomes the unpriced `anthropic:claude`,
    which is visible."""
    m = _build(usage=USAGE.model_copy(update={"model": None}))

    assert m.models_used[0].model == "claude"
    assert m.cost.coverage.unpriced_models == ["anthropic:claude"]
    assert "unpriced models excluded from cost: anthropic:claude" in m.notes


def test_unloadable_rate_tables_zero_the_cost_instead_of_raising(
    tmp_path, monkeypatch
) -> None:
    """A pricing problem must not turn a successful 25-minute apply into a skip.

    This is the one test that exercises the loader path (the others inject
    tables), so it is what pins the `(OSError, ValueError)` guard.
    """
    from spotlights_engine.costing.rates import RATES_ENV_VAR

    bad = tmp_path / "broken-rates.json"
    bad.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setenv(RATES_ENV_VAR, str(bad))

    m = build_apply_usage_manifest(
        candidate_id="cand-pkg_a-0001",
        module_qualified_name="pkg/a",
        date="2026-08-26T12:00:00+00:00",
        objective="reduce latency",
        target_commit_sha="641b9806",
        repo_url="git@github.com:example/repo.git",
        spotlights_commit_sha="e1c787c9",
        usage=USAGE,
        duration_s=99.0,
        agent_error=None,
    )

    assert m.cost.amount_usd == 0.0
    assert m.external_cost is not None
    assert m.external_cost.amount_usd == 0.0
    assert "cost unavailable: could not load the rate tables:" in m.notes
    # The rest of the manifest is intact — this is a pricing failure, not a
    # manifest failure.
    assert m.total_tokens == 37000
    assert m.timing.wall_clock_s == 99.0
    assert m.target.commit_sha == "641b9806"


def test_missing_provenance_is_noted_but_never_fatal() -> None:
    """`resolve_repo_url` and `spotlights_commit_sha` degrade to "" rather than
    raising — a wheel install has no engine checkout, and a repo may have no
    origin remote. Both are worth saying out loud."""
    m = _build(repo_url="", spotlights_commit_sha="")

    assert m.target.repo_url == ""
    assert m.spotlights.commit_sha == ""
    assert m.notes == "target repo URL unavailable; Spotlights commit unavailable"


def test_notes_join_every_applicable_reason_like_the_run_manifest_does() -> None:
    """`"; ".join` over the parts that apply, matching
    `build_run_manifest`'s assembly, so a reader who knows one file reads the
    other."""
    m = _build(
        usage=None,
        agent_error="claude timed out after 1800.0s",
        repo_url="",
        spotlights_commit_sha="",
    )
    assert m.notes == (
        "no usage records found: claude timed out after 1800.0s; "
        "target repo URL unavailable; Spotlights commit unavailable"
    )
