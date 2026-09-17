"""Adding up the usage of two invocations that produced one result.

Exists for the step-5 retry: a rate-limited attempt has usually already paid for
tokens, so the winning attempt's usage alone under-bills the run. Cost is the
figure this engine quotes to people, so the arithmetic gets pinned here rather
than trusted.
"""

from __future__ import annotations

from spotlights_engine.costing.usage import AgentUsage, merge_agent_usage


def test_counters_add() -> None:
    merged = merge_agent_usage(
        AgentUsage(input=100, output=10, cache_read=5, cache_create=1),
        AgentUsage(input=200, output=20, cache_read=7, cache_create=2),
    )

    assert merged is not None
    assert (merged.input, merged.output) == (300, 30)
    assert (merged.cache_read, merged.cache_create) == (12, 3)


def test_a_missing_side_is_passed_through_unchanged() -> None:
    """The common shape: the first attempt died before any usage was parsable."""
    only = AgentUsage(input=100)

    assert merge_agent_usage(None, only) is only
    assert merge_agent_usage(only, None) is only
    assert merge_agent_usage(None, None) is None


def test_optional_figures_add_but_stay_none_when_nobody_reported_them() -> None:
    """`None` means "the CLI never said", not zero.

    The rate table treats an absent `cli_reported_cost_usd` differently from a
    reported $0, so merging must not invent one.
    """
    both = merge_agent_usage(
        AgentUsage(api_time_s=2.0, cli_reported_cost_usd=0.01),
        AgentUsage(api_time_s=3.5, cli_reported_cost_usd=0.02),
    )
    assert both is not None
    assert both.api_time_s == 5.5
    assert both.cli_reported_cost_usd == 0.03

    one_side = merge_agent_usage(AgentUsage(api_time_s=2.0), AgentUsage())
    assert one_side is not None
    assert one_side.api_time_s == 2.0
    assert one_side.cli_reported_cost_usd is None

    neither = merge_agent_usage(AgentUsage(input=1), AgentUsage(input=2))
    assert neither is not None
    assert neither.api_time_s is None
    assert neither.cli_reported_cost_usd is None


def test_the_model_survives_from_whichever_attempt_named_it() -> None:
    """Retrying does not change the model, but a truncated stream may not name it.

    A merged record with `model=None` would land in the manifest as an unpriced
    model and silently drop out of the cost total.
    """
    from_earlier = merge_agent_usage(
        AgentUsage(model="aws/claude-opus-4-8"), AgentUsage(model=None)
    )
    from_later = merge_agent_usage(
        AgentUsage(model=None), AgentUsage(model="aws/claude-opus-4-8")
    )

    assert from_earlier is not None and from_earlier.model == "aws/claude-opus-4-8"
    assert from_later is not None and from_later.model == "aws/claude-opus-4-8"


def test_merging_leaves_the_inputs_alone() -> None:
    """The loop merges in place across attempts; aliasing would double-count."""
    earlier = AgentUsage(input=100)
    later = AgentUsage(input=200)

    merged = merge_agent_usage(earlier, later)

    assert merged is not None and merged.input == 300
    assert (earlier.input, later.input) == (100, 200)
