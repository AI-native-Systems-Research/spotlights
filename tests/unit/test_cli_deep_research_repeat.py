"""`--deep-research-repeat N`: parsed as int, defaults to 1, guarded >= 1."""

from __future__ import annotations

import pytest

from spotlights_engine.cli import _build_argparser


def test_deep_research_repeat_defaults_to_one() -> None:
    args = _build_argparser().parse_args([])
    assert args.deep_research_repeat == 1


def test_deep_research_repeat_parses_int() -> None:
    args = _build_argparser().parse_args(["--deep-research-repeat", "5"])
    assert args.deep_research_repeat == 5


def test_deep_research_repeat_rejects_non_int() -> None:
    with pytest.raises(SystemExit):
        _build_argparser().parse_args(["--deep-research-repeat", "abc"])
