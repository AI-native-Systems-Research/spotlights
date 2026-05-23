"""CLI defaults for the local spotlights manager runner."""

from __future__ import annotations

from scripts.run_spotlights_manager import _build_argparser


def test_step4_debug_pair_cap_is_disabled_by_default() -> None:
    args = _build_argparser().parse_args([])

    assert args.debug_first_n_pairs is None


def test_step4_debug_pair_cap_can_be_enabled_explicitly() -> None:
    args = _build_argparser().parse_args(["--debug-first-n-pairs", "5"])

    assert args.debug_first_n_pairs == 5
