"""CLI defaults for the local spotlights manager runner."""

from __future__ import annotations

from pathlib import Path

from scripts.run_spotlights_manager import _build_argparser


def test_no_personal_paths_in_scripts():
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    offenders = []
    for py in scripts_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for needle in ("/Users/ophir", "/Users/iklamer", "PycharmProjects", "GoProjects"):
            if needle in text:
                offenders.append(f"{py}: {needle}")
    assert not offenders, f"personal paths leaked into shipped scripts: {offenders}"


def test_step4_debug_pair_cap_is_disabled_by_default() -> None:
    args = _build_argparser().parse_args([])

    assert args.debug_first_n_pairs is None


def test_step4_debug_pair_cap_can_be_enabled_explicitly() -> None:
    args = _build_argparser().parse_args(["--debug-first-n-pairs", "5"])

    assert args.debug_first_n_pairs == 5
