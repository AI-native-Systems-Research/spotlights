"""Materialize / CLI tests (plan §10.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.prep_evolve.api import (
    PrepEvolveConfig,
    PrepEvolveInput,
    prep_evolve,
)
from spotlights_engine.prep_evolve.cli import main as prep_main
from spotlights_engine.prep_evolve.errors import (
    BundleExistsError,
    ScopeError,
    UnsupportedEvolverError,
)

from . import _fixtures as fx

_CFG = PrepEvolveConfig(captured_at="2026-06-16T00:00:00+00:00")


def _input(tmp_path: Path, repo: Path, **kw) -> PrepEvolveInput:
    base = dict(
        result=fx.write_result(tmp_path),
        module="v1/attention",
        candidate="cand-0002",
        repo=str(repo),
        evolver="skydiscover",
        out=tmp_path / "bundles",
    )
    base.update(kw)  # kw may override repo (e.g. repo=None for index fallback)
    return PrepEvolveInput(**base)


def test_emits_always_files(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    assert len(result.bundles) == 1
    bundle = Path(result.bundles[0].path)
    for name in ("evolve_spec.json", "findings_digest.md", "generated_files.json", "README.md"):
        assert (bundle / name).exists()


def test_manifest_excludes_itself(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    bundle = Path(result.bundles[0].path)
    manifest = json.loads((bundle / "generated_files.json").read_text())
    assert "generated_files.json" not in manifest["files"]
    assert "evolve_spec.json" in manifest["files"]


def test_force_required_for_existing_bundle(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    with pytest.raises(BundleExistsError):
        prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    # With --force it succeeds.
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", force=True), _CFG)
    assert result.bundles


def test_force_refused_without_prior_manifest(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    out = tmp_path / "bundles"
    # Pre-create a bundle dir with no manifest.
    bundle = out / "demo__v1_attention__cand-0002__coral"
    bundle.mkdir(parents=True)
    with pytest.raises(BundleExistsError):
        prep_evolve(_input(tmp_path, repo, evolver="coral", out=out, force=True), _CFG)


def test_hand_edited_evaluator_survives_force(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="skydiscover"), _CFG)
    bundle = Path(result.bundles[0].path)
    evaluator = bundle / "evaluator.py"
    evaluator.write_text("# HAND EDITED\n", encoding="utf-8")

    result2 = prep_evolve(_input(tmp_path, repo, evolver="skydiscover", force=True), _CFG)
    assert evaluator.read_text() == "# HAND EDITED\n"
    assert any("preserved user-modified" in w for w in result2.warnings)
    # config.yaml (overwrite=always) was rewritten, not preserved.
    assert (bundle / "config.yaml").exists()


def test_always_file_rewritten_on_force(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    bundle = Path(result.bundles[0].path)
    readme = bundle / "README.md"
    readme.write_text("garbage", encoding="utf-8")
    prep_evolve(_input(tmp_path, repo, evolver="coral", force=True), _CFG)
    assert readme.read_text() != "garbage"


def test_all_skips_skydiscover_for_multi_file(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(
        _input(tmp_path, repo, evolver="all", scope="module-main-files"), _CFG
    )
    evolvers = {b.evolver for b in result.bundles}
    assert evolvers == {"coral", "nous"}
    assert any(s.evolver == "skydiscover" for s in result.skipped)


def test_skydiscover_scope_error_single_request(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    with pytest.raises(ScopeError):
        prep_evolve(
            _input(tmp_path, repo, evolver="skydiscover", scope="module-main-files"),
            _CFG,
        )


def test_unknown_evolver(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    with pytest.raises(UnsupportedEvolverError):
        prep_evolve(_input(tmp_path, repo, evolver="nope"), _CFG)


def test_alias_resolves(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(
        _input(tmp_path, repo, evolver="agentic-strategy-evolution"), _CFG
    )
    assert result.bundles[0].evolver == "nous"


def test_repo_from_index_fallback(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    index = fx.write_index(tmp_path, repo)
    inp = _input(tmp_path, repo, index=index, evolver="coral")
    inp.repo = None
    result = prep_evolve(inp, _CFG)
    assert result.bundles


def test_direction_override(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(
        _input(tmp_path, repo, evolver="coral", direction="maximize"), _CFG
    )
    bundle = Path(result.bundles[0].path)
    spec = json.loads((bundle / "evolve_spec.json").read_text())
    assert spec["objective"]["direction"] == "maximize"
    # No inference warning when explicitly given.
    assert not any("inferred" in w for w in result.warnings)


def test_cli_main_success(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    rc = prep_main(
        [
            "--result",
            str(result_json),
            "--module",
            "v1/attention",
            "--candidate",
            "cand-0002",
            "--repo",
            str(repo),
            "--evolver",
            "coral",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0


def test_top_level_cli_dispatch(tmp_path: Path) -> None:
    from spotlights_engine import cli

    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    rc = cli.main(
        [
            "prep-evolve",
            "--result",
            str(result_json),
            "--module",
            "v1/attention",
            "--candidate",
            "cand-0002",
            "--repo",
            str(repo),
            "--evolver",
            "coral",
            "--out",
            str(tmp_path / "dispatch-out"),
        ]
    )
    assert rc == 0


def test_cli_main_clean_error(tmp_path: Path, capsys) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    rc = prep_main(
        [
            "--result",
            str(result_json),
            "--module",
            "v1/nope",
            "--candidate",
            "cand-0002",
            "--repo",
            str(repo),
            "--evolver",
            "coral",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "prep-evolve:" in capsys.readouterr().err
