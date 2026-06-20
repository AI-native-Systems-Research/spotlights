"""Materialize / CLI tests (plan §10.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.prep_evolve.adapters.base import GeneratedFile
from spotlights_engine.prep_evolve.api import (
    PrepEvolveConfig,
    PrepEvolveInput,
    _materialize,
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
        candidate="cand-v1_attention-0002",
        repo=str(repo),
        evolver="skydiscover",
        out=tmp_path / "bundles",
    )
    base.update(kw)  # kw may override repo (e.g. repo=None for index fallback)
    return PrepEvolveInput(**base)


def test_emits_readme_not_metadata_files(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    assert len(result.bundles) == 1
    bundle = Path(result.bundles[0].path)
    assert (bundle / "README.md").exists()
    for name in ("evolve_spec.json", "findings_digest.md", "generated_files.json"):
        assert not (bundle / name).exists()


def test_force_required_for_existing_bundle(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    with pytest.raises(BundleExistsError):
        prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    # With --force it succeeds.
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", force=True), _CFG)
    assert result.bundles


def test_force_overwrites_bundle_without_prior_manifest(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    out = tmp_path / "bundles"
    # Pre-create a bundle dir with no manifest; --force overwrites it.
    bundle = out / "demo__v1_attention__cand-v1_attention-0002__coral"
    bundle.mkdir(parents=True)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", out=out, force=True), _CFG)
    assert result.bundles
    assert (bundle / "task.yaml").exists()


def test_generated_path_escape_rejected_before_writing(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    with pytest.raises(BundleExistsError):
        _materialize(
            bundle,
            [GeneratedFile(path="../escape.txt", text="x")],
            force=False,
        )
    assert not bundle.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_bundle_dir_name_sanitizes_repo_name(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    payload = fx.make_result_dict()
    payload["project_tree"]["repository"]["name"] = "../demo repo"
    result_json = tmp_path / "custom_result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "bundles"
    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral", out=out),
        _CFG,
    )
    bundle = Path(result.bundles[0].path)
    assert bundle.parent == out
    assert bundle.name.startswith("demo_repo__")


def test_force_overwrites_hand_edited_config(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="skydiscover"), _CFG)
    bundle = Path(result.bundles[0].path)
    config = bundle / "config.yaml"
    config.write_text("# HAND EDITED\n", encoding="utf-8")

    prep_evolve(_input(tmp_path, repo, evolver="skydiscover", force=True), _CFG)
    # Bundles are fully generator-owned: --force overwrites everything.
    assert config.read_text() != "# HAND EDITED\n"
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
    result = prep_evolve(_input(tmp_path, repo, evolver="all", scope="module-main-files"), _CFG)
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
    result = prep_evolve(_input(tmp_path, repo, evolver="agentic-strategy-evolution"), _CFG)
    assert result.bundles[0].evolver == "nous"


def test_repo_from_index_fallback(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    index = fx.write_index(tmp_path, repo)
    inp = _input(tmp_path, repo, index=index, evolver="coral")
    inp.repo = None
    result = prep_evolve(inp, _CFG)
    assert result.bundles


def test_direction_override(tmp_path: Path) -> None:
    import yaml

    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", direction="maximize"), _CFG)
    bundle = Path(result.bundles[0].path)
    task = yaml.safe_load((bundle / "task.yaml").read_text())
    assert task["grader"]["direction"] == "maximize"
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
            "cand-v1_attention-0002",
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
            "cand-v1_attention-0002",
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
            "cand-v1_attention-0002",
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
