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
    GeneratedPathError,
    ScopeError,
    SelectionError,
    UnsupportedEvolverError,
)

from . import _fixtures as fx

_CFG = PrepEvolveConfig(captured_at="2026-06-16T00:00:00+00:00")


def _input(tmp_path: Path, repo: Path, /, **kw) -> PrepEvolveInput:
    base = dict(
        module="v1/attention",
        candidate="cand-v1_attention-0002",
        repo=str(repo),
        evolver="skydiscover",
        out=tmp_path / "bundles",
    )
    base.update(kw)  # kw may override repo (e.g. repo=None for index fallback)
    # Only write the default single-candidate result.json when the test did not
    # supply its own --result (multi-candidate payloads, sorted dirs, etc.).
    if "result" not in base:
        base["result"] = fx.write_result(tmp_path)
    return PrepEvolveInput(**base)


def test_emits_readme_not_metadata_files(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    assert len(result.bundles) == 1
    bundle = Path(result.bundles[0].path)
    assert (bundle / "README.md").exists()
    for name in ("evolve_spec.json", "findings_digest.md", "generated_files.json"):
        assert not (bundle / name).exists()


def test_existing_bundle_skipped_without_force(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    first = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    assert first.bundles
    # Re-run without --force: the existing bundle is skipped, not overwritten,
    # and not an error.
    again = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    assert again.bundles == []
    assert any("exists" in s.reason for s in again.skipped)
    # With --force it is regenerated.
    forced = prep_evolve(_input(tmp_path, repo, evolver="coral", force=True), _CFG)
    assert forced.bundles


def test_force_overwrites_bundle_without_prior_manifest(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    out = tmp_path / "bundles"
    bundle = out / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"
    bundle.mkdir(parents=True)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", out=out, force=True), _CFG)
    assert result.bundles
    assert (bundle / "task.yaml").exists()


def test_generated_path_escape_rejected_before_writing(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    with pytest.raises(GeneratedPathError):
        _materialize(bundle, [GeneratedFile(path="../escape.txt", text="x")])
    assert not bundle.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_bundle_layout_ignores_repo_name(tmp_path: Path) -> None:
    # The new nested layout does not embed the repo name; a hostile repo name
    # never reaches the path.
    repo = fx.make_repo(tmp_path)
    payload = fx.make_result_dict()
    payload["report"]["project_tree"]["repository"]["name"] = "../demo repo"
    result_json = tmp_path / "custom_result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "bundles"
    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral", out=out), _CFG
    )
    bundle = Path(result.bundles[0].path)
    assert bundle == out / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"


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


def test_new_bundle_layout(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    out = tmp_path / "bundles"
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", out=out), _CFG)
    bundle = Path(result.bundles[0].path)
    assert bundle == out / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"


def test_out_defaults_to_run_dir(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral", out=None), _CFG
    )
    bundle = Path(result.bundles[0].path)
    # run dir is tmp_path (the folder holding result.json)
    assert bundle == tmp_path / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"


def test_result_directory_resolves_repo_from_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fx.write_result(run_dir)          # run/result.json
    repo = fx.make_repo(tmp_path)
    fx.write_index(run_dir, repo)      # run/index.md carrying Repo path
    result = prep_evolve(
        _input(tmp_path, repo, result=run_dir, repo=None, evolver="coral", out=None),
        _CFG,
    )
    bundle = Path(result.bundles[0].path)
    assert bundle == run_dir / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"


def test_module_inferred_when_omitted(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", module=None), _CFG)
    assert result.bundles[0].evolver == "coral"


def test_module_mismatch_errors(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    with pytest.raises(SelectionError):
        prep_evolve(_input(tmp_path, repo, evolver="coral", module="v1/nope"), _CFG)


def test_batch_all_candidates(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    payload = fx.make_result_dict()
    run = payload["module_runs"]["v1/attention"]
    second = dict(run["candidates"]["candidates"][0])
    second["id"] = "cand-v1_attention-0003"
    run["candidates"]["candidates"].append(second)
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")

    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral",
               candidate=None, out=tmp_path / "b"),
        _CFG,
    )
    ids = sorted(Path(b.path).parent.name for b in result.bundles)
    assert ids == ["cand-v1_attention-0002", "cand-v1_attention-0003"]


def test_batch_skips_invalid_candidate(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    payload = fx.make_result_dict()
    run = payload["module_runs"]["v1/attention"]
    bogus = dict(run["candidates"]["candidates"][0])
    bogus["id"] = "cand-v1_attention-0099"
    bogus["locations"] = [
        {"file": "pkg/attn/missing.py",
         "spans": [{"line_start": 1, "line_end": 2, "symbol": "x", "kind": "function"}]}
    ]
    run["candidates"]["candidates"].append(bogus)
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")

    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral",
               candidate=None, out=tmp_path / "b"),
        _CFG,
    )
    assert len(result.bundles) == 1
    assert Path(result.bundles[0].path).parent.name == "cand-v1_attention-0002"
    assert any(s.candidate_id == "cand-v1_attention-0099" for s in result.skipped)


def _two_candidate_run(tmp_path: Path) -> Path:
    """Write a result.json with two candidates; return its path."""
    payload = fx.make_result_dict()
    run = payload["module_runs"]["v1/attention"]
    second = dict(run["candidates"]["candidates"][0])
    second["id"] = "cand-v1_attention-0003"
    run["candidates"]["candidates"].append(second)
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")
    return result_json


def test_sorted_source_uses_ranking_order(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    _two_candidate_run(tmp_path)
    # Ranking lists 0003 first, then 0002.
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0003", "cand-v1_attention-0002"]
    )
    result = prep_evolve(
        _input(tmp_path, repo, result=sorted_dir, candidate=None,
               evolver="coral", out=tmp_path / "b"),
        _CFG,
    )
    order = [Path(b.path).parent.name for b in result.bundles]
    assert order == ["cand-v1_attention-0003", "cand-v1_attention-0002"]


def test_top_n_limits_ranked_selection(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    _two_candidate_run(tmp_path)
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0003", "cand-v1_attention-0002"]
    )
    result = prep_evolve(
        _input(tmp_path, repo, result=sorted_dir, candidate=None, top_n=1,
               evolver="coral", out=tmp_path / "b"),
        _CFG,
    )
    names = [Path(b.path).parent.name for b in result.bundles]
    assert names == ["cand-v1_attention-0003"]


def test_ranked_id_missing_in_result_is_skipped(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    fx.write_result(tmp_path)  # only cand-v1_attention-0002 exists
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0002", "cand-ghost-0009"]
    )
    result = prep_evolve(
        _input(tmp_path, repo, result=sorted_dir, candidate=None,
               evolver="coral", out=tmp_path / "b"),
        _CFG,
    )
    assert len(result.bundles) == 1
    assert any(s.candidate_id == "cand-ghost-0009" for s in result.skipped)


def test_top_n_without_ranking_errors(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    with pytest.raises(SelectionError):
        prep_evolve(
            _input(tmp_path, repo, result=result_json, candidate=None, top_n=5,
                   evolver="coral", out=tmp_path / "b"),
            _CFG,
        )


def test_cli_batch_no_candidate(tmp_path: Path, capsys) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    rc = prep_main(
        ["--result", str(result_json), "--repo", str(repo),
         "--evolver", "coral", "--out", str(tmp_path / "out")]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "bundle(s) written" in err


def test_cli_minimal_flags_directory(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fx.write_result(run_dir)
    repo = fx.make_repo(tmp_path)
    fx.write_index(run_dir, repo)
    # No --module, --candidate, --repo, or --out.
    rc = prep_main(["--result", str(run_dir), "--evolver", "coral"])
    assert rc == 0
    assert (run_dir / "evolve" / "v1_attention").is_dir()


def test_cli_top_n_from_sorted(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    _two_candidate_run(tmp_path)
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0003", "cand-v1_attention-0002"]
    )
    rc = prep_main(
        ["--result", str(sorted_dir), "--repo", str(repo), "--evolver", "coral",
         "--top-n", "1", "--out", str(tmp_path / "out")]
    )
    assert rc == 0
    # Only the top-ranked candidate's bundle exists.
    evolve = tmp_path / "out" / "evolve" / "v1_attention"
    assert (evolve / "cand-v1_attention-0003").is_dir()
    assert not (evolve / "cand-v1_attention-0002").exists()


def test_cli_bad_top_n_returns_2(tmp_path: Path, capsys) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    rc = prep_main(
        ["--result", str(result_json), "--evolver", "coral",
         "--top-n", "banana", "--out", str(tmp_path / "out")]
    )
    assert rc == 2
    assert "--top-n" in capsys.readouterr().err


def test_cli_clean_rerun_exits_0(tmp_path: Path, capsys) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    args = ["--result", str(result_json), "--repo", str(repo),
            "--evolver", "coral", "--out", str(tmp_path / "out")]
    assert prep_main(args) == 0            # first run writes the bundle
    capsys.readouterr()                    # drain
    rc = prep_main(args)                   # re-run: bundle exists -> skipped
    assert rc == 0                         # clean resume is success, not failure
    err = capsys.readouterr().err
    assert "0 bundle(s) written" in err
    assert "1 skipped" in err


def test_cli_stale_ranked_id_summary_omits_none(tmp_path: Path, capsys) -> None:
    repo = fx.make_repo(tmp_path)
    fx.write_result(tmp_path)  # only cand-v1_attention-0002 exists
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0002", "cand-ghost-0009"]
    )
    rc = prep_main(
        ["--result", str(sorted_dir), "--repo", str(repo), "--evolver", "coral",
         "--out", str(tmp_path / "out")]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "cand-ghost-0009" in err
    assert "None/" not in err
