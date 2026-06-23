"""Adapter tests (plan §10.4)."""

from __future__ import annotations

from pathlib import Path

import yaml

from spotlights_engine.prep_evolve.adapters.coral import CoralAdapter
from spotlights_engine.prep_evolve.adapters.nous import NousAdapter
from spotlights_engine.prep_evolve.adapters.skydiscover import SkydiscoverAdapter
from spotlights_engine.prep_evolve.extract import build_spec
from spotlights_engine.prep_evolve.resolve import (
    load_result,
    resolve_candidate,
    resolve_candidates,
    resolve_findings,
    resolve_module,
    resolve_module_run,
)
from spotlights_engine.prep_evolve.spec import SourceRevision
from spotlights_engine.prep_evolve.validate_target import (
    ValidatedCandidate,
    validate_candidate_target,
)

from . import _fixtures as fx


def _spec(tmp_path: Path, scope="candidate", repo: Path | None = None):
    loaded = load_result(fx.write_result(tmp_path))
    run = resolve_module_run(loaded, "v1/attention")
    candidates = resolve_candidates(run, "v1/attention")
    candidate = resolve_candidate(candidates, "cand-v1_attention-0002")
    module = resolve_module(loaded.project_tree, "v1/attention")
    if repo is not None:
        validated = validate_candidate_target(repo, candidate)
        repo_path = str(repo)
    else:
        validated = ValidatedCandidate(fx.CAND_START, fx.CAND_END, "x")
        repo_path = "/tmp/repo"
    return build_spec(
        loaded=loaded,
        module=module,
        qn="v1/attention",
        candidate=candidate,
        findings=resolve_findings(run, "v1/attention"),
        repo_path=repo_path,
        validated=validated,
        revision=SourceRevision(git_commit=None, dirty=None, captured_at="t"),
        scope=scope,
        direction="minimize",
    )


def _by_path(files):
    return {f.path: f for f in files}


# --- skydiscover ---


def test_skydiscover_evolve_block_boundaries(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    spec = _spec(tmp_path, repo=repo)
    files = _by_path(SkydiscoverAdapter().render(spec))
    seed_lines = files["seed.py"].text.splitlines()
    start_idx = seed_lines.index("# EVOLVE-BLOCK-START")
    end_idx = seed_lines.index("# EVOLVE-BLOCK-END")
    # The line immediately after the start marker is the recorded block start
    # (1-indexed line CAND_START in the source).
    assert seed_lines[start_idx + 1] == f"def {fx.CAND_SYMBOL}(x):  # block start"
    # Block is CAND_END - CAND_START + 1 lines.
    assert end_idx - start_idx - 1 == fx.CAND_END - fx.CAND_START + 1


def test_skydiscover_config_keys(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    spec = _spec(tmp_path, repo=repo)
    files = _by_path(SkydiscoverAdapter().render(spec))
    cfg = yaml.safe_load(files["config.yaml"].text)
    assert cfg["language"] == "python"
    assert cfg["search"]["type"] == "adaevolve"
    assert cfg["llm"]["models"][0]["name"]
    # system_message is multi-line and long -> stays inline, not read as path.
    sm = cfg["prompt"]["system_message"]
    assert "\n" in sm and len(sm) > 256


def test_skydiscover_emits_only_seed_and_config(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    spec = _spec(tmp_path, repo=repo)
    files = _by_path(SkydiscoverAdapter().render(spec))
    assert set(files) == {"seed.py", "config.yaml"}


def test_skydiscover_rejects_multi_file_scope(tmp_path: Path) -> None:
    spec = _spec(tmp_path, scope="module-main-files")
    ok, reason = SkydiscoverAdapter().supports(spec)
    assert not ok
    assert reason


# --- CORAL ---


def test_coral_task_yaml_mapping(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    files = _by_path(CoralAdapter().render(spec))
    task = yaml.safe_load(files["task.yaml"].text)
    assert task["grader"]["direction"] == "minimize"
    # CORAL auto-discovers eval/grader.py — no entrypoint / package install.
    assert "entrypoint" not in task["grader"]
    assert "setup" not in task["grader"]
    assert task["grader"]["timeout"] == 600
    assert task["workspace"]["repo_path"] == "/tmp/repo"
    assert task["workspace"]["results_dir"] == "./results"
    assert task["workspace"]["setup"] == []
    assert fx.CAND_FILE in task["grader"]["args"]["target_files"]


def test_coral_emits_only_task_yaml(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    files = _by_path(CoralAdapter().render(spec))
    assert set(files) == {"task.yaml"}


def test_coral_multi_file_scope_supported(tmp_path: Path) -> None:
    spec = _spec(tmp_path, scope="module-main-files")
    ok, _ = CoralAdapter().supports(spec)
    assert ok
    files = _by_path(CoralAdapter().render(spec))
    task = yaml.safe_load(files["task.yaml"].text)
    assert fx.MAIN_FILE_EXTRA in task["grader"]["args"]["target_files"]


# --- Nous ---


def test_nous_theory_references_only_allowed_keys(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    files = _by_path(NousAdapter().render(spec))
    camp = yaml.safe_load(files["campaign.yaml"].text)
    allowed = {"name", "statement", "how", "use_as", "independent_of_detector"}
    for tr in camp["theory_references"]:
        assert set(tr) <= allowed
        assert "url" not in tr and "evidence" not in tr


def test_nous_campaign_required_fields(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    files = _by_path(NousAdapter().render(spec))
    camp = yaml.safe_load(files["campaign.yaml"].text)
    assert camp["research_question"]
    assert camp["target_system"]["name"] == "demo"
    assert camp["prompts"]["methodology_layer"] == "prompts/methodology"


def test_nous_bundle_is_only_campaign(tmp_path: Path) -> None:
    # The nous bundle is a single self-contained campaign.yaml — no bundle.yaml
    # and no vendored methodology prompts.
    spec = _spec(tmp_path)
    files = NousAdapter().render(spec)
    assert [f.path for f in files] == ["campaign.yaml"]


def test_nous_omits_empty_observable_metrics(tmp_path: Path) -> None:
    # A spec whose performance oracle parses to nothing should omit the key
    # entirely (minItems: 1) rather than emit [].
    spec = _spec(tmp_path)
    for t in spec.targets:
        t.oracles.performance = None
    files = _by_path(NousAdapter().render(spec))
    camp = yaml.safe_load(files["campaign.yaml"].text)
    assert "observable_metrics" not in camp["target_system"]


def test_nous_preserves_tokens_per_second_metric(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    for t in spec.targets:
        t.oracles.performance = "tokens/s"
    files = _by_path(NousAdapter().render(spec))
    camp = yaml.safe_load(files["campaign.yaml"].text)
    assert camp["target_system"]["observable_metrics"] == ["tokens/s"]


