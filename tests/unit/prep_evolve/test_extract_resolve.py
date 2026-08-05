"""Extract / resolve tests (plan §10.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.prep_evolve.errors import RepoResolutionError, SelectionError
from spotlights_engine.prep_evolve.extract import (
    build_spec,
    infer_direction,
    parse_correctness_oracles,
    parse_performance_oracle,
)
from spotlights_engine.prep_evolve.resolve import (
    load_result,
    parse_repo_path_from_index,
    resolve_candidate,
    resolve_candidates,
    resolve_findings,
    resolve_module,
    resolve_module_run,
    resolve_repo_path,
)
from spotlights_engine.prep_evolve.spec import SourceRevision
from spotlights_engine.prep_evolve.validate_target import ValidatedCandidate

from . import _fixtures as fx


def _load(tmp_path: Path):
    return load_result(fx.write_result(tmp_path))


def test_resolve_nested_submodule(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    module = resolve_module(loaded.project_tree, "v1/attention")
    assert module.name == "attention"
    assert module.path == "v1/attention"


def test_missing_module_lists_available(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    with pytest.raises(SelectionError) as exc:
        resolve_module(loaded.project_tree, "v1/nope")
    assert "v1/attention" in str(exc.value)


def test_missing_candidate_lists_available(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    run = resolve_module_run(loaded, "v1/attention")
    candidates = resolve_candidates(run, "v1/attention")
    with pytest.raises(SelectionError) as exc:
        resolve_candidate(candidates, "cand-9999")
    assert "cand-v1_attention-0002" in str(exc.value)


def test_findings_only_candidate_linked(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    run = resolve_module_run(loaded, "v1/attention")
    candidates = resolve_candidates(run, "v1/attention")
    candidate = resolve_candidate(candidates, "cand-v1_attention-0002")
    findings = resolve_findings(run, "v1/attention")
    module = resolve_module(loaded.project_tree, "v1/attention")

    spec = build_spec(
        loaded=loaded,
        module=module,
        qn="v1/attention",
        candidate=candidate,
        findings=findings,
        repo_path="/tmp/repo",
        validated=ValidatedCandidate(
            line_start=fx.CAND_START,
            line_end=fx.CAND_END,
            source_excerpt_sha256="deadbeef",
        ),
        revision=SourceRevision(git_commit=None, dirty=None, captured_at="t"),
        scope="candidate",
        direction="minimize",
    )

    # Only find-v1_attention-0001 is candidate-linked; the unlinked module finding
    # find-v1_attention-0002 is dropped rather than appended.
    assert [f.finding_id for f in spec.findings] == ["find-v1_attention-0001"]


def test_proposal_provenance(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    run = resolve_module_run(loaded, "v1/attention")
    candidates = resolve_candidates(run, "v1/attention")
    candidate = resolve_candidate(candidates, "cand-v1_attention-0002")
    module = resolve_module(loaded.project_tree, "v1/attention")
    spec = build_spec(
        loaded=loaded,
        module=module,
        qn="v1/attention",
        candidate=candidate,
        findings=resolve_findings(run, "v1/attention"),
        repo_path="/tmp/repo",
        validated=ValidatedCandidate(fx.CAND_START, fx.CAND_END, "x"),
        revision=SourceRevision(git_commit=None, dirty=None, captured_at="t"),
        scope="candidate",
        direction="minimize",
    )
    deep = [p for p in spec.proposals if p.origin == "deep_research"]
    agent = [p for p in spec.proposals if p.origin == "agent"]
    assert deep[0].finding_id == "find-v1_attention-0001"
    assert deep[0].agent == "claude"
    assert agent[0].finding_id is None
    assert agent[0].agent == "codex"
    # Decision D5: the structured research fields flow onto ProposalRef for the
    # deep-research proposal; the agent-knowledge proposal (which never set them)
    # stays None.
    assert deep[0].mechanism == "Derive tile size from the register budget."
    assert deep[0].required_changes == "Rewrite the tile-size heuristic function."
    assert deep[0].expected_effect == "Higher occupancy, lower TPOT."
    assert deep[0].evaluation_metric == "Median TPOT on the existing benchmark."
    assert agent[0].mechanism is None
    assert agent[0].required_changes is None
    assert agent[0].expected_effect is None
    assert agent[0].evaluation_metric is None


def test_scope_main_files_adds_whole_file_targets(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    run = resolve_module_run(loaded, "v1/attention")
    candidates = resolve_candidates(run, "v1/attention")
    candidate = resolve_candidate(candidates, "cand-v1_attention-0002")
    module = resolve_module(loaded.project_tree, "v1/attention")
    spec = build_spec(
        loaded=loaded,
        module=module,
        qn="v1/attention",
        candidate=candidate,
        findings=resolve_findings(run, "v1/attention"),
        repo_path="/tmp/repo",
        validated=ValidatedCandidate(fx.CAND_START, fx.CAND_END, "x"),
        revision=SourceRevision(git_commit=None, dirty=None, captured_at="t"),
        scope="module-main-files",
        direction="minimize",
    )
    kinds = [(t.scope_kind, t.file) for t in spec.targets]
    # candidate file first, then the extra main file as whole-file scope.
    assert kinds[0] == ("candidate", fx.CAND_FILE)
    assert ("module_main_file", fx.MAIN_FILE_EXTRA) in kinds
    # candidate file is not duplicated as a scope-only target.
    assert kinds.count(("module_main_file", fx.CAND_FILE)) == 0
    scope_only = [t for t in spec.targets if t.scope_kind == "module_main_file"]
    assert all(t.line_start is None and t.candidate_id is None for t in scope_only)


@pytest.mark.parametrize(
    "objective,expected",
    [
        ("reduce median latency", "minimize"),
        ("minimize TPOT", "minimize"),
        ("increase throughput", "maximize"),
        ("improve throughput on decode", "maximize"),
        ("make it good", "minimize"),  # ambiguous -> default minimize
        ("reduce latency but increase throughput", "minimize"),  # both -> default
    ],
)
def test_direction_inference(objective: str, expected: str) -> None:
    assert infer_direction(objective) == expected


def test_oracle_parsing() -> None:
    rationale = "Correctness oracle: tests/kernels/test_tile.py. Performance oracle is median TPOT."
    assert "pytest tests/kernels/test_tile.py" in parse_correctness_oracles(rationale)
    perf = parse_performance_oracle(rationale, "objective", "expl")
    assert perf is not None and "TPOT" in perf


def test_oracle_parsing_preserves_explicit_pytest_command() -> None:
    rationale = "Correctness oracle: pytest tests/kernels/test_tile.py -q."
    assert parse_correctness_oracles(rationale) == ["pytest tests/kernels/test_tile.py -q"]


def test_oracle_parsing_trims_pytest_command_before_following_prose() -> None:
    rationale = (
        "Correctness oracle: pytest tests/kernels/test_tile.py -q. "
        "Performance oracle is median TPOT."
    )
    assert parse_correctness_oracles(rationale) == ["pytest tests/kernels/test_tile.py -q"]


def test_load_result_missing_field(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"context": {}}', encoding="utf-8")
    with pytest.raises(SelectionError):
        load_result(bad)


def test_parse_repo_path_from_index() -> None:
    text = "## Repository\n- **Repo path:** /Users/me/vllm\n- **X:** y\n"
    assert parse_repo_path_from_index(text) == "/Users/me/vllm"
    assert parse_repo_path_from_index("- **Repo path:** _(unknown)_\n") is None
    assert parse_repo_path_from_index("no marker") is None


def test_resolve_repo_path_from_index_fallback(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path, git=False)
    index = fx.write_index(tmp_path, repo)
    resolved = resolve_repo_path(None, index)
    assert resolved == repo.resolve()


def test_resolve_repo_path_repo_wins(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path, git=False)
    resolved = resolve_repo_path(str(repo), None)
    assert resolved == repo.resolve()


def test_resolve_repo_path_none_resolves(tmp_path: Path) -> None:
    with pytest.raises(RepoResolutionError):
        resolve_repo_path(None, None)
