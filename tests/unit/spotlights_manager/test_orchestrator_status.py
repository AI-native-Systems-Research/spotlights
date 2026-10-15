"""Status mapping per plan §5.3: SKIPPED, SUCCEEDED, DEGRADED, FAILED."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.candidate_discovery import (
    DiscoveryAgentFailureError,
    DiscoverySetupError,
    DiscoveryTruncation,
    DiscoveryValidationError,
)
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from tests.unit.spotlights_manager._fakes import (
    make_discovery_result,
    make_extractor_result,
    make_input,
    make_research_output,
    make_tree,
    patch_agent_proposals,
    patch_proposal_from_finding,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    a = tmp_path / "artifacts"
    a.mkdir()
    return a


def _patch_extractor(monkeypatch, tree) -> None:
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )


def test_succeeded_happy_path(monkeypatch, repo: Path, artifacts: Path) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(inp.module_qualified_name),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(n_findings=2),
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "SUCCEEDED"
    assert len(mr.findings) == 2
    assert mr.candidates is not None and len(mr.candidates.candidates) == 1


def test_skipped_when_no_candidates(monkeypatch, repo: Path, artifacts: Path) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(
            inp.module_qualified_name, n_candidates=0
        ),
    )

    research_called = {"n": 0}

    def _no_research(inp, options=None, **_kw):  # pragma: no cover - asserted not called
        research_called["n"] += 1
        return make_research_output()

    monkeypatch.setattr(orch, "research_module", _no_research)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "SKIPPED"
    assert mr.findings == []
    assert research_called["n"] == 0


def _truncation() -> DiscoveryTruncation:
    return DiscoveryTruncation(
        iteration=3,
        agent="codex",
        error="DiscoveryAgentFailureError",
        cause=(
            "schema parse failed twice (iteration=3, agent=codex, cause=codex "
            "last_message.json missing; agent reported: exceeded retry limit, "
            "last status: 429 Too Many Requests)"
        ),
        completed_iterations=3,
    )


def test_degraded_when_discovery_truncated(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """A salvaged discovery keeps its candidates and reports DEGRADED, not FAILED.

    Regression guard for the IOCR run: eight modules were marked FAILED with
    zero candidates because a rate limit hit their last refinement iteration.
    """
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(
            inp.module_qualified_name, truncated_by=_truncation()
        ),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(n_findings=2),
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "DEGRADED"
    # The candidates are kept, and the pipeline ran on them.
    assert mr.candidates is not None and len(mr.candidates.candidates) == 1
    assert len(mr.findings) == 2
    # The issue names the real cause and is recoverable, so a resume can retry.
    issues = [i for i in mr.issues if i.step == "candidate_discovery"]
    assert len(issues) == 1
    assert issues[0].recoverable is True
    assert "429" in issues[0].message
    assert "iteration 3" in issues[0].message


def test_discovery_truncation_persists_for_resume(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """The truncation lands in the telemetry sidecar and reloads from it."""
    import json

    from spotlights_engine.spotlights_manager import persistence as P

    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(
            inp.module_qualified_name, truncated_by=_truncation()
        ),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(n_findings=1),
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    module_paths = P.ManagerPaths(artifacts).for_module("v1/kv_offload")
    payload = json.loads(module_paths.discovery_telemetry_path.read_text())
    assert payload["truncated_by"]["iteration"] == 3
    assert payload["truncated_by"]["agent"] == "codex"

    reloaded = P.read_module_state(module_paths)
    assert reloaded.discovery_truncation is not None
    assert reloaded.discovery_truncation.completed_iterations == 3


def test_degraded_on_recoverable_issue(monkeypatch, repo: Path, artifacts: Path) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(inp.module_qualified_name),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(
            n_findings=1,
            issues=[
                StepIssue(
                    step="module_deep_research",
                    severity="warning",
                    message="rate limited",
                    recoverable=True,
                )
            ],
        ),
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "DEGRADED"
    assert len(mr.findings) == 1


def test_degraded_on_step4_recoverable_issue(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(inp.module_qualified_name),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(n_findings=1),
    )

    def _fake_step4(
        inp,
        *,
        config,
        runner=None,
        candidate_states=None,
        proposal_id_start=1,
        segment=None,
    ):
        from spotlights_engine.proposal_from_finding_creator.api import (
            ProposalFromFindingCreatorResult,
        )
        from spotlights_engine.schemas.candidate import Candidates
        from spotlights_engine.schemas.common import StepIssue
        from spotlights_engine.schemas.pipeline import (
            ProposalFromFindingCreatorOutput,
        )

        cands = inp.candidates
        # New-shape candidates carry no `state`; a step-4 run minting no
        # research-backed proposals returns them unchanged.
        advanced = Candidates(
            module_qualified_name=cands.module_qualified_name,
            candidates=[c.model_copy(deep=True) for c in cands.candidates],
        )
        return ProposalFromFindingCreatorResult(
            output=ProposalFromFindingCreatorOutput(
                candidates=advanced,
                issues=[
                    StepIssue(
                        step="proposal_from_finding_creator",
                        severity="warning",
                        message="agent failure (cand-0001, find-0001): timeout",
                        recoverable=True,
                    )
                ],
            ),
            per_pair_durations_s={},
            total_duration_s=0.0,
        )

    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _fake_step4)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)
    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "DEGRADED"
    assert any(
        iss.step == "proposal_from_finding_creator" and iss.recoverable
        for iss in mr.issues
    )


def test_failed_on_unrecoverable_issue(monkeypatch, repo: Path, artifacts: Path) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(inp.module_qualified_name),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(
            n_findings=0,
            issues=[
                StepIssue(
                    step="module_deep_research",
                    severity="error",
                    message="schema invalid",
                    recoverable=False,
                )
            ],
        ),
    )
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "FAILED"
    assert any(not iss.recoverable for iss in mr.issues)


def test_step2_exception_marks_failed_retryable(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)

    def _boom(inp, *, config):
        raise RuntimeError("upstream went away")

    monkeypatch.setattr(orch, "discover", _boom)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "FAILED"
    assert mr.candidates is None
    assert any(iss.recoverable for iss in mr.issues)


def _run_with_discovery_raising(
    monkeypatch, repo: Path, artifacts: Path, exc: Exception
):
    """Run one module whose step 2 dies, and hand back its checkpoint."""
    from spotlights_engine.spotlights_manager import persistence as P

    tree = make_tree()
    _patch_extractor(monkeypatch, tree)

    def _boom(inp, *, config):
        raise exc

    monkeypatch.setattr(orch, "discover", _boom)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)
    module_paths = P.ManagerPaths(artifacts).for_module("v1/kv_offload")
    checkpoint = P.ModuleCheckpoint.model_validate_json(
        module_paths.status_path.read_text(encoding="utf-8")
    )
    return result, checkpoint


def test_an_unsalvageable_rate_limit_stays_retryable(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """A 429 in a bootstrap pass must not cost the module permanently.

    `DiscoveryAgentFailureError` subclasses `DiscoveryValidationError` so older
    handlers keep working, which makes it easy to file an environmental failure
    as a contract violation. If that happens the checkpoint records
    `retryable=False`, `_plan_module` returns `skip=True` for it, and no
    `--resume` will ever try the module again -- the transient becomes permanent.
    The salvage path in the discovery orchestrator cannot cover this case: with
    no earlier iteration to keep, it re-raises.
    """
    result, cp = _run_with_discovery_raising(
        monkeypatch,
        repo,
        artifacts,
        DiscoveryAgentFailureError(
            "schema parse failed twice",
            iteration=0,
            agent="claude_code",
            cause="exceeded retry limit, last status: 429 Too Many Requests",
        ),
    )

    assert cp.status == "FAILED"
    assert cp.retryable is True
    assert "429" in (cp.error or "")
    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "FAILED"
    assert all(iss.recoverable for iss in mr.issues)


def test_a_contract_violation_is_still_not_retryable(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """The other half: nothing a rerun can do about a malformed payload.

    Retrying it would spend a module's discovery budget to reach the same wall,
    so the distinction the subclass draws has to survive in both directions.
    """
    _, cp = _run_with_discovery_raising(
        monkeypatch,
        repo,
        artifacts,
        DiscoveryValidationError("candidate ids are not unique", iteration=2),
    )

    assert cp.status == "FAILED"
    assert cp.retryable is False


def test_fail_fast_writes_retryable_checkpoint_for_not_started_module(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    discover_calls: list[str] = []

    def _discover(inp, *, config):
        discover_calls.append(inp.module_qualified_name)
        if inp.module_qualified_name == "v1/kv_offload":
            raise DiscoverySetupError("bad discovery config")
        return make_discovery_result(inp.module_qualified_name)

    monkeypatch.setattr(orch, "discover", _discover)
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(n_findings=1),
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        max_parallel_sessions=1,
        module_filter=ModuleFilter(
            include=["v1/kv_offload", "v1/attention/paged_kv"]
        ),
    )
    inp = make_input(repo).model_copy(update={"continue_on_module_failure": False})
    result = run_with_telemetry(inp, config=cfg)

    assert discover_calls == ["v1/kv_offload"]
    mr = result.module_runs["v1/attention/paged_kv"]
    assert mr.status == "FAILED"
    assert mr.candidates is None
    assert mr.issues and all(iss.recoverable for iss in mr.issues)

    status_path = (
        artifacts
        / "spotlights_manager"
        / "modules"
        / "v1_attention_paged_kv"
        / "status.json"
    )
    assert status_path.exists()


def test_fail_fast_cancels_after_unhandled_task_exception(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    _patch_extractor(monkeypatch, tree)
    discover_calls: list[str] = []

    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: (
            discover_calls.append(inp.module_qualified_name),
            make_discovery_result(inp.module_qualified_name),
        )[1],
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(n_findings=1),
    )

    original_write_checkpoint = orch.P.write_checkpoint

    def _write_checkpoint(module_paths, checkpoint):
        original_write_checkpoint(module_paths, checkpoint)
        if (
            checkpoint.module_qualified_name == "v1/kv_offload"
            and checkpoint.status == "PENDING"
        ):
            raise RuntimeError("checkpoint write failed")

    monkeypatch.setattr(orch.P, "write_checkpoint", _write_checkpoint)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        max_parallel_sessions=1,
        module_filter=ModuleFilter(
            include=["v1/kv_offload", "v1/attention/paged_kv"]
        ),
    )
    inp = make_input(repo).model_copy(update={"continue_on_module_failure": False})
    result = run_with_telemetry(inp, config=cfg)

    failed = result.module_runs["v1/kv_offload"]
    assert failed.status == "FAILED"
    assert any(
        "checkpoint write failed" in iss.message and not iss.recoverable
        for iss in failed.issues
    )
    assert discover_calls == []

    cancelled = result.module_runs["v1/attention/paged_kv"]
    assert cancelled.status == "FAILED"
    assert cancelled.issues and all(iss.recoverable for iss in cancelled.issues)
