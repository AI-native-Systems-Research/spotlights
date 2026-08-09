"""Mode dispatch for steps 3-4, plus the module-mode regression backstops.

The whole point of the two-mode change is that `module` mode's runtime behaviour
is preserved; the `_module_mode_*` tests here are the regressions that pin it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from spotlights_engine.candidate_deep_research import (
    CandidateCliUsage,
    CandidateDeepResearchResult,
)
from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.proposal_from_candidate_finding_creator import (
    ProposalFromCandidateFindingConfig,
)
from spotlights_engine.proposal_from_candidate_finding_creator.api import (
    ProposalFromCandidateFindingCreatorResult,
)
from spotlights_engine.proposal_from_finding_creator import ProposalFromFindingConfig
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ProposalFromFindingCreatorOutput
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    ResumeMismatchError,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.persistence import ManagerPaths
from tests.unit.spotlights_manager._fakes import (
    make_discovery_result,
    make_extractor_result,
    make_input,
    make_research_output,
    make_tree,
    patch_agent_proposals,
    patch_proposal_from_finding,
)

QN = "v1/kv_offload"
SEGMENT = "v1_kv_offload"


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


def _candidate_finding(candidate_idx: int, finding_idx: int = 0) -> Finding:
    return Finding(
        finding_id=(f"find-{SEGMENT}-{candidate_idx + 1:04d}-{finding_idx + 1:04d}"),
        title="t",
        url=f"https://example.com/{candidate_idx}-{finding_idx}",
        source_type="paper",
        technique_summary="s",
        candidate_id=f"cand-{SEGMENT}-{candidate_idx + 1:04d}",
    )


def _cfg(artifacts: Path, **kwargs) -> SpotlightsManagerConfig:
    return SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=[QN]),
        **kwargs,
    )


def _candidate_input(repo: Path, *, n_candidates: int = 2, **overrides):
    update = {"deep_research_mode": "candidate"}
    update.update(overrides)
    return make_input(repo).model_copy(update=update)


def _wire(
    monkeypatch,
    *,
    n_candidates: int = 2,
    research=None,
    candidate_research=None,
    candidate_step4=None,
) -> None:
    tree = make_tree()
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(
            inp.module_qualified_name, n_candidates=n_candidates
        ),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        research or (lambda inp, options=None, **_kw: make_research_output(n_findings=1)),
    )
    monkeypatch.setattr(
        orch,
        "research_candidates",
        candidate_research
        or (
            lambda inp, options=None, **_kw: orch.ModuleDeepResearchOutput(
                findings=[_candidate_finding(i) for i in range(len(inp.candidates))]
            )
        ),
    )
    patch_proposal_from_finding(monkeypatch, orch)
    if candidate_step4 is not None:
        monkeypatch.setattr(orch, "create_candidate_proposals_with_telemetry", candidate_step4)
    else:
        monkeypatch.setattr(
            orch,
            "create_candidate_proposals_with_telemetry",
            _fake_candidate_step4(),
        )
    patch_agent_proposals(monkeypatch, orch)


def _fake_candidate_step4(seen: list | None = None):
    def _fake(
        inp,
        *,
        config,
        runner=None,
        candidate_states=None,
        proposal_id_start=1,
        segment=None,
    ):
        if seen is not None:
            seen.append((inp, config))
        return ProposalFromCandidateFindingCreatorResult(
            output=ProposalFromFindingCreatorOutput(
                candidates=Candidates(
                    module_qualified_name=inp.candidates.module_qualified_name,
                    candidates=[c.model_copy(deep=True) for c in inp.candidates.candidates],
                ),
                issues=[],
            ),
            per_pair_durations_s={},
            total_duration_s=0.0,
        )

    return _fake


# module-mode regressions -----------------------------------------------------


def test_module_mode_is_the_default_and_still_calls_research_module(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    module_calls: list = []
    candidate_calls: list = []

    def _research(inp, options=None, **_kw):
        module_calls.append(inp)
        return make_research_output(n_findings=1)

    def _research_candidates(inp, options=None, **_kw):
        candidate_calls.append(inp)
        return orch.ModuleDeepResearchOutput()

    _wire(monkeypatch, research=_research, candidate_research=_research_candidates)

    run_with_telemetry(make_input(repo), config=_cfg(artifacts))

    assert len(module_calls) == 1
    assert candidate_calls == []


def test_module_mode_research_module_monkeypatch_seam_survives() -> None:
    # Several existing tests patch `orch.research_module`; the alias and its
    # candidate-mode sibling must both stay module-level names.
    assert orch.research_module is not None
    assert orch.research_candidates is not None
    assert orch.research_module is not orch.research_candidates


def test_module_mode_findings_sidecar_carries_null_candidate_id(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _wire(monkeypatch)

    run_with_telemetry(make_input(repo), config=_cfg(artifacts))

    mp = ManagerPaths(artifacts).for_module(QN)
    payload = json.loads(mp.deep_research_path.read_text(encoding="utf-8"))
    findings = payload["output"]["findings"]
    assert findings
    assert all(f["candidate_id"] is None for f in findings)


def test_module_mode_uses_the_single_last_message_file(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    seen: list[CodexExecOptions | None] = []

    def _research(inp, options=None, **_kw):
        seen.append(options)
        return make_research_output(n_findings=1)

    _wire(monkeypatch, research=_research)

    run_with_telemetry(make_input(repo), config=_cfg(artifacts))

    mp = ManagerPaths(artifacts).for_module(QN)
    assert seen[0] is not None
    assert seen[0].output_last_message == mp.deep_research_last_message_path
    assert not mp.deep_research_last_message_dir.exists()


def test_module_mode_pair_count_log_is_unchanged(
    monkeypatch, repo: Path, artifacts: Path, caplog
) -> None:
    # 2 candidates x 2 module-wide findings = the historical cartesian count.
    _wire(
        monkeypatch,
        n_candidates=2,
        research=lambda inp, options=None, **_kw: make_research_output(n_findings=2),
    )

    with caplog.at_level(logging.INFO):
        run_with_telemetry(make_input(repo), config=_cfg(artifacts))

    messages = [r.getMessage() for r in caplog.records]
    assert any("proposal_from_finding: start — 4 (candidate, finding) pairs" in m for m in messages)


def test_module_mode_usage_records_keep_their_historical_invocation_ids(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    from spotlights_engine.costing.usage import CliUsage
    from spotlights_engine.module_deep_research.api import ModuleDeepResearchResult

    def _research(inp, options=None, **_kw):
        return ModuleDeepResearchResult(
            output=make_research_output(n_findings=1),
            usages=[CliUsage(cli="codex", usage=AgentUsage(input=10, output=20))],
        )

    _wire(monkeypatch, research=_research)

    run_with_telemetry(make_input(repo), config=_cfg(artifacts))

    ids = _usage_invocation_ids(artifacts)
    assert ids == ["deep-research:codex"]


def _usage_invocation_ids(artifacts: Path) -> list[str]:
    mp = ManagerPaths(artifacts).for_module(QN)
    records, notes = P.read_usage_records(mp, "module_deep_research")
    assert notes == []
    return [r.invocation_id for r in records if r.role == "deep_research"]


# candidate-mode dispatch -----------------------------------------------------


def test_candidate_mode_dispatches_step3_to_research_candidates(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    module_calls: list = []
    candidate_calls: list = []

    def _research(inp, options=None, **_kw):
        module_calls.append(inp)
        return make_research_output(n_findings=1)

    def _research_candidates(inp, options=None, **_kw):
        candidate_calls.append(inp)
        return orch.ModuleDeepResearchOutput(
            findings=[_candidate_finding(i) for i in range(len(inp.candidates))]
        )

    _wire(monkeypatch, research=_research, candidate_research=_research_candidates)

    result = run_with_telemetry(_candidate_input(repo), config=_cfg(artifacts))

    assert module_calls == []
    assert len(candidate_calls) == 1
    request = candidate_calls[0]
    # The step-3 input is the candidate-mode contract, and the candidates are
    # the iteration set rather than advisory hot spots.
    assert type(request).__name__ == "CandidateDeepResearchInput"
    assert len(request.candidates) == 2
    assert result.module_runs[QN].status == "SUCCEEDED"


def test_candidate_mode_threads_its_consensus_knobs(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    seen: list = []

    def _research_candidates(inp, options=None, **_kw):
        seen.append(inp)
        return orch.ModuleDeepResearchOutput(findings=[_candidate_finding(0)])

    _wire(monkeypatch, candidate_research=_research_candidates)

    run_with_telemetry(
        _candidate_input(repo, num_search_runs=4, search_consensus_threshold=2),
        config=_cfg(artifacts),
    )

    assert seen[0].num_search_runs == 4
    assert seen[0].search_consensus_threshold == 2


def test_candidate_mode_passes_the_per_candidate_last_message_dir(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    kwargs: list[dict] = []

    def _research_candidates(inp, options=None, **kw):
        kwargs.append(kw)
        return orch.ModuleDeepResearchOutput(findings=[_candidate_finding(0)])

    _wire(monkeypatch, candidate_research=_research_candidates)

    run_with_telemetry(_candidate_input(repo), config=_cfg(artifacts))

    mp = ManagerPaths(artifacts).for_module(QN)
    assert kwargs[0]["last_message_dir"] == mp.deep_research_last_message_dir
    assert kwargs[0]["segment"] == SEGMENT


def test_candidate_mode_leaves_output_last_message_unset_on_the_base_options(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    # A shared `output_last_message` would let one candidate's survey read
    # another's JSON; the base options must not carry one (D6).
    seen: list[CodexExecOptions | None] = []

    def _research_candidates(inp, options=None, **_kw):
        seen.append(options)
        return orch.ModuleDeepResearchOutput(findings=[_candidate_finding(0)])

    _wire(monkeypatch, candidate_research=_research_candidates)

    run_with_telemetry(
        _candidate_input(repo),
        config=_cfg(artifacts, deep_research=CodexExecOptions(model="custom-model")),
    )

    assert seen[0] is not None
    assert seen[0].output_last_message is None
    assert seen[0].cwd == repo
    assert seen[0].model == "custom-model"


def test_candidate_mode_dispatches_step4_to_the_sibling_package(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    seen: list = []
    _wire(monkeypatch, candidate_step4=_fake_candidate_step4(seen))

    run_with_telemetry(_candidate_input(repo), config=_cfg(artifacts))

    assert len(seen) == 1
    _inp, config = seen[0]
    assert isinstance(config, ProposalFromCandidateFindingConfig)
    assert config.repo_path == repo
    assert config.artifacts_dir == ManagerPaths(artifacts).for_module(QN).dir


def test_candidate_mode_step4_config_reads_its_own_slot(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    seen: list = []
    _wire(monkeypatch, candidate_step4=_fake_candidate_step4(seen))

    run_with_telemetry(
        _candidate_input(repo),
        config=_cfg(
            artifacts,
            proposal_from_candidate_finding=ProposalFromCandidateFindingConfig(
                max_parallel_pairs=3, debug_first_n_pairs=2
            ),
            # The module-mode slot must be ignored in candidate mode.
            proposal_from_finding=ProposalFromFindingConfig(max_parallel_pairs=9),
        ),
    )

    _inp, config = seen[0]
    assert config.max_parallel_pairs == 3
    assert config.debug_first_n_pairs == 2


def test_candidate_mode_pair_count_log_is_the_grouped_count(
    monkeypatch, repo: Path, artifacts: Path, caplog
) -> None:
    # 2 candidates, 2 findings each => 4 grouped pairs (cartesian would be 8).
    _wire(
        monkeypatch,
        n_candidates=2,
        candidate_research=lambda inp, options=None, **_kw: orch.ModuleDeepResearchOutput(
            findings=[_candidate_finding(c, f) for c in range(2) for f in range(2)]
        ),
    )

    with caplog.at_level(logging.INFO):
        run_with_telemetry(_candidate_input(repo), config=_cfg(artifacts))

    messages = [r.getMessage() for r in caplog.records]
    assert any("proposal_from_finding: start — 4 (candidate, finding) pairs" in m for m in messages)


def test_candidate_mode_usage_records_are_keyed_by_candidate(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    def _research_candidates(inp, options=None, **_kw):
        return CandidateDeepResearchResult(
            output=orch.ModuleDeepResearchOutput(
                findings=[_candidate_finding(i) for i in range(2)]
            ),
            usages=[
                CandidateCliUsage(
                    candidate_id=f"cand-{SEGMENT}-{i + 1:04d}",
                    cli="codex",
                    usage=AgentUsage(input=10, output=20),
                    duration_s=1.5,
                )
                for i in range(2)
            ],
        )

    _wire(monkeypatch, candidate_research=_research_candidates)

    run_with_telemetry(_candidate_input(repo), config=_cfg(artifacts))

    assert _usage_invocation_ids(artifacts) == [
        f"cand-{SEGMENT}-0001:codex",
        f"cand-{SEGMENT}-0002:codex",
    ]


def test_candidate_mode_findings_persist_with_their_candidate_tag(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _wire(monkeypatch)

    run_with_telemetry(_candidate_input(repo), config=_cfg(artifacts))

    mp = ManagerPaths(artifacts).for_module(QN)
    payload = json.loads(mp.deep_research_path.read_text(encoding="utf-8"))
    tags = [f["candidate_id"] for f in payload["output"]["findings"]]
    assert tags == [f"cand-{SEGMENT}-0001", f"cand-{SEGMENT}-0002"]


# resume ----------------------------------------------------------------------


def test_switching_mode_on_resume_raises_resume_mismatch(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _wire(monkeypatch)
    cfg = _cfg(artifacts)

    run_with_telemetry(make_input(repo), config=cfg)

    with pytest.raises(ResumeMismatchError):
        run_with_telemetry(_candidate_input(repo), config=cfg)


def test_candidate_mode_resumes_itself_cleanly(monkeypatch, repo: Path, artifacts: Path) -> None:
    research_calls: list = []

    def _research_candidates(inp, options=None, **_kw):
        research_calls.append(inp)
        return orch.ModuleDeepResearchOutput(
            findings=[_candidate_finding(i) for i in range(len(inp.candidates))]
        )

    _wire(monkeypatch, candidate_research=_research_candidates)
    cfg = _cfg(artifacts)

    run_with_telemetry(_candidate_input(repo), config=cfg)
    run_with_telemetry(_candidate_input(repo), config=cfg)

    # The second run resumes from the sidecar rather than re-surveying.
    assert len(research_calls) == 1


def test_candidate_mode_manifest_fingerprint_records_the_mode(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _wire(monkeypatch)

    run_with_telemetry(_candidate_input(repo), config=_cfg(artifacts))

    manifest = P.read_manifest(ManagerPaths(artifacts))
    assert manifest is not None
    assert manifest["input_fingerprint"]["deep_research_mode"] == "candidate"
    assert "proposal_from_candidate_finding_hash" in manifest["config_fingerprint"]


def test_module_mode_manifest_fingerprint_keeps_the_historical_keys(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _wire(monkeypatch)

    run_with_telemetry(make_input(repo), config=_cfg(artifacts))

    manifest = P.read_manifest(ManagerPaths(artifacts))
    assert manifest is not None
    assert "deep_research_mode" not in manifest["input_fingerprint"]
    assert "proposal_from_candidate_finding_hash" not in manifest["config_fingerprint"]
