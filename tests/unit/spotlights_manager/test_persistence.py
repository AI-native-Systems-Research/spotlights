"""Persistence layer: round-trip, atomic writes, crash recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.candidate_discovery.api import DiscoveryConfig
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.modules_extractor import ExtractorConfig
from spotlights_engine.proposal_from_candidate_finding_creator import (
    ProposalFromCandidateFindingConfig,
)
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.persistence import (
    ManagerPaths,
    ModuleCheckpoint,
)
from tests.unit.spotlights_manager._fakes import (
    make_candidates,
    make_research_output,
)


def test_module_checkpoint_round_trip(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    paths.modules_root.mkdir(parents=True)
    mp = paths.for_module("v1/kv_offload")
    cp = ModuleCheckpoint(
        module_qualified_name="v1/kv_offload",
        status="DISCOVERED",
        last_step="candidate_discovery",
        started_at="2026-05-21T00:00:00+00:00",
        updated_at="2026-05-21T00:00:00+00:00",
    )
    P.write_checkpoint(mp, cp)
    state = P.read_module_state(mp)
    assert state.checkpoint == cp


def test_atomic_write_recovers_from_orphan_tmp(tmp_path: Path) -> None:
    """A leftover `.tmp` from a previous crashed write must not block the next write."""
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)

    # Plant a leftover .tmp file (sim'd crash mid-write).
    (mp.status_path.with_suffix(mp.status_path.suffix + ".tmp")).write_text(
        "garbage", encoding="utf-8"
    )

    cp = ModuleCheckpoint(
        module_qualified_name="v1/kv_offload",
        status="PENDING",
        last_step=None,
        started_at="2026-05-21T00:00:00+00:00",
        updated_at="2026-05-21T00:00:00+00:00",
    )
    P.write_checkpoint(mp, cp)
    payload = json.loads(mp.status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PENDING"


def test_candidates_and_deep_research_round_trip(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)

    cands = make_candidates("v1/kv_offload", n=2)
    P.write_candidates(mp, cands)

    output = make_research_output(n_findings=3)
    P.write_deep_research(mp, output, duration_s=4.2)

    state = P.read_module_state(mp)
    assert state.candidates is not None
    assert len(state.candidates.candidates) == 2
    assert state.deep_research is not None
    assert len(state.deep_research.findings) == 3
    assert state.deep_research_duration_s == 4.2


def test_read_module_state_loads_old_sidecar_without_search_queries(
    tmp_path: Path,
) -> None:
    """Resume back-compat: an old `module_deep_research.json` written before
    `search_queries` existed still loads, defaulting the field to []."""
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)

    # Emulate a pre-feature sidecar: output payload has no search_queries key.
    legacy = {
        "output": {"findings": [], "issues": []},
        "duration_s": 1.0,
    }
    mp.deep_research_path.write_text(json.dumps(legacy), encoding="utf-8")

    state = P.read_module_state(mp)
    assert state.deep_research is not None
    assert state.deep_research.search_queries == []


def test_write_deep_research_search_log_renders_markdown(tmp_path: Path) -> None:
    from spotlights_engine.schemas.search import SearchQueryLog, SearchResult

    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)

    output = make_research_output(n_findings=0)
    output = output.model_copy(
        update={
            "search_queries": [
                SearchQueryLog(
                    agent="codex",
                    query="paged kv cache",
                    tool="web_search",
                    results=[SearchResult(title="vLLM", url="https://x", snippet="s")],
                )
            ]
        }
    )
    P.write_deep_research_search_log(mp, output, "v1/kv_offload")

    md = mp.deep_research_search_log_path.read_text(encoding="utf-8")
    assert "# Deep-research search log — v1/kv_offload" in md
    assert "## Agent: codex" in md
    assert "### 1. `paged kv cache`  (tool: web_search)" in md


def test_proposal_from_finding_round_trip(tmp_path: Path) -> None:
    from spotlights_engine.schemas.candidate import Candidates
    from spotlights_engine.schemas.pipeline import (
        ProposalFromFindingCreatorOutput,
    )

    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)

    cands = make_candidates("v1/kv_offload", n=2)
    advanced = Candidates(
        module_qualified_name=cands.module_qualified_name,
        candidates=[c.model_copy(deep=True) for c in cands.candidates],
    )
    output = ProposalFromFindingCreatorOutput(candidates=advanced, issues=[])
    P.write_proposal_from_finding(
        mp,
        output,
        duration_s=2.5,
        per_pair_durations_s={"cand-0001__find-0001": 1.0},
    )

    state = P.read_module_state(mp)
    assert state.proposal_from_finding is not None
    assert len(state.proposal_from_finding.candidates.candidates) == 2
    assert state.proposal_from_finding_duration_s == 2.5
    assert state.proposal_from_finding_per_pair_durations_s == {"cand-0001__find-0001": 1.0}


def test_clear_proposal_from_finding_artifacts_removes_sidecar_and_dir(
    tmp_path: Path,
) -> None:
    from spotlights_engine.schemas.candidate import Candidates
    from spotlights_engine.schemas.pipeline import (
        ProposalFromFindingCreatorOutput,
    )

    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)

    cands = make_candidates("v1/kv_offload")
    advanced = Candidates(
        module_qualified_name=cands.module_qualified_name,
        candidates=[c.model_copy(deep=True) for c in cands.candidates],
    )
    P.write_proposal_from_finding(
        mp,
        ProposalFromFindingCreatorOutput(candidates=advanced, issues=[]),
        duration_s=0.0,
        per_pair_durations_s={},
    )
    mp.proposal_from_finding_last_message_dir.mkdir()
    (mp.proposal_from_finding_last_message_dir / "cand-0001__find-0001.json").write_text(
        "{}", encoding="utf-8"
    )

    P.clear_proposal_from_finding_artifacts(mp)
    assert not mp.proposal_from_finding_path.exists()
    assert not mp.proposal_from_finding_last_message_dir.exists()


def test_clear_helpers_remove_artifacts(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)
    mp.discovery_run_dir.mkdir()
    (mp.discovery_run_dir / "marker").write_text("x")
    P.write_candidates(mp, make_candidates("v1/kv_offload"))
    P.write_deep_research(mp, make_research_output(), duration_s=1.0)
    mp.deep_research_last_message_path.write_text("hello", encoding="utf-8")
    P.write_deep_research_search_log(mp, make_research_output(), "v1/kv_offload")

    P.clear_discovery_artifacts(mp)
    assert not mp.discovery_run_dir.exists()
    assert not mp.candidates_path.exists()

    P.clear_deep_research_artifacts(mp)
    assert not mp.deep_research_path.exists()
    assert not mp.deep_research_last_message_path.exists()
    assert not mp.deep_research_search_log_path.exists()


def test_clear_deep_research_artifacts_removes_the_candidate_message_dir(
    tmp_path: Path,
) -> None:
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)
    mp.deep_research_last_message_dir.mkdir()
    (mp.deep_research_last_message_dir / "cand-v1_kv_offload-0001.md").write_text(
        "{}", encoding="utf-8"
    )
    P.write_deep_research(mp, make_research_output(), duration_s=1.0)

    P.clear_deep_research_artifacts(mp)

    assert not mp.deep_research_last_message_dir.exists()
    assert not mp.deep_research_path.exists()


def test_clear_deep_research_artifacts_tolerates_a_missing_candidate_dir(
    tmp_path: Path,
) -> None:
    # Module-mode run dirs never have this directory; clearing must not raise.
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1/kv_offload")
    mp.dir.mkdir(parents=True)

    P.clear_deep_research_artifacts(mp)

    assert not mp.deep_research_last_message_dir.exists()


def test_candidate_and_module_last_message_paths_do_not_collide(
    tmp_path: Path,
) -> None:
    mp = ManagerPaths(tmp_path).for_module("v1/kv_offload")

    assert mp.deep_research_last_message_path.name == "module_deep_research.last_message.md"
    assert mp.deep_research_last_message_dir.name == "module_deep_research.last_messages"
    assert mp.deep_research_last_message_dir != mp.deep_research_last_message_path


def test_init_manifest_creates_tree(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    manifest = P.init_manifest(
        paths,
        input_fingerprint={"a": 1},
        config_fingerprint={"b": 2},
        context=SpotlightContext(objective="x"),
    )
    assert paths.manifest_path.exists()
    assert paths.modules_root.exists()
    assert manifest["status"] == "RUNNING"
    re_read = P.read_manifest(paths)
    assert re_read is not None
    assert re_read["input_fingerprint"] == {"a": 1}


def test_slug_for_replaces_unsafe_chars() -> None:
    # Slash-form qns collapse their separators to `_` for the on-disk dir name.
    assert P.slug_for("v1/kv_offload") == "v1_kv_offload"
    assert P.slug_for("v1/attention/paged_kv") == "v1_attention_paged_kv"
    assert P.slug_for("a.b c") == "a.b_c"


def test_config_fingerprint_treats_none_as_effective_defaults() -> None:
    base = P.build_config_fingerprint(
        module_filter=None,
        extractor_cfg=ExtractorConfig(),
        discovery_cfg=None,
        deep_research_cfg=None,
        proposal_from_finding_cfg=None,
        agent_proposals_cfg=None,
    )
    explicit = P.build_config_fingerprint(
        module_filter=None,
        extractor_cfg=ExtractorConfig(),
        discovery_cfg=DiscoveryConfig(),
        deep_research_cfg=CodexExecOptions(),
        proposal_from_finding_cfg=ProposalFromFindingConfig(),
        agent_proposals_cfg=AgentProposalsConfig(),
    )
    assert base == explicit


def test_input_fingerprint_changes_with_enable_claude_search() -> None:
    context = SpotlightContext(objective="reduce latency")
    kwargs: dict[str, Any] = dict(
        repo_path=Path("/tmp/example-repo"),
        context=context,
        max_findings_per_module=30,
        continue_on_module_failure=True,
    )
    off = P.build_input_fingerprint(**kwargs, enable_claude_search=False)
    on = P.build_input_fingerprint(**kwargs, enable_claude_search=True)

    assert off["enable_claude_search"] is False
    assert on["enable_claude_search"] is True
    assert off != on


# mode-aware fingerprints (candidate deep research) ---------------------------

_FP_BASE: dict[str, Any] = dict(
    repo_path=Path("/tmp/example-repo"),
    context=SpotlightContext(objective="reduce latency"),
    max_findings_per_module=30,
    continue_on_module_failure=True,
)


def test_module_mode_input_fingerprint_is_exactly_the_historical_six_keys() -> None:
    # HARD constraint: a run dir written before candidate mode existed must
    # still fingerprint-match, so module mode may not gain or lose a key.
    fp = P.build_input_fingerprint(**_FP_BASE)

    assert set(fp) == {
        "repo_path",
        "context_hash",
        "max_findings_per_module",
        "continue_on_module_failure",
        "include_candidate_hotspots",
        "enable_claude_search",
    }


def test_passing_the_module_mode_default_explicitly_changes_nothing() -> None:
    assert P.build_input_fingerprint(**_FP_BASE) == P.build_input_fingerprint(
        **_FP_BASE, deep_research_mode="module", max_findings_per_candidate=10
    )


def test_candidate_mode_input_fingerprint_omits_the_module_only_knobs() -> None:
    fp = P.build_input_fingerprint(
        **_FP_BASE, deep_research_mode="candidate", max_findings_per_candidate=7
    )

    assert set(fp) == {
        "repo_path",
        "context_hash",
        "continue_on_module_failure",
        "enable_claude_search",
        "deep_research_mode",
        "max_findings_per_candidate",
    }
    # Both modes consume `enable_claude_search`, so it stays effective.
    assert fp["enable_claude_search"] is False
    assert fp["max_findings_per_candidate"] == 7


def test_candidate_mode_ignores_changes_to_knobs_it_never_reads() -> None:
    kwargs = dict(_FP_BASE)
    kwargs["max_findings_per_module"] = 999
    first = P.build_input_fingerprint(
        **_FP_BASE, deep_research_mode="candidate", max_findings_per_candidate=7
    )
    second = P.build_input_fingerprint(
        **kwargs,
        deep_research_mode="candidate",
        max_findings_per_candidate=7,
        include_candidate_hotspots=False,
    )

    assert first == second


def test_switching_mode_can_never_produce_an_equal_input_fingerprint() -> None:
    module_fp = P.build_input_fingerprint(**_FP_BASE)
    candidate_fp = P.build_input_fingerprint(
        **_FP_BASE, deep_research_mode="candidate", max_findings_per_candidate=30
    )

    assert module_fp != candidate_fp


def test_candidate_mode_input_fingerprint_tracks_its_own_cap() -> None:
    first = P.build_input_fingerprint(
        **_FP_BASE, deep_research_mode="candidate", max_findings_per_candidate=5
    )
    second = P.build_input_fingerprint(
        **_FP_BASE, deep_research_mode="candidate", max_findings_per_candidate=6
    )

    assert first != second


def _config_fp(**overrides: Any) -> dict:
    kwargs: dict[str, Any] = dict(
        module_filter=None,
        extractor_cfg=ExtractorConfig(),
        discovery_cfg=None,
        deep_research_cfg=None,
        proposal_from_finding_cfg=None,
        agent_proposals_cfg=None,
    )
    kwargs.update(overrides)
    return P.build_config_fingerprint(**kwargs)


def test_module_mode_config_fingerprint_gains_no_key() -> None:
    # HARD constraint: `build_config_fingerprint` hashes an `or <Default>()` per
    # step, so an unconditional new key would invalidate every existing run dir.
    historical = _config_fp()

    assert "proposal_from_candidate_finding_hash" not in historical
    assert historical == _config_fp(
        proposal_from_candidate_finding_cfg=ProposalFromCandidateFindingConfig(),
        deep_research_mode="module",
    )


def test_candidate_mode_config_fingerprint_adds_its_own_hash() -> None:
    fp = _config_fp(deep_research_mode="candidate")

    assert "proposal_from_candidate_finding_hash" in fp
    assert set(fp) - set(_config_fp()) == {"proposal_from_candidate_finding_hash"}


def test_candidate_mode_config_fingerprint_tracks_the_new_config() -> None:
    default = _config_fp(deep_research_mode="candidate")
    tuned = _config_fp(
        deep_research_mode="candidate",
        proposal_from_candidate_finding_cfg=ProposalFromCandidateFindingConfig(
            max_parallel_pairs=9
        ),
    )

    assert default != tuned
    # Paths are excluded, as they are for every other step's hash.
    assert default == _config_fp(
        deep_research_mode="candidate",
        proposal_from_candidate_finding_cfg=ProposalFromCandidateFindingConfig(
            repo_path=Path("/somewhere"), artifacts_dir=Path("/else")
        ),
    )
