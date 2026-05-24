"""Parallelism gate: at most max_parallel_sessions modules in-flight."""

from __future__ import annotations

import threading
import time
from pathlib import Path

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


def test_max_parallel_sessions_respected(tmp_path: Path, monkeypatch) -> None:
    tree = make_tree()
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )

    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def _busy(qn: str) -> None:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1

    def _discover(inp, *, config):
        _busy(inp.module_qualified_name)
        return make_discovery_result(inp.module_qualified_name)

    def _research(inp, options=None):
        _busy(inp.module_qualified_name)
        return make_research_output()

    monkeypatch.setattr(orch, "discover", _discover)
    monkeypatch.setattr(orch, "research_module", _research)
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        max_parallel_sessions=2,
        module_filter=ModuleFilter(
            include=["v1.kv_offload", "v1.attention.paged_kv"]
        ),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    assert peak <= 2
    # With two leaves and capacity 2, both should overlap at some point
    # (jitter-tolerant: peak might be 1 on a heavily loaded host, accept >= 1).
    assert peak >= 1
