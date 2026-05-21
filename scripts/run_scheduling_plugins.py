"""Run candidate_discovery against llm-d-inference-scheduler's scheduling-plugins module.

Usage:
    uv run --no-sync python scripts/run_scheduling_plugins.py
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path


def _stub_observability_if_missing() -> None:
    try:
        import spotlight_observability  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    pkg = types.ModuleType("spotlight_observability")
    pkg.__path__ = []
    signals = types.ModuleType("spotlight_observability.signals")

    class _Stub:
        pass

    signals.Anomaly = _Stub
    signals.TraceSummary = _Stub
    signals.WorkloadProfile = _Stub
    sys.modules["spotlight_observability"] = pkg
    sys.modules["spotlight_observability.signals"] = signals


_stub_observability_if_missing()

from spotlights_engine.candidate_discovery import DiscoveryConfig, discover  # noqa: E402
from spotlights_engine.schemas.project import File, Module  # noqa: E402

REPO_PATH = Path("/Users/ophir/GoProjects/llm-d-inference-scheduler-main")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "scheduling_plugins"


def main() -> None:
    run_dir = ARTIFACTS_DIR / "candidate_discovery"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    _ctx_path = REPO_PATH / "docs" / "repo_context.md"
    _repo_context = (
        _ctx_path.read_text(encoding="utf-8") if _ctx_path.is_file() else None
    )

    cfg = DiscoveryConfig(
        repo_path=REPO_PATH,
        module_qualified_name="epp/framework/plugins/scheduling",
        module=Module(
            name="scheduling_plugins",
            path="pkg/epp/framework/plugins/scheduling",
            description=(
                "Built-in plugin implementations selected at runtime by the "
                "EPP scheduler: filters (by-label, prefix-cache affinity, "
                "SLO headroom), scorers (KV-cache utilization, prefix, "
                "latency, load-aware, LoRA/session affinity, queue depth, "
                "active-request, token-load), pickers (max-score, random, "
                "weighted-random), and profile handlers (single, disagg "
                "P/D, data-parallel).\n"
                "Role in flow: each enabled plugin runs inside the "
                "scheduler's per-request hot path; scorers loop over "
                "candidate pods and the picker chooses among them.\n"
                "Call frequency: once per request per active plugin, with "
                "scorers further looping over pods; prefix-cache and "
                "KV-cache scorers query llm-d-kv-cache on every scored "
                "request.\n"
                "Headroom signals: per-pod scoring inner loops, scoring "
                "formula and normalization, pod-set short-circuiting, "
                "cache lookup batching, dispatch between plugin variants, "
                "weight tables.\n"
                "Entry points: each plugin directory carries a *Factory "
                "constructor; plugin selection and registration lives in "
                "pkg/epp/framework/plugins/register.go."
            ),
            depends_on=[],
            main_files=[
                File(
                    path="pkg/epp/framework/plugins/scheduling/picker/maxscore",
                    role="Picker that selects the highest-scoring pod",
                ),
                File(
                    path="pkg/epp/framework/plugins/scheduling/scorer/activerequest",
                    role="Scorer penalizing pods with many active requests",
                ),
                File(
                    path="pkg/epp/framework/plugins/scheduling/scorer/prefix",
                    role="Prefix-cache affinity scorer",
                ),
                File(
                    path="pkg/epp/framework/plugins/scheduling/profilehandler/disagg",
                    role="Profile handler for P/D disaggregated scheduling",
                ),
            ],
        ),
        artifacts_dir=ARTIFACTS_DIR,
        repo_context_markdown=_repo_context,
        num_review_iterations=1,
        claude_max_turns=30,
        per_iteration_wallclock_s=600,
    )

    result = discover(cfg)

    print(f"candidates: {len(result.candidates.candidates)}")
    print(f"total_duration_s: {result.total_duration_s:.1f}")
    print(f"total_cost_usd: {result.total_cost_usd}")
    print(f"final artifact: {ARTIFACTS_DIR / 'candidates.json'}")


if __name__ == "__main__":
    main()
