"""Run candidate_discovery against llm-d-inference-scheduler's scheduling module.

Usage:
    uv run python scripts/run_scheduling.py
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
from spotlights_engine.schemas.modules import File, Module  # noqa: E402

REPO_PATH = Path("/Users/ophir/GoProjects/llm-d-inference-scheduler-main")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "scheduling"


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
        module_qualified_name="epp/scheduling",
        module=Module(
            name="scheduling",
            path="pkg/epp/scheduling",
            description=(
                "Profile-based scheduler that combines filter/scorer/picker "
                "plugins to choose endpoints; supports weighted scorer "
                "composition."
            ),
            depends_on=[],
            main_files=[
                File(
                    path="pkg/epp/scheduling/scheduler.go",
                    role=(
                        "Scheduler entry point: runs configured profiles and "
                        "aggregates results"
                    ),
                ),
                File(
                    path="pkg/epp/scheduling/scheduler_profile.go",
                    role="Single-profile execution: filter -> score -> pick",
                ),
                File(
                    path="pkg/epp/scheduling/weighted_scorer.go",
                    role="Weighted aggregation of multiple scorer outputs",
                ),
                File(
                    path="pkg/epp/scheduling/scheduler_config.go",
                    role="Scheduler configuration model and validation",
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
