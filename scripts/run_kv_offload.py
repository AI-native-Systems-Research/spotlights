"""Run candidate_discovery against vllm's kv_offload module.

Usage:
    uv run python scripts/run_kv_offload.py
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

REPO_PATH = Path("/Users/ophir/PycharmProjects/vllm")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "kv_offload"


def main() -> None:
    run_dir = ARTIFACTS_DIR / "candidate_discovery"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = DiscoveryConfig(
        repo_path=REPO_PATH,
        module_qualified_name="v1/kv_offload",
        module=Module(
            name="kv_offload",
            path="vllm/v1/kv_offload",
            description=(
                "KV-cache offload framework with CPU offload manager, reuse "
                "tracking, worker hooks, and offload policies."
            ),
            depends_on=["config", "distributed", "utils"],
            main_files=[
                File(
                    path="vllm/v1/kv_offload/base.py",
                    role="Abstract offload manager interface",
                ),
                File(
                    path="vllm/v1/kv_offload/factory.py",
                    role="Offload manager factory",
                ),
                File(
                    path="vllm/v1/kv_offload/reuse_manager.py",
                    role="Cross-request block-reuse tracking",
                ),
                File(
                    path="vllm/v1/kv_offload/cpu/manager.py",
                    role="CPU offload implementation",
                ),
                File(
                    path="vllm/v1/kv_offload/worker/worker.py",
                    role="Worker-side offload integration",
                ),
            ],
        ),
        artifacts_dir=ARTIFACTS_DIR,
        num_review_iterations=1,
        claude_max_turns=8,
        per_iteration_wallclock_s=600,
    )

    result = discover(cfg)

    print(f"candidates: {len(result.candidates.candidates)}")
    print(f"total_duration_s: {result.total_duration_s:.1f}")
    print(f"total_cost_usd: {result.total_cost_usd}")
    print(f"final artifact: {ARTIFACTS_DIR / 'candidates.json'}")


if __name__ == "__main__":
    main()
