"""Run candidate_discovery against llm-d-inference-scheduler's epp module.

Usage:
    uv run --no-sync python scripts/run_epp.py
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
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "epp"


def main() -> None:
    run_dir = ARTIFACTS_DIR / "candidate_discovery"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = DiscoveryConfig(
        repo_path=REPO_PATH,
        module_qualified_name="epp",
        module=Module(
            name="epp",
            path="pkg/epp",
            description=(
                "Heart of the project: implements the Endpoint Picker (EPP) "
                "— an ext-proc gRPC server that ingests inference requests, "
                "applies flow control, schedules them across `InferencePool` "
                "pods using a plugin pipeline (filters, scorers, pickers), "
                "and tracks per-pod metrics via a data layer."
            ),
            depends_on=[
                "apix",
                "client_go",
                "common",
                "metrics",
                "telemetry",
                "internal",
            ],
            main_files=[
                File(
                    path="pkg/epp/server/runserver.go",
                    role=(
                        "Top-level EPP server assembly: controller manager, "
                        "gRPC ext-proc server, plugin wiring"
                    ),
                ),
                File(
                    path="pkg/epp/handlers/server.go",
                    role=(
                        "Envoy ext-proc gRPC handlers for request/response "
                        "streaming"
                    ),
                ),
                File(
                    path="pkg/epp/requestcontrol/director.go",
                    role=(
                        "Central director that drives admission, routing, "
                        "scheduling, and reporting for each request"
                    ),
                ),
                File(
                    path="pkg/epp/scheduling/scheduler.go",
                    role=(
                        "Plugin-driven scheduler that runs filters/scorers/"
                        "pickers per profile to pick an endpoint"
                    ),
                ),
                File(
                    path="pkg/epp/framework/plugins/register.go",
                    role=(
                        "Registers all built-in EPP plugins (datalayer, "
                        "scheduling, requestcontrol, flowcontrol, "
                        "requesthandling)"
                    ),
                ),
            ],
        ),
        artifacts_dir=ARTIFACTS_DIR,
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
