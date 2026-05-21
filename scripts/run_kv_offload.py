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
from spotlights_engine.schemas.common import SpotlightContext  # noqa: E402
from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput  # noqa: E402
from spotlights_engine.schemas.project import (  # noqa: E402
    File,
    Module,
    ProjectTree,
    Repository,
)

REPO_PATH = Path("/Users/ophir/PycharmProjects/vllm")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "kv_offload"
MODULE_QUALIFIED_NAME = "v1.kv_offload"


def build_project_tree() -> ProjectTree:
    kv_offload = Module(
        name="kv_offload",
        path="vllm/v1/kv_offload",
        description=(
            "KV-cache offload framework that mediates between GPU "
            "KV-cache blocks and CPU host memory in vLLM's V1 stack: "
            "routes block lookup, eviction, and cross-request reuse.\n"
            "Role in flow: sits behind the V1 KV-cache manager; worker "
            "hooks fire on forward-pass steps where blocks need "
            "migration between GPU and host.\n"
            "Call frequency: per-block during prefill cache fills and "
            "evictions; reuse-manager consulted on each block lookup.\n"
            "Headroom signals: offload policy choice, transfer "
            "batching, pinned host-buffer reuse, lookup-table layout, "
            "factory dispatch.\n"
            "Entry points: base.py (abstract manager interface), "
            "factory.py (backend selection), worker/worker.py "
            "(per-step hook integration)."
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
    )

    return ProjectTree(
        repository=Repository(
            name="vllm",
            summary=(
                "High-throughput LLM inference and serving engine; "
                "the V1 stack hosts the paged KV-cache manager and "
                "the GPU/CPU offload framework under inspection."
            ),
            external_dependencies=["pytorch", "cuda", "ray"],
        ),
        modules=[
            Module(
                name="v1",
                path="vllm/v1",
                description="V1 inference stack: scheduler, KV-cache manager, offload, worker hooks.",
                submodules=[kv_offload],
            ),
        ],
    )


def main() -> None:
    run_dir = ARTIFACTS_DIR / "candidate_discovery"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    _ctx_path = REPO_PATH / "docs" / "repo_context.md"
    _repo_context = (
        _ctx_path.read_text(encoding="utf-8") if _ctx_path.is_file() else None
    )

    inp = CandidateDiscoveryInput(
        project_tree=build_project_tree(),
        module_qualified_name=MODULE_QUALIFIED_NAME,
        context=SpotlightContext(
            objective=(
                "expand effective KV-cache capacity and hit rate by improving "
                "GPU<->CPU offload throughput and cross-request block reuse"
            ),
            workload_hints=[
                "decode-heavy traffic with long prompts",
                "8k-128k context windows",
                "single-node multi-GPU A100/H100 with PCIe host memory",
                "high request concurrency with shared prefixes",
            ],
        ),
    )
    cfg = DiscoveryConfig(
        repo_path=REPO_PATH,
        artifacts_dir=ARTIFACTS_DIR,
        repo_context_markdown=_repo_context,
        num_review_iterations=1,
        claude_max_turns=30,
        per_iteration_wallclock_s=600,
    )

    result = discover(inp, config=cfg)

    print(f"candidates: {len(result.candidates.candidates)}")
    print(f"total_duration_s: {result.total_duration_s:.1f}")
    print(f"total_cost_usd: {result.total_cost_usd}")
    print(f"final artifact: {ARTIFACTS_DIR / 'candidates.json'}")


if __name__ == "__main__":
    main()
