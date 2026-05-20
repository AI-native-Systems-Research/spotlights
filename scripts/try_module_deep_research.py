"""Run module deep research against vllm's kv_offload module.

Usage:
    uv run --no-sync python scripts/try_module_deep_research.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spotlights_engine.module_deep_research import (
    CodexExecOptions,
    research_module,
)
from spotlights_engine.schemas.deep_research import (
    ModuleDeepResearchInput,
    SpotlightContext,
)
from spotlights_engine.schemas.modules import File, Module, ProjectTree, Repository

REPO_PATH = Path("/Users/ophir/PycharmProjects/vllm")
MODULE_QUALIFIED_NAME = "v1/kv_offload"


def build_kv_offload_tree() -> ProjectTree:
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
                path="vllm/v1/kv_offload/cpu/spec.py",
                role="CPU offloading spec, configuration, and handler construction",
            ),
            File(
                path="vllm/v1/kv_offload/cpu/manager.py",
                role="CPU offload implementation",
            ),
            File(
                path="vllm/v1/kv_offload/cpu/gpu_worker.py",
                role="CPU/GPU transfer handlers and batched block-copy setup",
            ),
            File(
                path="vllm/v1/kv_offload/cpu/policies/base.py",
                role="CPU cache-policy interface and invariants",
            ),
            File(
                path="vllm/v1/kv_offload/cpu/policies/lru.py",
                role="LRU CPU cache-policy implementation",
            ),
            File(
                path="vllm/v1/kv_offload/cpu/policies/arc.py",
                role="ARC CPU cache-policy implementation",
            ),
            File(
                path="vllm/v1/kv_offload/cpu/shared_offload_region.py",
                role="Shared mmap-backed host-memory region",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        default=(
            "expand effective KV-cache capacity and hit rate by improving "
            "GPU<->CPU offload throughput and cross-request block reuse"
        ),
        help="Caller objective threaded into the prompt.",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=10,
        help="Cap on findings the agent is allowed to return.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=REPO_PATH,
        help="Working directory passed to `codex exec`.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional codex --model override (e.g. gpt-5-codex-pro).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    request = ModuleDeepResearchInput(
        project_tree=build_kv_offload_tree(),
        module_qualified_name=MODULE_QUALIFIED_NAME,
        context=SpotlightContext(
            objective=args.objective,
            workload_hints=[
                "decode-heavy traffic with long prompts",
                "8k-128k context windows",
                "single-node multi-GPU A100/H100 with PCIe host memory",
                "high request concurrency with shared prefixes",
            ],
        ),
        max_findings_per_module=args.max_findings,
    )

    print(f"[codex] researching module={MODULE_QUALIFIED_NAME!r} cwd={args.cwd}")
    output = research_module(
        request,
        codex_options=CodexExecOptions(cwd=args.cwd, model=args.model),
    )

    print()
    print(f"findings: {len(output.findings)}")
    for f in output.findings:
        print(f"  {f.finding_id}  [{f.source_type}]  {f.title}")
        print(f"    url: {f.url}")
        print(f"    summary: {f.technique_summary}")
        if f.supporting_evidence:
            print(f"    evidence: {f.supporting_evidence}")
    print()
    print(f"issues: {len(output.issues)}")
    for i in output.issues:
        marker = "recoverable" if i.recoverable else "FATAL"
        print(f"  [{i.severity}/{marker}] {i.message}")


if __name__ == "__main__":
    main()
