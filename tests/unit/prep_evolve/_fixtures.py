"""Fixture builders for prep_evolve tests.

`make_result_dict()` returns a minimal `result.json`-shaped payload (the
architecture `SpotlightsResult` shape, which `load_result` also accepts). It
contains a nested project tree (so slash-form submodule resolution is
exercised), one candidate with both deep-research and agent proposals, and a
findings list with one candidate-linked finding and one unlinked finding.

`make_repo(tmp_path)` writes a synthetic source tree on disk that matches the
candidate's file/line range so `validate_target` passes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# The candidate target file + range used across tests.
CAND_FILE = "pkg/attn/tile.py"
CAND_START = 3
CAND_END = 5
CAND_SYMBOL = "_get_tile_size"
MAIN_FILE_EXTRA = "pkg/attn/launch.py"


def make_result_dict() -> dict:
    """A result.json payload: nested tree, one candidate, mixed proposals."""
    candidate = {
        "id": "cand-0002",
        "file": CAND_FILE,
        "line_start": CAND_START,
        "line_end": CAND_END,
        "symbol": CAND_SYMBOL,
        "kind": "function",
        "description": "Tile-size heuristic for the attention kernel.",
        "current_approach": "A compact closed-form heuristic.",
        "evolve_rationale": (
            "Replace with a tuning table. Correctness oracle: "
            "tests/kernels/test_tile.py covers it. Performance oracle is "
            "median TPOT and TTFT."
        ),
        "estimated_impact": "high",
        "estimated_impact_explanation": "Drives occupancy; affects throughput.",
        "state": "AGENT_PROPOSALS_CREATED",
        "deep_research_proposals": [
            {
                "title": "Make tile size GQA-aware",
                "detailed_description": "Use a register-budget formula.",
                "finding_id": "find-0001",
                "proposal_rationale": "POD-Attention suggests this.",
                "created_by": "claude",
            }
        ],
        "agent_proposals": [
            {
                "title": "Use decode tile size for 2D launches",
                "detailed_description": "Branch on decode vs prefill.",
                "agent_name": "codex",
                "novelty_rationale": "Not covered by the findings.",
            }
        ],
    }

    findings = [
        {
            "finding_id": "find-0002",
            "title": "Unlinked broader-context finding",
            "url": "https://example.com/find2",
            "source_type": "blog",
            "technique_summary": "General attention tuning tips.",
            "supporting_evidence": "",
        },
        {
            "finding_id": "find-0001",
            "title": "POD-Attention",
            "url": "https://arxiv.org/abs/2410.18038",
            "source_type": "paper",
            "technique_summary": "Overlap prefill and decode.",
            "supporting_evidence": "2x throughput in the paper.",
        },
    ]

    # Nested tree: attention is a submodule of v1. With source_root="" the
    # qualified names are the (normalized) repo-relative paths, so the leaf is
    # `v1/attention` — the single canonical (slash-form) qualified name. Module
    # `name` must equal the normalized basename of `path`. The candidate/main_files
    # paths under `pkg/attn/` are independent on-disk file references (see make_repo).
    tree = {
        "repository": {
            "name": "demo",
            "summary": "Demo repo.",
            "source_root": "",
            "external_dependencies": ["torch"],
        },
        "modules": [
            {
                "name": "v1",
                "path": "v1",
                "description": "v1 core.",
                "depends_on": [],
                "main_files": [],
                "submodules": [
                    {
                        "name": "attention",
                        "path": "v1/attention",
                        "description": "Attention kernels.",
                        "depends_on": [],
                        "main_files": [
                            {"path": CAND_FILE, "role": "kernel"},
                            {"path": MAIN_FILE_EXTRA, "role": "launcher"},
                        ],
                        "submodules": [],
                    }
                ],
            }
        ],
    }

    return {
        "project_tree": tree,
        "context": {
            "objective": "reduce the median TTFT and median TPOT",
            "workload_hints": ["multi-turn agentic workload"],
            "validation_plan": [],
        },
        "module_runs": {
            "v1/attention": {
                "module_qualified_name": "v1/attention",
                "status": "SUCCEEDED",
                "candidates": {
                    "module_qualified_name": "v1/attention",
                    "candidates": [candidate],
                },
                "findings": findings,
                "issues": [],
            }
        },
    }


def write_result(tmp_path: Path) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(make_result_dict()), encoding="utf-8")
    return path


def make_repo(tmp_path: Path, *, git: bool = True) -> Path:
    """Synthetic repo matching the candidate file/lines."""
    repo = tmp_path / "repo"
    for rel in (CAND_FILE, MAIN_FILE_EXTRA):
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# line {i}" for i in range(1, CAND_END + 4)]
        lines[CAND_START - 1] = f"def {CAND_SYMBOL}(x):  # block start"
        lines[CAND_START] = "    return x * 2"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if git:
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=t@t.t",
                "-c",
                "user.name=t",
                "commit",
                "-qm",
                "init",
            ],
            cwd=repo,
            check=True,
        )
    return repo


def write_index(tmp_path: Path, repo_path: Path) -> Path:
    """A minimal index.md carrying the Repo path line."""
    index = tmp_path / "index.md"
    index.write_text(
        f"# Spotlights Run\n\n## Repository\n- **Repo path:** {repo_path}\n",
        encoding="utf-8",
    )
    return index
