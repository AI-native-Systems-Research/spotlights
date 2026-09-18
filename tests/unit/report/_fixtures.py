"""Fixture builders for the report renderer tests.

`report.render.load()` reads raw JSON off disk, so these write plain dicts
rather than going through the pydantic schemas the way
`tests/unit/results_renderer/_fixtures.py` does. The renderer never sees a
schema object, and a fixture built from one would pin the wrong contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def minimal_result() -> dict[str, Any]:
    """Two candidates in one module: one high impact with a location, one low.

    The low-impact one carries a `<script>` tag in its description, so any test
    that renders this fixture exercises escaping.
    """
    return {
        "module_runs": {"demo/mod": {"status": "SUCCEEDED"}},
        "report": {
            "context": {"objective": "cut median TTFT", "workload_hints": ["batch=8"]},
            "run": {
                "run_id": "run-min-0001",
                "pipeline": "deterministic",
                "started_at": "2026-09-18T10:00:00Z",
                "finished_at": "2026-09-18T10:05:00Z",
            },
            "candidates": [
                {
                    "id": "cand-demo-0001",
                    "module_qualified_name": "demo/mod",
                    "description": "Fuse the two elementwise passes.",
                    "current_approach": "two separate kernels",
                    "estimated_impact": "high",
                    "estimated_impact_explanation": "removes one full pass over the tensor",
                    "origin": "codebase",
                    "locations": [
                        {
                            "symbol": "fuse_pass",
                            "file": "src/demo/mod.py",
                            "line_start": 10,
                            "line_end": 42,
                        }
                    ],
                    "proposals": [{"author": "codex", "source": "codebase"}],
                },
                {
                    "id": "cand-demo-0002",
                    "module_qualified_name": "demo/mod",
                    "description": "Cache the <script>alert(1)</script> lookup table.",
                    "estimated_impact": "low",
                    "locations": [],
                    "proposals": [],
                },
            ],
            "issues": [],
        },
    }


def minimal_ranking() -> dict[str, Any]:
    """A `sorted/` overlay whose rank order contradicts the impact order.

    cand-0002 ranks first while carrying the *worse* overlay impact (`low`
    against cand-0001's `medium`), so the unranked impact sort would put
    cand-0001 first. Only a real rank sort produces [0002, 0001] — that is what
    makes the ordering assertion in `test_load.py` discriminating. Giving the
    rank-1 candidate the better impact would let both sort branches agree and
    the assertion would pass even with the rank sort disabled.

    cand-0001's overlay impact (`medium`) also differs from its `result.json`
    impact (`high`), pinning overlay-wins-over-result precedence.
    """
    return {
        "run_id": "run-min-0001",
        "objective": "cut median TTFT",
        "method": "weighted-impact-v1",
        "candidates": [
            {
                "id": "cand-demo-0002",
                "rank": 1,
                "score": 0.91,
                "impact": "low",
                "symbol": "lookup_table",
                "rationale": "hot path, cheap change",
            },
            {
                "id": "cand-demo-0001",
                "rank": 2,
                "score": 0.44,
                "impact": "medium",
                "symbol": "fuse_pass",
                "rationale": "bigger change, less certain",
            },
        ],
    }


def write_run(root: Path, *, ranked: bool = False) -> Path:
    """Write a run directory under `root` and return its path.

    With `ranked=False` there is no `run_manifest.json`, no `sorted/`, no
    `evolve/` and no `apply/` — only the one input `load()` requires.
    """
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps(minimal_result()), encoding="utf-8")
    if ranked:
        sorted_dir = run / "sorted"
        sorted_dir.mkdir(exist_ok=True)
        (sorted_dir / "sorted_candidates.json").write_text(
            json.dumps(minimal_ranking()), encoding="utf-8"
        )
    return run
