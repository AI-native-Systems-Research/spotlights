"""Sharded Stage-3/4 orchestration: executor bounds, repair, artifacts, review.

The fake Claude is keyed by *prompt content* rather than by call order, because
shards run concurrently and completion order is not deterministic. Each shard's
`SCOPE (data)` block carries its `key`, which is what the fake dispatches on.

The synthetic repos are:

- `_two_branch_repo` — `alpha/` and `beta/` at the repo root (`source_root=""`),
  so derivation yields two independent top-level shards.
- `_spine_repo` — one branch that splits into a spine plus two child sub-shards,
  which is the shape the whole plan exists to make possible.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from spotlights_engine.module_deep_research.codex_exec import (
    CodexExecResult,
    CodexExecTimeout,
)
from spotlights_engine.modules_extractor.coverage import (
    CrossArtifactError,
    compute_coverage,
    validate_enriched_tree,
)
from spotlights_engine.modules_extractor.errors import (
    ExtractorAgentError,
    ExtractorCoverageError,
    ExtractorReviewError,
)
from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.modules_extractor.sharding import (
    derive_enrich_shards,
    has_several_source_roots,
    merge_fragments,
)
from spotlights_engine.modules_extractor.skeleton import build_skeleton
from spotlights_engine.modules_extractor.stage_schemas import EnrichedTree
from spotlights_engine.modules_extractor.two_phase import run_two_phase_extraction
from spotlights_engine.schemas.project import Repository
from spotlights_engine.signal_pipeline._subprocess_util import StreamingResult

_KEY_RE = re.compile(r'"key":\s*"([^"]+)"')
_SCOPE_ROOT_RE = re.compile(r'"root_path":\s*"([^"]*)"')


# ── Repos ─────────────────────────────────────────────────────────────────


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _two_branch_repo(tmp_path: Path) -> Path:
    for rel in (
        "alpha/a1.py",
        "alpha/a2.py",
        "alpha/kv_offload/k1.py",
        "alpha/kv_offload/k2.py",
        "alpha/scheduler/s1.py",
        "alpha/scheduler/s2.py",
        "alpha/util/helper.py",
        "beta/b1.py",
        "beta/b2.py",
        "beta/core/c1.py",
        "beta/core/c2.py",
        "beta/io/i1.py",
        "beta/io/i2.py",
    ):
        _write(tmp_path / rel)
    return tmp_path


def _spine_repo(tmp_path: Path) -> Path:
    for rel in (
        "pkg/root.py",
        "pkg/a/a1.py",
        "pkg/a/a2.py",
        "pkg/b/b1.py",
        "pkg/b/b2.py",
        "pkg/light1/l.py",
        "pkg/light2/m.py",
    ):
        _write(tmp_path / rel)
    return tmp_path


SPINE_CFG = dict(enrich_subshard_threshold=2, enrich_subshard_child_min=1)


def _decision(source_root: str = "") -> dict:
    return {
        "repository": {
            "name": "demo",
            "summary": "A demo repository.",
            "source_root": source_root,
        },
        "excluded_source_paths": [],
    }


# ── Fragment payloads ─────────────────────────────────────────────────────


def _file(path: str) -> dict:
    return {"path": path, "role": f"Role of {path}."}


ALPHA = {
    "modules": [
        {
            "name": "alpha",
            "path": "alpha",
            "description": "Alpha runtime.",
            "main_files": [_file("alpha/a1.py"), _file("alpha/util/helper.py")],
            "submodules": [
                {
                    "name": "kv_offload",
                    "path": "alpha/kv_offload",
                    "description": "KV offloading.",
                    "main_files": [_file("alpha/kv_offload/k1.py")],
                },
                {
                    "name": "scheduler",
                    "path": "alpha/scheduler",
                    "description": "Scheduling.",
                    "main_files": [_file("alpha/scheduler/s1.py")],
                },
            ],
        }
    ],
    "folds": [
        {
            "path": "alpha/util",
            "into": "alpha",
            "reason": "single-file helper",
            "evidence_files": ["alpha/util/helper.py"],
        }
    ],
}

BETA = {
    "modules": [
        {
            "name": "beta",
            "path": "beta",
            "description": "Beta services.",
            "main_files": [_file("beta/b1.py")],
            "submodules": [
                {
                    "name": "core",
                    "path": "beta/core",
                    "description": "Core.",
                    "main_files": [_file("beta/core/c1.py")],
                },
                {
                    "name": "io",
                    "path": "beta/io",
                    "description": "IO.",
                    "main_files": [_file("beta/io/i1.py")],
                },
            ],
        }
    ],
    "folds": [],
}

# The spine emits `pkg` plus exactly ONE light child and folds the other, which
# looks like a single-child parent to its own pruned subtree.
PKG_SPINE = {
    "modules": [
        {
            "name": "pkg",
            "path": "pkg",
            "description": "The package root.",
            "main_files": [_file("pkg/root.py"), _file("pkg/light2/m.py")],
            "submodules": [
                {
                    "name": "light1",
                    "path": "pkg/light1",
                    "description": "A light leaf.",
                    "main_files": [_file("pkg/light1/l.py")],
                }
            ],
        }
    ],
    "folds": [
        {
            "path": "pkg/light2",
            "into": "pkg",
            "reason": "single-file helper",
            "evidence_files": ["pkg/light2/m.py"],
        }
    ],
}

PKG_A = {
    "modules": [
        {
            "name": "a",
            "path": "pkg/a",
            "description": "Sub-package a.",
            "main_files": [_file("pkg/a/a1.py")],
            "submodules": [],
        }
    ],
    "folds": [],
}

PKG_B = {
    "modules": [
        {
            "name": "b",
            "path": "pkg/b",
            "description": "Sub-package b.",
            "main_files": [_file("pkg/b/b1.py")],
            "submodules": [],
        }
    ],
    "folds": [],
}

TWO_BRANCH = {"alpha": [ALPHA], "beta": [BETA]}
SPINE_SHARDS = {"pkg__spine": [PKG_SPINE], "pkg__a": [PKG_A], "pkg__b": [PKG_B]}


# ── Fakes ─────────────────────────────────────────────────────────────────


def _result(payload: dict, session: str) -> StreamingResult:
    event = {
        "type": "result",
        "subtype": "success",
        "session_id": session,
        "structured_output": payload,
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    return StreamingResult(
        stdout=(json.dumps(event) + "\n").encode("utf-8"),
        stderr=b"",
        returncode=0,
        duration_s=1.0,
    )


class _ShardClaude:
    """Thread-safe fake `run_streaming_claude`, dispatching on the shard key.

    `per_shard[key]` is the payload sequence for that shard's attempts; the last
    entry repeats, so a single-element list means "always answer this".
    """

    def __init__(self, decision: dict, per_shard: dict, *, delay: float = 0.0,
                 fail_keys: tuple[str, ...] = ()) -> None:
        self.decision = decision
        self.per_shard = {k: list(v) for k, v in per_shard.items()}
        self.delay = delay
        self.fail_keys = set(fail_keys)
        self.calls: list[str] = []
        self.timeouts: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._live = 0
        self.peak = 0

    def calls_for(self, key: str) -> int:
        return self.calls.count(key)

    def __call__(self, **kwargs) -> StreamingResult:
        prompt = kwargs["prompt"]
        match = _KEY_RE.search(prompt)
        if match is None:
            with self._lock:
                self.calls.append("01_source_root")
            return _result(self.decision, "sess-root")
        key = match.group(1)
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)
            self.calls.append(key)
            self.timeouts.setdefault(key, kwargs.get("timeout_s"))
            attempt = self.calls.count(key)
        try:
            if self.delay:
                time.sleep(self.delay)
            if key in self.fail_keys:
                return StreamingResult(
                    stdout=b"", stderr=b"boom", returncode=1, duration_s=0.1
                )
            queue = self.per_shard[key]
            payload = queue[min(attempt - 1, len(queue) - 1)]
            return _result(payload, f"sess-{key}-{attempt}")
        finally:
            with self._lock:
                self._live -= 1


class _BranchCodex:
    """Fake `CodexExecClient` factory dispatching on the review scope root."""

    def __init__(self, behavior) -> None:
        self.behavior = behavior
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def factory(self, options):
        return _BranchCodexClient(self)


class _BranchCodexClient:
    def __init__(self, parent: _BranchCodex) -> None:
        self.parent = parent

    def run(self, prompt: str, *, check: bool = True) -> CodexExecResult:
        match = _KEY_RE.search(prompt)
        key = match.group(1) if match else ""
        with self.parent._lock:
            self.parent.calls.append(key)
        return self.parent.behavior(key)


def _codex_result(report: dict, *, returncode: int = 0) -> CodexExecResult:
    return CodexExecResult(
        command=["codex"],
        returncode=returncode,
        stdout="{}\n",
        stderr="",
        final_message=json.dumps(report),
        usage=None,
        output_last_message=None,
    )


_OK = {"ok": True, "issues": []}


def _patch(monkeypatch, claude, codex) -> None:
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.resolve_claude_argv0",
        lambda _: ["claude"],
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.run_streaming_claude",
        claude,
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.two_phase.CodexExecClient",
        codex.factory,
    )


def _run(repo, claude, codex, monkeypatch, *, artifacts=None, **cfg):
    _patch(monkeypatch, claude, codex)
    return run_two_phase_extraction(
        repo,
        config=ExtractorConfig(**cfg),
        on_event=None,
        artifacts_dir=artifacts,
    )


# ── Happy paths ───────────────────────────────────────────────────────────


def test_two_branches_run_as_two_shards_and_merge(tmp_path, monkeypatch) -> None:
    repo = _two_branch_repo(tmp_path)
    claude = _ShardClaude(_decision(), TWO_BRANCH)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    run = _run(repo, claude, codex, monkeypatch)

    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]
    assert claude.calls_for("alpha") == 1
    assert claude.calls_for("beta") == 1
    # One review per top-level branch.
    assert sorted(codex.calls) == ["alpha", "beta"]
    # The extractor no longer emits dependency data; the public field stays empty.
    assert run.project_tree.modules[0].depends_on == []


def test_oversized_branch_runs_as_spine_plus_child_subshards(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path)
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    run = _run(repo, claude, codex, monkeypatch, **SPINE_CFG)

    # No single call spanned the whole branch; three scoped calls did.
    assert sorted(k for k in claude.calls if k != "01_source_root") == [
        "pkg__a",
        "pkg__b",
        "pkg__spine",
    ]
    top = run.project_tree.modules[0]
    assert top.name == "pkg"
    assert [s.name for s in top.submodules] == ["a", "b", "light1"]
    # Review stays at top-level granularity: one call for the whole branch.
    assert codex.calls == ["pkg"]


# ── Dispatch: several source folders override a `single` request ───────────


def test_several_source_folders_force_the_sharded_path_even_in_single_mode(
    tmp_path, monkeypatch
) -> None:
    """Two top-level folders + `enrich_sharding="single"` must still fan out
    per-folder, never collapse into one monolithic pass."""
    repo = _two_branch_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    # The predicate is what makes this the sharded case.
    assert has_several_source_roots(build_skeleton(repo, "")) is True

    claude = _ShardClaude(_decision(), TWO_BRANCH)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    run = _run(
        repo, claude, codex, monkeypatch,
        artifacts=artifacts, enrich_sharding="single",
    )

    # Merged to two top-level modules — not one lumped pass.
    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]
    # One scoped call per branch; no monolithic whole-repository call.
    assert claude.calls_for("alpha") == 1
    assert claude.calls_for("beta") == 1

    # The sharded path ran (`plan is not None` → shards.json exists), with two
    # branch shards and no sub-sharding.
    plan = json.loads((artifacts / "03_enrich" / "shards.json").read_text())
    assert sorted(s["key"] for s in plan["shards"]) == ["alpha", "beta"]
    assert plan["branch_roots"] == {"alpha": "alpha", "beta": "beta"}
    assert all(not s["is_subshard"] and s["depth"] == 0 for s in plan["shards"])
    assert not any("__spine" in s["key"] for s in plan["shards"])


def test_auto_mode_on_two_branches_is_unchanged(tmp_path, monkeypatch) -> None:
    """Regression guard (a): `enrich_sharding="auto"` still fans out per branch,
    exactly as before this change."""
    repo = _two_branch_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    claude = _ShardClaude(_decision(), TWO_BRANCH)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    run = _run(
        repo, claude, codex, monkeypatch,
        artifacts=artifacts, enrich_sharding="auto",
    )

    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]
    assert claude.calls_for("alpha") == 1
    assert claude.calls_for("beta") == 1
    plan = json.loads((artifacts / "03_enrich" / "shards.json").read_text())
    assert sorted(s["key"] for s in plan["shards"]) == ["alpha", "beta"]


def test_single_top_level_folder_in_single_mode_keeps_the_monolithic_pass(
    tmp_path, monkeypatch
) -> None:
    """Regression guard (b): one top-level folder + `single` still takes
    `_stage3_enrich_single` (predicate False), the monolithic whole-repository
    pass — no per-folder sharding."""
    repo = _spine_repo(tmp_path / "repo")
    # A single top-level folder — the predicate is False, so `single` is honored.
    assert has_several_source_roots(build_skeleton(repo, "")) is False

    seen: dict = {}

    def fake(**kwargs):
        prompt = kwargs["prompt"]
        if "whole_repository" in prompt:
            seen["monolithic"] = True
            return _result(_whole_pkg(), "sess-mono")
        return _result(_decision(), "sess-root")

    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _patch(monkeypatch, fake, codex)
    run = run_two_phase_extraction(
        repo,
        config=ExtractorConfig(enrich_sharding="single"),
        on_event=None,
        artifacts_dir=None,
    )
    # The monolithic (single-pass) prompt was used, not a sharded fan-out.
    assert seen.get("monolithic") is True
    assert [m.name for m in run.project_tree.modules] == ["pkg"]


# ── Executor bounds and determinism ───────────────────────────────────────


def test_peak_concurrency_respects_the_semaphore(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path)
    claude = _ShardClaude(_decision(), SPINE_SHARDS, delay=0.05)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _run(
        repo, claude, codex, monkeypatch,
        max_parallel_enrich_shards=2, **SPINE_CFG,
    )
    assert claude.peak <= 2


def test_extraction_works_inside_a_caller_owned_event_loop(tmp_path, monkeypatch) -> None:
    """`run_two_phase_extraction` is synchronous but is not always called from a
    bare thread: `spotlights_manager.orchestrator` drives step 1 from inside
    `asyncio.run(_run_async(...))`. A plain `asyncio.run` in the shard executor
    would raise "cannot be called from a running event loop" and take the whole
    manager path down.
    """
    repo = _two_branch_repo(tmp_path)
    claude = _ShardClaude(_decision(), TWO_BRANCH)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _patch(monkeypatch, claude, codex)

    async def _caller():
        return run_two_phase_extraction(
            repo, config=ExtractorConfig(), on_event=None, artifacts_dir=None
        )

    run = asyncio.run(_caller())
    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]
    # Stage 4's own executor ran too, not just Stage 3's.
    assert sorted(codex.calls) == ["alpha", "beta"]


def test_sequential_fallback_matches_concurrent_output(tmp_path, monkeypatch) -> None:
    outputs = []
    for i, parallel in enumerate((1, 4)):
        repo = _spine_repo(tmp_path / f"repo{i}")
        artifacts = tmp_path / f"run{i}"
        artifacts.mkdir(parents=True)
        claude = _ShardClaude(_decision(), SPINE_SHARDS)
        codex = _BranchCodex(lambda key: _codex_result(_OK))
        _run(
            repo, claude, codex, monkeypatch,
            artifacts=artifacts,
            max_parallel_enrich_shards=parallel,
            max_parallel_review_shards=parallel,
            **SPINE_CFG,
        )
        outputs.append((artifacts / "project_tree.json").read_text(encoding="utf-8"))
        if parallel == 1:
            assert claude.peak == 1
    assert outputs[0] == outputs[1]


def test_a_failed_shard_fails_the_run_and_never_launches_queued_siblings(
    tmp_path, monkeypatch
) -> None:
    repo = _spine_repo(tmp_path)
    # Shards run in root_path order (`pkg`, `pkg/a`, `pkg/b`); with a semaphore
    # of 1 the spine finishes, `pkg__a` fails, and `pkg__b` is still queued.
    claude = _ShardClaude(_decision(), SPINE_SHARDS, fail_keys=("pkg__a",))
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    with pytest.raises(ExtractorAgentError) as exc:
        _run(
            repo, claude, codex, monkeypatch,
            max_parallel_enrich_shards=1, **SPINE_CFG,
        )
    assert "pkg__a" in str(exc.value)
    # Cancellation guarantees only that a *queued* shard never invoked the CLI.
    assert claude.calls_for("pkg__b") == 0
    assert claude.calls_for("pkg__spine") == 1
    # The failing shard still spent its one repair before giving up.
    assert claude.calls_for("pkg__a") == 1  # a subprocess error is not repairable
    # No review ran, since Stage 3 never completed.
    assert codex.calls == []


# ── Per-shard deadline selection ──────────────────────────────────────────


def test_real_shards_get_the_per_shard_deadline(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path)
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _run(
        repo, claude, codex, monkeypatch,
        timeout_s=5400, enrich_timeout_s=1800, **SPINE_CFG,
    )
    assert claude.timeouts["pkg__spine"] == 1800
    assert claude.timeouts["pkg__a"] == 1800


def test_a_whole_skeleton_shard_keeps_the_full_timeout(tmp_path, monkeypatch) -> None:
    """A repo that degenerates to one shard must not newly time out."""
    repo = _spine_repo(tmp_path)
    claude = _ShardClaude(_decision(), {"pkg": [_whole_pkg()]})
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _run(
        repo, claude, codex, monkeypatch,
        timeout_s=5400,
        enrich_timeout_s=1800,
        enrich_subshard_threshold=1000,  # never split
    )
    assert claude.timeouts["pkg"] == 5400


def _whole_pkg() -> dict:
    return {
        "modules": [
            {
                "name": "pkg",
                "path": "pkg",
                "description": "The whole package.",
                "main_files": [_file("pkg/root.py"), _file("pkg/light2/m.py")],
                "submodules": [
                    {
                        "name": "a",
                        "path": "pkg/a",
                        "description": "Sub-package a.",
                        "main_files": [_file("pkg/a/a1.py")],
                    },
                    {
                        "name": "b",
                        "path": "pkg/b",
                        "description": "Sub-package b.",
                        "main_files": [_file("pkg/b/b1.py")],
                    },
                    {
                        "name": "light1",
                        "path": "pkg/light1",
                        "description": "A light leaf.",
                        "main_files": [_file("pkg/light1/l.py")],
                    },
                ],
            }
        ],
        "folds": [
            {
                "path": "pkg/light2",
                "into": "pkg",
                "reason": "single-file helper",
                "evidence_files": ["pkg/light2/m.py"],
            }
        ],
    }


def test_single_mode_uses_the_full_timeout_and_the_monolithic_prompt(
    tmp_path, monkeypatch
) -> None:
    repo = _spine_repo(tmp_path)
    seen: dict = {}

    def fake(**kwargs):
        prompt = kwargs["prompt"]
        if "whole_repository" in prompt:
            seen["timeout"] = kwargs["timeout_s"]
            return _result(_whole_pkg(), "sess-mono")
        return _result(_decision(), "sess-root")

    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _patch(monkeypatch, fake, codex)
    run_two_phase_extraction(
        repo,
        config=ExtractorConfig(
            enrich_sharding="single", timeout_s=5400, enrich_timeout_s=1800
        ),
        on_event=None,
        artifacts_dir=None,
    )
    assert seen["timeout"] == 5400


# ── Per-shard repair ──────────────────────────────────────────────────────


def _incomplete_spine() -> dict:
    """Spine that forgets its fold, leaving nothing missing but a bad shape…"""
    payload = json.loads(json.dumps(PKG_SPINE))
    payload["modules"][0]["submodules"] = []
    payload["folds"] = []
    payload["modules"][0]["main_files"] = [_file("pkg/root.py")]
    return payload


def _incomplete_a() -> dict:
    """`pkg/a` shard that emits nothing at all — a shard-scoped coverage miss."""
    return {"modules": [], "folds": []}


def test_only_the_failing_shard_is_repaired(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path)
    per_shard = {
        "pkg__spine": [PKG_SPINE],
        "pkg__a": [_incomplete_a(), PKG_A],
        "pkg__b": [PKG_B],
    }
    claude = _ShardClaude(_decision(), per_shard)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    run = _run(repo, claude, codex, monkeypatch, **SPINE_CFG)

    assert claude.calls_for("pkg__a") == 2  # one bounded repair
    assert claude.calls_for("pkg__b") == 1  # siblings ran once
    assert claude.calls_for("pkg__spine") == 1
    assert [s.name for s in run.project_tree.modules[0].submodules] == [
        "a",
        "b",
        "light1",
    ]


def test_shard_coverage_failure_after_repair_hard_fails(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path)
    per_shard = {
        "pkg__spine": [PKG_SPINE],
        "pkg__a": [_incomplete_a()],
        "pkg__b": [PKG_B],
    }
    claude = _ShardClaude(_decision(), per_shard)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    with pytest.raises(Exception) as exc:
        _run(repo, claude, codex, monkeypatch, **SPINE_CFG)
    assert "pkg__a" in str(exc.value)
    assert claude.calls_for("pkg__a") == 2


# ── The coverage gate is still authoritative on the merged tree ───────────


def test_missing_required_path_fails_with_the_owning_shard_named(
    tmp_path, monkeypatch
) -> None:
    repo = _two_branch_repo(tmp_path)
    dropped = json.loads(json.dumps(ALPHA))
    # Drop kv_offload entirely: `alpha` keeps ≥2 children? No — drop it and the
    # single remaining child would trip Rule 4, so drop both and fold neither.
    dropped["modules"][0]["submodules"] = []
    claude = _ShardClaude(_decision(), {"alpha": [dropped], "beta": [BETA]})
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    with pytest.raises(ExtractorCoverageError) as exc:
        _run(repo, claude, codex, monkeypatch)
    assert "alpha" in str(exc.value)
    assert "alpha/kv_offload" in exc.value.context["missing"]  # type: ignore[operator]


def test_global_coverage_reports_a_dropped_required_path(tmp_path) -> None:
    """The same omission, asserted directly against the *merged* tree — proving
    the guarantee still lives in Stage 5 regardless of sharding."""
    repo = _two_branch_repo(tmp_path)
    skeleton = build_skeleton(repo, "")
    plan = derive_enrich_shards(skeleton, ExtractorConfig())
    dropped = json.loads(json.dumps(ALPHA))
    dropped["modules"][0]["submodules"] = []
    pairs = [
        (next(s for s in plan.shards if s.key == "alpha"),
         EnrichedTree.model_validate(dropped)),
        (next(s for s in plan.shards if s.key == "beta"),
         EnrichedTree.model_validate(BETA)),
    ]
    merged = merge_fragments(pairs, skeleton=skeleton, plan=plan)
    assert "alpha/kv_offload" in compute_coverage(merged, skeleton).missing


def test_a_cross_branch_fold_is_rejected_by_the_full_validator(tmp_path) -> None:
    repo = _two_branch_repo(tmp_path)
    skeleton = build_skeleton(repo, "")
    plan = derive_enrich_shards(skeleton, ExtractorConfig())
    # `alpha` leaves `alpha/util` unaccounted (it is optional); `beta` injects a
    # fold of it into itself — a shape no shard could construct, since a shard
    # only ever sees its own subtree.
    alpha_no_fold = json.loads(json.dumps(ALPHA))
    alpha_no_fold["folds"] = []
    injected = json.loads(json.dumps(BETA))
    injected["folds"] = [
        {
            "path": "alpha/util",
            "into": "beta",
            "reason": "injected cross-branch fold",
            "evidence_files": ["alpha/util/helper.py"],
        }
    ]
    pairs = [
        (next(s for s in plan.shards if s.key == "alpha"),
         EnrichedTree.model_validate(alpha_no_fold)),
        (next(s for s in plan.shards if s.key == "beta"),
         EnrichedTree.model_validate(injected)),
    ]
    merged = merge_fragments(pairs, skeleton=skeleton, plan=plan)
    with pytest.raises(CrossArtifactError, match="not physically nested under target"):
        validate_enriched_tree(
            merged,
            repo,
            Repository(name="demo", summary="s", source_root=""),
            skeleton,
        )


# ── Per-shard artifacts ───────────────────────────────────────────────────


def test_sharded_artifacts_layout(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _run(repo, claude, codex, monkeypatch, artifacts=artifacts, **SPINE_CFG)

    enrich = artifacts / "03_enrich"
    plan = json.loads((enrich / "shards.json").read_text())
    assert sorted(s["key"] for s in plan["shards"]) == ["pkg__a", "pkg__b", "pkg__spine"]
    assert plan["branch_roots"] == {"pkg": "pkg"}
    # A split branch has NO directory of its own — only its spine + children.
    assert not (enrich / "pkg").exists()
    for key in ("pkg__spine", "pkg__a", "pkg__b"):
        assert (enrich / key / "fragment.json").exists()
        assert (enrich / key / "attempt_01" / "stream.jsonl").exists()
        assert (enrich / key / "attempt_01" / "coverage.json").exists()
    scope = json.loads((enrich / "pkg__spine" / "scope.json").read_text())
    assert scope["promoted_children"] == ["pkg/a", "pkg/b"]
    assert (enrich / "branches" / "pkg" / "coverage.json").exists()
    assert (enrich / "merged" / "coverage.json").exists()
    # sessions.json stayed valid JSON across the concurrent run.
    sessions = json.loads((artifacts / "sessions.json").read_text())
    assert {s["stage"] for s in sessions["sessions"]} >= {
        "03_enrich[pkg__spine]",
        "03_enrich[pkg__a]",
        "03_enrich[pkg__b]",
        "04_review[pkg]",
    }


def test_failed_shard_leaves_its_evidence_and_the_repair_writes_attempt_02(
    tmp_path, monkeypatch
) -> None:
    repo = _spine_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    per_shard = {
        "pkg__spine": [PKG_SPINE],
        "pkg__a": [_incomplete_a(), PKG_A],
        "pkg__b": [PKG_B],
    }
    claude = _ShardClaude(_decision(), per_shard)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    _run(repo, claude, codex, monkeypatch, artifacts=artifacts, **SPINE_CFG)

    shard_dir = artifacts / "03_enrich" / "pkg__a"
    assert (shard_dir / "attempt_01" / "last_message.json").exists()
    cov1 = json.loads((shard_dir / "attempt_01" / "coverage.json").read_text())
    assert cov1["missing"] == ["pkg/a"]
    cov2 = json.loads((shard_dir / "attempt_02" / "coverage.json").read_text())
    assert cov2["missing"] == []
    # attempt_01 was never promoted to the accepted fragment.
    fragment = json.loads((shard_dir / "fragment.json").read_text())
    assert fragment["modules"][0]["path"] == "pkg/a"


# ── Stage 4: per-shard skip, merge, revision ──────────────────────────────


def _timeout(_key: str):
    raise CodexExecTimeout(
        cmd=["codex"],
        timeout=1.0,
        stdout="partial",
        stderr="",
        final_message=None,
        duration_s=1.0,
        output_last_message=None,
    )


def test_one_branch_review_skips_while_the_other_completes(tmp_path, monkeypatch) -> None:
    repo = _two_branch_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    claude = _ShardClaude(_decision(), TWO_BRANCH)
    codex = _BranchCodex(
        lambda key: _timeout(key) if key == "alpha" else _codex_result(_OK)
    )
    run = _run(repo, claude, codex, monkeypatch, artifacts=artifacts)

    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]
    merged = json.loads((artifacts / "04_review" / "merged_review.json").read_text())
    assert merged["status"] == "completed"
    assert [s["key"] for s in merged["skipped"]] == ["alpha"]
    # `review.json` stays a plain ReviewArtifact dump — no extra keys.
    public = json.loads((artifacts / "review.json").read_text())
    assert set(public) == {"status", "report", "error"}


def test_all_branches_skipping_never_reports_a_false_ok(tmp_path, monkeypatch) -> None:
    repo = _two_branch_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    claude = _ShardClaude(_decision(), TWO_BRANCH)
    codex = _BranchCodex(_timeout)
    _run(repo, claude, codex, monkeypatch, artifacts=artifacts)

    public = json.loads((artifacts / "review.json").read_text())
    assert public["status"] == "skipped"
    assert public["report"] is None
    merged = json.loads((artifacts / "04_review" / "merged_review.json").read_text())
    assert sorted(s["key"] for s in merged["skipped"]) == ["alpha", "beta"]


def test_issues_from_two_branches_merge_into_one_sorted_report(
    tmp_path, monkeypatch
) -> None:
    repo = _two_branch_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    reports = {
        "alpha": {
            "ok": False,
            "issues": [
                {"kind": "bad_description", "path": "alpha", "detail": "terse"}
            ],
        },
        "beta": {
            "ok": False,
            "issues": [
                {"kind": "weak_main_files", "path": "beta/core", "detail": "weak"}
            ],
        },
    }
    claude = _ShardClaude(_decision(), TWO_BRANCH)
    codex = _BranchCodex(lambda key: _codex_result(reports[key]))
    _run(repo, claude, codex, monkeypatch, artifacts=artifacts)

    public = json.loads((artifacts / "review.json").read_text())
    assert public["status"] == "completed"
    assert public["report"]["ok"] is False
    assert [i["path"] for i in public["report"]["issues"]] == ["alpha", "beta/core"]
    # One bounded revision per owning shard: `alpha`'s and `beta`'s.
    assert claude.calls_for("alpha") == 2
    assert claude.calls_for("beta") == 2


def test_revision_is_scoped_to_the_shard_that_owns_the_issue(
    tmp_path, monkeypatch
) -> None:
    repo = _spine_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    report = {
        "ok": False,
        "issues": [
            {"kind": "bad_description", "path": "pkg/a", "detail": "terse"}
        ],
    }
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(report))
    _run(repo, claude, codex, monkeypatch, artifacts=artifacts, **SPINE_CFG)

    # Only `pkg__a` was re-run — not the spine, not `pkg__b`.
    assert claude.calls_for("pkg__a") == 2
    assert claude.calls_for("pkg__b") == 1
    assert claude.calls_for("pkg__spine") == 1
    attribution = json.loads(
        (artifacts / "04_review" / "pkg" / "revision" / "attribution.json").read_text()
    )
    assert attribution["issues"][0]["shard"] == "pkg__a"
    assert attribution["issues"][0]["unmapped"] == "False"


def test_unmappable_issue_path_falls_back_to_the_branch_spine(
    tmp_path, monkeypatch
) -> None:
    repo = _spine_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    report = {
        "ok": False,
        "issues": [
            {"kind": "missing_dir", "path": "not/a/real/path", "detail": "?"}
        ],
    }
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(report))
    _run(repo, claude, codex, monkeypatch, artifacts=artifacts, **SPINE_CFG)

    attribution = json.loads(
        (artifacts / "04_review" / "pkg" / "revision" / "attribution.json").read_text()
    )
    assert attribution["issues"][0]["shard"] == "pkg__spine"
    assert attribution["issues"][0]["unmapped"] == "True"
    assert claude.calls_for("pkg__spine") == 2


def test_advisory_mode_discards_an_invalid_revision(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path / "repo")
    report = {
        "ok": False,
        "issues": [{"kind": "bad_description", "path": "pkg/a", "detail": "terse"}],
    }
    per_shard = {
        "pkg__spine": [PKG_SPINE],
        "pkg__a": [PKG_A, _incomplete_a()],  # the revision is empty → invalid
        "pkg__b": [PKG_B],
    }
    claude = _ShardClaude(_decision(), per_shard)
    codex = _BranchCodex(lambda key: _codex_result(report))
    run = _run(repo, claude, codex, monkeypatch, **SPINE_CFG)

    # The known-valid fragment survived.
    assert [s.name for s in run.project_tree.modules[0].submodules] == [
        "a",
        "b",
        "light1",
    ]


def test_strict_mode_raises_on_an_invalid_revision(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path / "repo")
    report = {
        "ok": False,
        "issues": [{"kind": "bad_description", "path": "pkg/a", "detail": "terse"}],
    }
    per_shard = {
        "pkg__spine": [PKG_SPINE],
        "pkg__a": [PKG_A, _incomplete_a()],
        "pkg__b": [PKG_B],
    }
    claude = _ShardClaude(_decision(), per_shard)
    codex = _BranchCodex(lambda key: _codex_result(report))
    with pytest.raises(ExtractorReviewError):
        _run(
            repo, claude, codex, monkeypatch,
            fail_on_review_issues=True, **SPINE_CFG,
        )


def test_strict_rereview_runs_once_per_branch_on_the_revised_tree(
    tmp_path, monkeypatch
) -> None:
    repo = _spine_repo(tmp_path / "repo")
    report = {
        "ok": False,
        "issues": [{"kind": "bad_description", "path": "pkg/a", "detail": "terse"}],
    }
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(report))
    with pytest.raises(ExtractorReviewError, match="strict review"):
        _run(
            repo, claude, codex, monkeypatch,
            fail_on_review_issues=True, **SPINE_CFG,
        )
    # One review, one re-review — per branch.
    assert codex.calls == ["pkg", "pkg"]


# ── Telemetry ─────────────────────────────────────────────────────────────


def test_sessions_json_records_every_shard_invocation(tmp_path, monkeypatch) -> None:
    repo = _spine_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    per_shard = {
        "pkg__spine": [PKG_SPINE],
        "pkg__a": [_incomplete_a(), PKG_A],
        "pkg__b": [PKG_B],
    }
    claude = _ShardClaude(_decision(), per_shard)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    run = _run(repo, claude, codex, monkeypatch, artifacts=artifacts, **SPINE_CFG)

    sessions = json.loads((artifacts / "sessions.json").read_text())
    stages = [s["stage"] for s in sessions["sessions"]]
    assert "03_enrich[pkg__a]" in stages
    assert "03_enrich[pkg__a]#repair" in stages
    assert stages.count("03_enrich[pkg__spine]") == 1
    # Totals sum across every invocation: 1 source-root + 4 enrichment calls.
    assert run.invocation.input_tokens == 10 * 5


def test_primary_session_is_the_branch_owning_shard(tmp_path, monkeypatch) -> None:
    """A lexicographic-key rule would wrongly pick `pkg__a` over `pkg__spine`."""
    repo = _spine_repo(tmp_path)
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(_OK))
    run = _run(repo, claude, codex, monkeypatch, **SPINE_CFG)
    assert run.invocation.session_id == "sess-pkg__spine-1"


def test_primary_session_is_stable_across_concurrency(tmp_path, monkeypatch) -> None:
    ids = []
    for i, parallel in enumerate((1, 4)):
        repo = _spine_repo(tmp_path / f"repo{i}")
        claude = _ShardClaude(_decision(), SPINE_SHARDS, delay=0.01)
        codex = _BranchCodex(lambda key: _codex_result(_OK))
        run = _run(
            repo, claude, codex, monkeypatch,
            max_parallel_enrich_shards=parallel, **SPINE_CFG,
        )
        ids.append(run.invocation.session_id)
    assert ids[0] == ids[1] == "sess-pkg__spine-1"


def test_a_revision_of_the_primary_shard_updates_the_primary_session(
    tmp_path, monkeypatch
) -> None:
    repo = _spine_repo(tmp_path / "repo")
    report = {
        "ok": False,
        "issues": [{"kind": "bad_description", "path": "pkg", "detail": "terse"}],
    }
    claude = _ShardClaude(_decision(), SPINE_SHARDS)
    codex = _BranchCodex(lambda key: _codex_result(report))
    run = _run(repo, claude, codex, monkeypatch, **SPINE_CFG)
    assert run.invocation.session_id == "sess-pkg__spine-2"
