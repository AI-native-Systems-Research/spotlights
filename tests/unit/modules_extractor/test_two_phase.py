"""Two-phase orchestration tests with a fake Claude.

Fake Claude follows the same boundary as ``test_agent.py``: patch
``run_streaming_claude`` in ``claude_stage`` to return a ``StreamingResult``
whose terminal ``result`` event carries ``structured_output``.

The synthetic repo is the ``pkg/core`` layout from ``test_coverage``: two
direct files, two required nested dirs (``kv_offload``, ``scheduler``), and a
foldable single-file ``util`` leaf.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor import two_phase as two_phase_module
from spotlights_engine.modules_extractor.errors import (
    ExtractorCoverageError,
    ExtractorValidationError,
)
from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.modules_extractor.sharding import derive_enrich_shards
from spotlights_engine.modules_extractor.skeleton import build_skeleton
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    SourceRootDecision,
)
from spotlights_engine.modules_extractor.tree_report import (
    load_report_from_run_dir,
    render_markdown,
    report_json_text,
)
from spotlights_engine.modules_extractor.two_phase import (
    _merge_and_gate,
    run_two_phase_extraction,
)
from spotlights_engine.signal_pipeline._subprocess_util import StreamingResult

# ── Fixtures ──────────────────────────────────────────────────────────────


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    _write(tmp_path / "pkg" / "core" / "engine.py")
    _write(tmp_path / "pkg" / "core" / "runner.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "a.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "b.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s1.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s2.py")
    _write(tmp_path / "pkg" / "core" / "util" / "helper.py")
    return tmp_path


def _source_root_decision() -> dict:
    return {
        "repository": {
            "name": "demo",
            "summary": "A demo package.",
            "source_root": "pkg",
        },
        "excluded_source_paths": [],
    }


def _enriched_tree() -> dict:
    return {
        "modules": [
            {
                "name": "core",
                "path": "pkg/core",
                "description": "Core runtime.",
                "main_files": [
                    {"path": "pkg/core/engine.py", "role": "Engine."},
                    {"path": "pkg/core/util/helper.py", "role": "Helper (folded)."},
                ],
                "submodules": [
                    {
                        "name": "kv_offload",
                        "path": "pkg/core/kv_offload",
                        "description": "KV offloading.",
                        "main_files": [
                            {"path": "pkg/core/kv_offload/a.py", "role": "A."}
                        ],
                    },
                    {
                        "name": "scheduler",
                        "path": "pkg/core/scheduler",
                        "description": "Scheduling.",
                        "main_files": [
                            {"path": "pkg/core/scheduler/s1.py", "role": "S1."}
                        ],
                    },
                ],
            }
        ],
        "folds": [
            {
                "path": "pkg/core/util",
                "into": "pkg/core",
                "reason": "single-file helper",
                "evidence_files": ["pkg/core/util/helper.py"],
            }
        ],
    }


def _stream_result(payload: dict, *, in_tok: int = 10, out_tok: int = 20) -> StreamingResult:
    event = {
        "type": "result",
        "subtype": "success",
        "session_id": f"sess-{in_tok}-{out_tok}",
        "structured_output": payload,
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    return StreamingResult(
        stdout=(json.dumps(event) + "\n").encode("utf-8"),
        stderr=b"",
        returncode=0,
        duration_s=1.0,
    )


class _FakeClaude:
    """Serves queued StreamingResults and records prompts + stage order."""

    def __init__(self, results: list[StreamingResult]) -> None:
        self.results = results
        self.prompts: list[str] = []
        self.cwds: list = []

    def __call__(self, **kwargs) -> StreamingResult:
        self.prompts.append(kwargs["prompt"])
        self.cwds.append(kwargs.get("cwd"))
        return self.results.pop(0)


def _patch(monkeypatch, fake_claude: _FakeClaude) -> None:
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.resolve_claude_argv0",
        lambda _: ["claude"],
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.run_streaming_claude",
        fake_claude,
    )


# ── Happy path ──────────────────────────────────────────────────────────


def test_happy_path_two_sessions(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision(), in_tok=5, out_tok=6),
        _stream_result(_enriched_tree(), in_tok=10, out_tok=20),
    ])
    _patch(monkeypatch, claude)

    artifacts = tmp_path / "run"
    artifacts.mkdir()
    run = run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=artifacts
    )

    # ProjectTree assembled with the enrichment content.
    assert run.project_tree.modules[0].name == "core"
    subs = {s.name for s in run.project_tree.modules[0].submodules}
    assert subs == {"kv_offload", "scheduler"}

    # Exactly 2 Claude sessions.
    assert len(claude.prompts) == 2

    # Primary session id = accepted Stage-3 enrichment session.
    assert run.invocation.session_id == "sess-10-20"

    # Serialized bytes match ProjectTree.to_json for the same object.
    written = (artifacts / "project_tree.json").read_text(encoding="utf-8")
    out2 = tmp_path / "expected.json"
    run.project_tree.to_json(out2)
    assert written == out2.read_text(encoding="utf-8")


def test_stage_order_and_cwd(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(_enriched_tree()),
    ])
    _patch(monkeypatch, claude)
    run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=None
    )
    # Stage 1 prompt is the identify-source-root prompt; stage 3 is enrichment.
    assert "source root" in claude.prompts[0].lower()
    assert "enrich" in claude.prompts[1].lower() or "skeleton" in claude.prompts[1].lower()
    # Both Claude calls run with cwd == repo.
    assert all(c == repo for c in claude.cwds)


# ── Stage-3 repair ────────────────────────────────────────────────────────


def test_stage3_one_coverage_repair_then_success(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    # First enrichment drops kv_offload (missing required); second is complete.
    # Drop both nested submodules so core is a valid zero-child leaf but both
    # required nested dirs are missing (a pure coverage miss, not a shape error).
    incomplete = _enriched_tree()
    incomplete["modules"][0]["submodules"] = []
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(incomplete),
        _stream_result(_enriched_tree()),
    ])
    _patch(monkeypatch, claude)

    run = run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=None
    )
    assert {s.name for s in run.project_tree.modules[0].submodules} == {
        "kv_offload",
        "scheduler",
    }
    # 1 source-root + 2 enrichment attempts.
    assert len(claude.prompts) == 3
    assert "Repair" in claude.prompts[2] or "Missing required" in claude.prompts[2]


def test_stage3_coverage_failure_after_repair_raises(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    # Drop both nested submodules so core is a valid zero-child leaf but both
    # required nested dirs are missing (a pure coverage miss, not a shape error).
    incomplete = _enriched_tree()
    incomplete["modules"][0]["submodules"] = []
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(incomplete),
        _stream_result(incomplete),
    ])
    _patch(monkeypatch, claude)

    with pytest.raises(ExtractorCoverageError) as exc:
        run_two_phase_extraction(
            repo, config=ExtractorConfig(), on_event=None, artifacts_dir=None
        )
    assert "pkg/core/kv_offload" in exc.value.context["missing"]  # type: ignore[operator]


# ── Telemetry ─────────────────────────────────────────────────────────────


def test_telemetry_sums_and_selects_primary_session(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision(), in_tok=5, out_tok=6),
        _stream_result(_enriched_tree(), in_tok=10, out_tok=20),
    ])
    _patch(monkeypatch, claude)
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    run = run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=artifacts
    )
    # Input tokens summed across both Claude sessions.
    assert run.invocation.input_tokens == 15
    assert run.invocation.output_tokens == 26
    # Primary session = accepted enrichment session.
    assert run.invocation.session_id == "sess-10-20"
    # sessions.json lists both Claude stages. Stage-3 records are tagged with the
    # shard key (`03_enrich[pkg__core]`) since enrichment is sharded; this repo
    # has exactly one top-level branch, so exactly one shard.
    sessions = json.loads((artifacts / "sessions.json").read_text())
    stages = [s["stage"] for s in sessions["sessions"]]
    assert "01_source_root" in stages
    assert [s for s in stages if s.startswith("03_enrich")] == ["03_enrich[pkg__core]"]


# ── Derived tree-decision report ──────────────────────────────────────────


def test_tree_decisions_written_on_success(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(_enriched_tree()),
    ])
    _patch(monkeypatch, claude)
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=artifacts
    )

    data = json.loads((artifacts / "tree_decisions.json").read_text())
    assert data["schema_version"] == "tree_decisions.v1"
    decisions = {
        n["path"]: n["decision"]
        for n in [data["nodes"][0], *data["nodes"][0]["children"]]
    }
    assert decisions["pkg/core"] == "emitted_parent"
    assert decisions["pkg/core/kv_offload"] == "emitted_leaf"
    assert decisions["pkg/core/util"] == "folded"

    md = (artifacts / "tree_decisions.md").read_text()
    assert "Legend:" in md
    assert "## Problems" not in md

    # Byte-identical to rebuilding the report from the artifacts on disk.
    rebuilt = load_report_from_run_dir(artifacts)
    assert (artifacts / "tree_decisions.json").read_text() == report_json_text(rebuilt)
    assert md == render_markdown(rebuilt)


def test_tree_decisions_best_effort_on_stage3_failure(tmp_path, monkeypatch) -> None:
    """A failed run still gets the visualization — that is when it helps most."""
    repo = _repo(tmp_path)
    incomplete = _enriched_tree()
    incomplete["modules"][0]["submodules"] = []
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(incomplete),
        _stream_result(incomplete),
    ])
    _patch(monkeypatch, claude)
    artifacts = tmp_path / "run"
    artifacts.mkdir()

    with pytest.raises(ExtractorCoverageError):
        run_two_phase_extraction(
            repo,
            config=ExtractorConfig(enrich_sharding="single"),
            on_event=None,
            artifacts_dir=artifacts,
        )

    data = json.loads((artifacts / "tree_decisions.json").read_text())
    children = {n["path"]: n["decision"] for n in data["nodes"][0]["children"]}
    assert children["pkg/core/kv_offload"] == "missing"
    assert children["pkg/core/scheduler"] == "missing"
    md = (artifacts / "tree_decisions.md").read_text()
    assert "## Problems" in md
    assert "Missing required paths (2)" in md


def test_tree_decisions_failure_never_masks_the_original_error(
    tmp_path, monkeypatch
) -> None:
    """The best-effort path must swallow its own failure, whatever its shape.

    Rendering is not an `OSError`, so it has to be caught where the report is
    built — not left in an argument expression outside the guard, where it
    would replace the coverage failure the run actually hit.
    """
    repo = _repo(tmp_path)
    incomplete = _enriched_tree()
    incomplete["modules"][0]["submodules"] = []
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(incomplete),
        _stream_result(incomplete),
    ])
    _patch(monkeypatch, claude)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("rendering exploded")

    monkeypatch.setattr(two_phase_module, "render_markdown", _boom)
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    events: list[str] = []

    with pytest.raises(ExtractorCoverageError):
        run_two_phase_extraction(
            repo,
            config=ExtractorConfig(enrich_sharding="single"),
            on_event=events.append,
            artifacts_dir=artifacts,
        )

    assert any("tree_decisions" in e for e in events)
    # Nothing half-written: the JSON is not emitted once the render fails.
    assert not (artifacts / "tree_decisions.json").exists()
    assert not (artifacts / "tree_decisions.md").exists()


def test_tree_decisions_best_effort_on_merged_failure(tmp_path) -> None:
    """The merged-tree failure branch (`_merge_and_gate`) writes it too.

    Called directly: a merged tree that fails validation while every shard
    passed would be a `sharding.py` bug, which is exactly the case the report
    is meant to help debug.
    """
    repo = _repo(tmp_path)
    config = ExtractorConfig()
    decision = SourceRootDecision.model_validate(_source_root_decision())
    skeleton = build_skeleton(
        repo, "pkg", excluded_dirs=frozenset(), excluded_files=frozenset()
    )
    plan = derive_enrich_shards(skeleton, config)
    # `core` keeps a single emitted child (Rule 4 violation on the merged tree)
    # and folds `scheduler`, so the per-branch coverage precheck still passes.
    fragment = _enriched_tree()
    fragment["modules"][0]["submodules"] = [
        s
        for s in fragment["modules"][0]["submodules"]
        if s["name"] == "kv_offload"
    ]
    fragment["modules"][0]["main_files"].append(
        {"path": "pkg/core/scheduler/s1.py", "role": "Scheduler (folded)."}
    )
    fragment["folds"].append(
        {
            "path": "pkg/core/scheduler",
            "into": "pkg/core",
            "reason": "folded for the test",
            "evidence_files": ["pkg/core/scheduler/s1.py"],
        }
    )
    fragments = {
        s.key: EnrichedTree.model_validate(fragment) for s in plan.shards
    }
    artifacts = tmp_path / "run"
    artifacts.mkdir()

    with pytest.raises(ExtractorValidationError):
        _merge_and_gate(
            repo,
            decision.repository,
            skeleton,
            plan,
            fragments,
            base=artifacts,
            decision=decision,
            on_event=None,
        )

    data = json.loads((artifacts / "tree_decisions.json").read_text())
    children = {n["path"]: n["decision"] for n in data["nodes"][0]["children"]}
    assert children["pkg/core/scheduler"] == "folded"
    assert (artifacts / "tree_decisions.md").read_text().startswith(
        "# Module extractor"
    )
