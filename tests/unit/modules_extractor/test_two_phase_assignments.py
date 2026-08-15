"""Assignment-contract orchestration tests with a fake Claude.

Same fake boundary as the tree-contract suites: patch `run_streaming_claude`
in `claude_stage`. Dispatch is on prompt content — a `"whole_repository":
true` scope marks the single Stage-3A call, a `"key"` marks a shard or a
metadata batch — because shards and batches run concurrently.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor.errors import (
    ExtractorCoverageError,
    ExtractorValidationError,
)
from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.modules_extractor.two_phase import run_two_phase_extraction
from spotlights_engine.signal_pipeline._subprocess_util import StreamingResult

_KEY_RE = re.compile(r'"key":\s*"([^"]+)"')


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _single_root_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "pkg" / "core" / "engine.py")
    _write(tmp_path / "pkg" / "core" / "runner.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "a.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "b.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s1.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s2.py")
    _write(tmp_path / "pkg" / "core" / "util" / "helper.py")
    return tmp_path


def _two_branch_repo(tmp_path: Path) -> Path:
    for rel in (
        "alpha/a1.py",
        "alpha/a2.py",
        "alpha/kv_offload/k1.py",
        "alpha/kv_offload/k2.py",
        "beta/b1.py",
        "beta/core/c1.py",
        "beta/core/c2.py",
    ):
        _write(tmp_path / rel)
    return tmp_path


def _decision(source_root: str) -> dict:
    return {
        "repository": {
            "name": "demo",
            "summary": "A demo package.",
            "source_root": source_root,
        },
        "excluded_source_paths": [],
    }


CORE_ASSIGNMENTS = {
    "assignments": {
        "pkg/core": "MODULE",
        "pkg/core/kv_offload": "PART",
        "pkg/core/scheduler": "PART",
        "pkg/core/util": "PART",
    },
    "module_decisions": {"pkg/core": {"keep_reason": None}},
}

CORE_METADATA = {
    "modules": {
        "pkg/core": {
            "description": "Core runtime.",
            "main_files": [
                {"path": "pkg/core/engine.py", "role": "Engine."},
                {"path": "pkg/core/util/helper.py", "role": "Helper."},
            ],
        }
    }
}

ALPHA_ASSIGNMENTS = {
    "assignments": {"alpha": "MODULE", "alpha/kv_offload": "PART"},
    "module_decisions": {"alpha": {"keep_reason": None}},
}

BETA_ASSIGNMENTS = {
    "assignments": {"beta": "MODULE", "beta/core": "PART"},
    "module_decisions": {"beta": {"keep_reason": None}},
}

TWO_BRANCH_METADATA = {
    "modules": {
        "alpha": {
            "description": "Alpha runtime.",
            "main_files": [{"path": "alpha/a1.py", "role": "A1."}],
        },
        "beta": {
            "description": "Beta services.",
            "main_files": [{"path": "beta/b1.py", "role": "B1."}],
        },
    }
}


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


class _AssignClaude:
    """Thread-safe fake keyed on prompt content.

    `per_key["single"]` answers the whole-repository Stage-3A call;
    `per_key[<shard key>]` and `per_key[<batch key>]` answer sharded and
    metadata calls. The last entry of a sequence repeats.
    """

    def __init__(self, decision: dict, per_key: dict) -> None:
        self.decision = decision
        self.per_key = {k: list(v) for k, v in per_key.items()}
        self.calls: list[str] = []
        self.timeouts: dict[str, int] = {}
        self._lock = threading.Lock()

    def calls_for(self, key: str) -> int:
        return self.calls.count(key)

    def __call__(self, **kwargs) -> StreamingResult:
        prompt = kwargs["prompt"]
        match = _KEY_RE.search(prompt)
        if match is not None:
            key = match.group(1)
        elif '"whole_repository": true' in prompt:
            key = "single"
        else:
            key = "01_source_root"
        with self._lock:
            self.calls.append(key)
            self.timeouts.setdefault(key, kwargs.get("timeout_s"))
            attempt = self.calls.count(key)
        if key == "01_source_root":
            return _result(self.decision, "sess-root")
        queue = self.per_key[key]
        payload = queue[min(attempt - 1, len(queue) - 1)]
        return _result(payload, f"sess-{key}-{attempt}")


def _patch(monkeypatch, claude) -> None:
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.resolve_claude_argv0",
        lambda _: ["claude"],
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.run_streaming_claude",
        claude,
    )


def _run(repo, claude, monkeypatch, *, artifacts=None, **cfg):
    _patch(monkeypatch, claude)
    cfg.setdefault("contract", "assignments")
    return run_two_phase_extraction(
        repo,
        config=ExtractorConfig(**cfg),
        on_event=None,
        artifacts_dir=artifacts,
    )


# ── Config gate ───────────────────────────────────────────────────────────


def test_assignments_contract_requires_two_phase() -> None:
    with pytest.raises(ValueError, match="two_phase"):
        ExtractorConfig(contract="assignments", two_phase=False)


# ── Single-call happy path ────────────────────────────────────────────────


def test_single_call_happy_path_and_artifacts(tmp_path, monkeypatch) -> None:
    repo = _single_root_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    claude = _AssignClaude(
        _decision("pkg"),
        {"single": [CORE_ASSIGNMENTS], "batch_00": [CORE_METADATA]},
    )
    run = _run(
        repo, claude, monkeypatch, artifacts=artifacts, enrich_sharding="single"
    )

    # One module, PART folders absorbed into its territory.
    assert [m.name for m in run.project_tree.modules] == ["core"]
    assert run.project_tree.modules[0].submodules == []
    assert run.project_tree.modules[0].description == "Core runtime."
    # A territory file under a PART folder is a legal main file.
    assert any(
        f.path == "pkg/core/util/helper.py"
        for f in run.project_tree.modules[0].main_files
    )
    # 1 source-root + 1 assignment + 1 metadata call.
    assert claude.calls_for("single") == 1
    assert claude.calls_for("batch_00") == 1
    # Accepted session is the Stage-3A call.
    assert run.invocation.session_id == "sess-single-1"

    # v2 artifact layout at the run root.
    for name in (
        "assignment_tree.json",
        "resolved_assignments.json",
        "coverage.json",
        "module_metadata.json",
        "metadata_coverage.json",
        "assignment_lints.json",
        "extractor_config.json",
        "project_tree.json",
        "tree_decisions.json",
        "tree_decisions.md",
    ):
        assert (artifacts / name).exists(), name
    coverage = json.loads((artifacts / "coverage.json").read_text())
    assert coverage["schema_version"] == "coverage.v2"
    assert coverage["missing"] == []
    resolved = json.loads((artifacts / "resolved_assignments.json").read_text())
    assert resolved["owners"]["pkg/core/util"] == "pkg/core"
    assert resolved["origins"]["pkg/core"] == "top_level_anchor"
    decisions = json.loads((artifacts / "tree_decisions.json").read_text())
    assert decisions["schema_version"] == "tree_decisions.v2"
    assert decisions["contract"] == "assignments"
    # No normalization artifact in assignment mode (no label rewrite exists).
    assert not (artifacts / "03_enrich" / "attempt_01" / "normalization.json").exists()
    # Metadata batch artifacts.
    assert (artifacts / "03_enrich" / "metadata" / "batch_00" / "scope.json").exists()
    assert (
        artifacts / "03_enrich" / "metadata" / "batch_00" / "module_metadata.json"
    ).exists()


# ── Repair classes ────────────────────────────────────────────────────────


def test_missing_label_spends_the_coverage_repair(tmp_path, monkeypatch) -> None:
    repo = _single_root_repo(tmp_path)
    incomplete = json.loads(json.dumps(CORE_ASSIGNMENTS))
    del incomplete["assignments"]["pkg/core/util"]
    claude = _AssignClaude(
        _decision("pkg"),
        {
            "single": [incomplete, CORE_ASSIGNMENTS],
            "batch_00": [CORE_METADATA],
        },
    )
    run = _run(repo, claude, monkeypatch, enrich_sharding="single")
    assert claude.calls_for("single") == 2
    assert [m.name for m in run.project_tree.modules] == ["core"]


def test_missing_keep_reason_spends_the_validation_repair(
    tmp_path, monkeypatch
) -> None:
    repo = _single_root_repo(tmp_path)
    # kv_offload labeled MODULE below threshold without a keep_reason.
    invalid = json.loads(json.dumps(CORE_ASSIGNMENTS))
    invalid["assignments"]["pkg/core/kv_offload"] = "MODULE"
    invalid["module_decisions"]["pkg/core/kv_offload"] = {"keep_reason": None}
    claude = _AssignClaude(
        _decision("pkg"),
        {
            "single": [invalid, CORE_ASSIGNMENTS],
            "batch_00": [CORE_METADATA],
        },
    )
    run = _run(repo, claude, monkeypatch, enrich_sharding="single")
    assert claude.calls_for("single") == 2
    assert [m.name for m in run.project_tree.modules] == ["core"]


def test_coverage_exhaustion_raises_and_writes_best_effort_report(
    tmp_path, monkeypatch
) -> None:
    repo = _single_root_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    incomplete = json.loads(json.dumps(CORE_ASSIGNMENTS))
    del incomplete["assignments"]["pkg/core/util"]
    claude = _AssignClaude(_decision("pkg"), {"single": [incomplete]})
    with pytest.raises(ExtractorCoverageError) as exc:
        _run(
            repo, claude, monkeypatch,
            artifacts=artifacts, enrich_sharding="single",
        )
    assert "pkg/core/util" in exc.value.context["missing"]  # type: ignore[operator]
    assert claude.calls_for("single") == 2  # initial + one coverage repair
    decisions = json.loads((artifacts / "tree_decisions.json").read_text())
    assert decisions["schema_version"] == "tree_decisions.v2"
    nodes = {
        n["path"]: n["decision"]
        for n in [decisions["nodes"][0], *decisions["nodes"][0]["children"]]
    }
    assert nodes["pkg/core/util"] == "missing"


def test_metadata_coverage_repair_then_success(tmp_path, monkeypatch) -> None:
    repo = _single_root_repo(tmp_path)
    claude = _AssignClaude(
        _decision("pkg"),
        {
            "single": [CORE_ASSIGNMENTS],
            "batch_00": [{"modules": {}}, CORE_METADATA],
        },
    )
    run = _run(repo, claude, monkeypatch, enrich_sharding="single")
    assert claude.calls_for("batch_00") == 2
    assert run.project_tree.modules[0].description == "Core runtime."


def test_metadata_validation_repair_on_unowned_main_file(
    tmp_path, monkeypatch
) -> None:
    repo = _single_root_repo(tmp_path)
    bad = json.loads(json.dumps(CORE_METADATA))
    bad["modules"]["pkg/core"]["main_files"] = [
        {"path": "pkg/core/nope.py", "role": "Ghost."}
    ]
    claude = _AssignClaude(
        _decision("pkg"),
        {
            "single": [CORE_ASSIGNMENTS],
            "batch_00": [bad, CORE_METADATA],
        },
    )
    run = _run(repo, claude, monkeypatch, enrich_sharding="single")
    assert claude.calls_for("batch_00") == 2
    assert [m.name for m in run.project_tree.modules] == ["core"]


def test_metadata_exhaustion_report_keeps_last_parseable_batch(
    tmp_path, monkeypatch
) -> None:
    repo = _single_root_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    partial = json.loads(json.dumps(CORE_METADATA))
    partial["modules"]["pkg/core"]["description"] = "Last parseable metadata."
    partial["modules"]["pkg/core"]["main_files"] = [
        {"path": "pkg/core/missing.py", "role": "Invalid file."}
    ]
    malformed = {"not_modules": {}}
    claude = _AssignClaude(
        _decision("pkg"),
        {
            "single": [CORE_ASSIGNMENTS],
            "batch_00": [partial, malformed, malformed],
        },
    )

    with pytest.raises(ExtractorValidationError):
        _run(
            repo,
            claude,
            monkeypatch,
            artifacts=artifacts,
            enrich_sharding="single",
        )

    assert claude.calls_for("batch_00") == 3
    report = json.loads((artifacts / "tree_decisions.json").read_text())
    assert report["nodes"][0]["description"] == "Last parseable metadata."
    assert "main_file_not_a_source_file" in report["nodes"][0]["invalid_codes"]


# ── Sharded Stage 3A ──────────────────────────────────────────────────────


def test_two_branches_shard_and_merge(tmp_path, monkeypatch) -> None:
    repo = _two_branch_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    claude = _AssignClaude(
        _decision(""),
        {
            "alpha": [ALPHA_ASSIGNMENTS],
            "beta": [BETA_ASSIGNMENTS],
            "batch_00": [TWO_BRANCH_METADATA],
        },
    )
    run = _run(repo, claude, monkeypatch, artifacts=artifacts)

    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]
    assert claude.calls_for("alpha") == 1
    assert claude.calls_for("beta") == 1
    assert claude.calls_for("batch_00") == 1

    # Fragments, branch prechecks, and the merged union all persisted.
    enrich = artifacts / "03_enrich"
    assert (enrich / "alpha" / "assignment_fragment.json").exists()
    assert (enrich / "beta" / "assignment_fragment.json").exists()
    assert (enrich / "branches" / "alpha" / "coverage.json").exists()
    merged = json.loads((enrich / "merged" / "assignment_tree.json").read_text())
    assert set(merged["assignments"]) == {
        "alpha",
        "alpha/kv_offload",
        "beta",
        "beta/core",
    }
    plan = json.loads((enrich / "shards.json").read_text())
    assert sorted(s["key"] for s in plan["shards"]) == ["alpha", "beta"]
    assert all(s["is_branch_root"] for s in plan["shards"])
    # Sessions carry both 3A shard tags and the 3B batch tag.
    sessions = json.loads((artifacts / "sessions.json").read_text())
    stages = {s["stage"] for s in sessions["sessions"]}
    assert {"03_enrich[alpha]", "03_enrich[beta]", "03_metadata[batch_00]"} <= stages
    # Primary session: the first branch's shard.
    assert run.invocation.session_id == "sess-alpha-1"


def test_single_mode_with_several_roots_still_shards(tmp_path, monkeypatch) -> None:
    repo = _two_branch_repo(tmp_path)
    claude = _AssignClaude(
        _decision(""),
        {
            "alpha": [ALPHA_ASSIGNMENTS],
            "beta": [BETA_ASSIGNMENTS],
            "batch_00": [TWO_BRANCH_METADATA],
        },
    )
    run = _run(repo, claude, monkeypatch, enrich_sharding="single")
    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]
    assert claude.calls_for("alpha") == 1
    assert claude.calls_for("beta") == 1


def test_only_failing_shard_is_repaired(tmp_path, monkeypatch) -> None:
    repo = _two_branch_repo(tmp_path)
    incomplete_alpha = {
        "assignments": {"alpha": "MODULE"},
        "module_decisions": {"alpha": {"keep_reason": None}},
    }
    claude = _AssignClaude(
        _decision(""),
        {
            "alpha": [incomplete_alpha, ALPHA_ASSIGNMENTS],
            "beta": [BETA_ASSIGNMENTS],
            "batch_00": [TWO_BRANCH_METADATA],
        },
    )
    run = _run(repo, claude, monkeypatch)
    assert claude.calls_for("alpha") == 2
    assert claude.calls_for("beta") == 1
    assert [m.name for m in run.project_tree.modules] == ["alpha", "beta"]


def test_exhausted_assignment_shard_writes_its_last_parseable_fragment(
    tmp_path, monkeypatch
) -> None:
    repo = _two_branch_repo(tmp_path / "repo")
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    invalid_alpha = {
        "assignments": {"alpha": "MODULE", "alpha/kv_offload": "MODULE"},
        "module_decisions": {
            "alpha": {"keep_reason": None},
            "alpha/kv_offload": {"keep_reason": None},
        },
    }
    malformed = {"not_assignments": {}}
    claude = _AssignClaude(
        _decision(""),
        {
            "alpha": [invalid_alpha, malformed, malformed],
            "beta": [BETA_ASSIGNMENTS],
        },
    )

    with pytest.raises(ExtractorValidationError):
        _run(repo, claude, monkeypatch, artifacts=artifacts)

    assert claude.calls_for("alpha") == 3
    report = json.loads((artifacts / "tree_decisions.json").read_text())
    alpha = report["nodes"][0]
    kv = alpha["children"][0]
    assert alpha["path"] == "alpha"
    assert kv["path"] == "alpha/kv_offload"
    assert kv["decision"] == "module_leaf"
    assert "keep_reason_missing" in kv["invalid_codes"]


def test_top_level_qn_collision_fails_before_any_stage3_call(
    tmp_path, monkeypatch
) -> None:
    _write(tmp_path / "repo" / "top-x" / "a.py")
    _write(tmp_path / "repo" / "top_x" / "b.py")
    repo = tmp_path / "repo"
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    claude = _AssignClaude(_decision(""), {})
    with pytest.raises(ExtractorValidationError, match="collide"):
        _run(repo, claude, monkeypatch, artifacts=artifacts)
    # Only the Stage-1 call ran; no Stage-3 Claude spend.
    assert claude.calls == ["01_source_root"]
    # Best-effort report with the root issue.
    decisions = json.loads((artifacts / "tree_decisions.json").read_text())
    assert decisions["schema_version"] == "tree_decisions.v2"
    assert any(
        i["code"] == "top_level_qn_collision" for i in decisions["root_issues"]
    )
    assert {n["path"]: n["decision"] for n in decisions["nodes"]} == {
        "top-x": "missing",
        "top_x": "missing",
    }
