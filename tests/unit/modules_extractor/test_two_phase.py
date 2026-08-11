"""Two-phase orchestration tests with fake Claude and fake Codex.

Fake Claude follows the same boundary as ``test_agent.py``: patch
``run_streaming_claude`` in ``claude_stage`` to return a ``StreamingResult``
whose terminal ``result`` event carries ``structured_output``. Fake Codex
patches ``CodexExecClient`` in ``two_phase`` to return a ``CodexExecResult``
(or raise ``CodexExecTimeout`` / ``FileNotFoundError``).

The synthetic repo is the ``pkg/core`` layout from ``test_coverage``: two
direct files, two required nested dirs (``kv_offload``, ``scheduler``), and a
foldable single-file ``util`` leaf.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.module_deep_research.codex_exec import (
    CodexExecResult,
    CodexExecTimeout,
)
from spotlights_engine.modules_extractor.errors import (
    ExtractorCoverageError,
    ExtractorReviewError,
)
from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.modules_extractor.two_phase import run_two_phase_extraction
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
            "external_dependencies": [],
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
                "depends_on": [],
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


class _FakeCodex:
    """Fake CodexExecClient factory. `behavior` decides what run() does."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls: list[str] = []

    def factory(self, options):
        self.options = options
        client = _FakeCodexClient(self.behavior, self.calls, options)
        return client


class _FakeCodexClient:
    def __init__(self, behavior, calls, options):
        self.behavior = behavior
        self.calls = calls
        self.options = options

    def run(self, prompt: str, *, check: bool = True) -> CodexExecResult:
        self.calls.append(prompt)
        return self.behavior(prompt, self.options)


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


def _patch(monkeypatch, fake_claude: _FakeClaude, fake_codex: _FakeCodex) -> None:
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.resolve_claude_argv0",
        lambda _: ["claude"],
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.claude_stage.run_streaming_claude",
        fake_claude,
    )
    monkeypatch.setattr(
        "spotlights_engine.modules_extractor.two_phase.CodexExecClient",
        fake_codex.factory,
    )


# ── Happy path ──────────────────────────────────────────────────────────


def test_happy_path_two_sessions_and_clean_review(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision(), in_tok=5, out_tok=6),
        _stream_result(_enriched_tree(), in_tok=10, out_tok=20),
    ])
    codex = _FakeCodex(lambda p, o: _codex_result({"ok": True, "issues": []}))
    _patch(monkeypatch, claude, codex)

    artifacts = tmp_path / "run"
    artifacts.mkdir()
    run = run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=artifacts
    )

    # ProjectTree assembled with the enrichment content.
    assert run.project_tree.modules[0].name == "core"
    subs = {s.name for s in run.project_tree.modules[0].submodules}
    assert subs == {"kv_offload", "scheduler"}

    # Exactly 2 Claude sessions + 1 Codex review.
    assert len(claude.prompts) == 2
    assert len(codex.calls) == 1

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
    codex = _FakeCodex(lambda p, o: _codex_result({"ok": True, "issues": []}))
    _patch(monkeypatch, claude, codex)
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
    codex = _FakeCodex(lambda p, o: _codex_result({"ok": True, "issues": []}))
    _patch(monkeypatch, claude, codex)

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
    codex = _FakeCodex(lambda p, o: _codex_result({"ok": True, "issues": []}))
    _patch(monkeypatch, claude, codex)

    with pytest.raises(ExtractorCoverageError) as exc:
        run_two_phase_extraction(
            repo, config=ExtractorConfig(), on_event=None, artifacts_dir=None
        )
    assert "pkg/core/kv_offload" in exc.value.context["missing"]  # type: ignore[operator]


# ── Review skip cases ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "behavior",
    [
        lambda p, o: _codex_result({"ok": True, "issues": []}, returncode=1),
        lambda p, o: (_ for _ in ()).throw(FileNotFoundError("no codex")),
        lambda p, o: CodexExecResult(
            command=["codex"],
            returncode=0,
            stdout="",
            stderr="",
            final_message="this is not json",
            usage=None,
            output_last_message=None,
        ),
    ],
    ids=["nonzero_exit", "missing_executable", "malformed_json"],
)
def test_review_skips_are_not_fatal(tmp_path, monkeypatch, behavior) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(_enriched_tree()),
    ])
    codex = _FakeCodex(behavior)
    _patch(monkeypatch, claude, codex)
    # Advisory mode (default): a skipped review still yields a tree.
    run = run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=None
    )
    assert run.project_tree.modules[0].name == "core"


def test_review_timeout_with_partial_output_skipped(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(_enriched_tree()),
    ])

    def _timeout(p, o):
        raise CodexExecTimeout(
            cmd=["codex"],
            timeout=1.0,
            stdout="partial",
            stderr="",
            final_message=None,
            duration_s=1.0,
            output_last_message=None,
        )

    codex = _FakeCodex(_timeout)
    _patch(monkeypatch, claude, codex)
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    run = run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=artifacts
    )
    assert run.project_tree.modules[0].name == "core"
    review = json.loads((artifacts / "review.json").read_text())
    assert review["status"] == "skipped"


# ── Review revision + strict re-review ───────────────────────────────────


def test_review_revision_accepted_when_valid(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    # First enrichment: valid but description flagged. Revision: full tree.
    first = _enriched_tree()
    first["modules"][0]["description"] = "core"
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(first),
        _stream_result(_enriched_tree(), in_tok=99, out_tok=88),
    ])
    # Codex flags an issue once (non-strict → one revision).
    review_report = {
        "ok": False,
        "issues": [
            {"kind": "bad_description", "path": "pkg/core", "detail": "too terse"}
        ],
    }
    codex = _FakeCodex(lambda p, o: _codex_result(review_report))
    _patch(monkeypatch, claude, codex)

    run = run_two_phase_extraction(
        repo, config=ExtractorConfig(), on_event=None, artifacts_dir=None
    )
    # 1 source-root + 1 enrichment + 1 revision = 3 Claude calls.
    assert len(claude.prompts) == 3
    # Accepted session updates to the revision session.
    assert run.invocation.session_id == "sess-99-88"


def test_single_mode_revision_prompt_keeps_the_dependency_vocabulary(
    tmp_path, monkeypatch
) -> None:
    """The monolithic revision reuses the enrich template, whose
    `TOP_LEVEL_MODULES` block the prompt calls "the complete, authoritative
    vocabulary" of internal `depends_on` targets. Rendering it empty would tell
    the model that no internal dependency is legal at all.
    """
    repo = _repo(tmp_path)
    first = _enriched_tree()
    first["modules"][0]["description"] = "core"
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(first),
        _stream_result(_enriched_tree()),
    ])
    codex = _FakeCodex(lambda p, o: _codex_result({
        "ok": False,
        "issues": [
            {"kind": "bad_description", "path": "pkg/core", "detail": "too terse"}
        ],
    }))
    _patch(monkeypatch, claude, codex)

    run_two_phase_extraction(
        repo,
        config=ExtractorConfig(enrich_sharding="single"),
        on_event=None,
        artifacts_dir=None,
    )
    # 1 source-root + 1 monolithic enrichment + 1 revision.
    assert len(claude.prompts) == 3
    block = claude.prompts[-1].split("## TOP_LEVEL_MODULES (data)")[1].split("```")[1]
    assert '"core"' in block


def test_strict_mode_reraises_when_rereview_still_flags(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    first = _enriched_tree()
    first["modules"][0]["description"] = "core"
    claude = _FakeClaude([
        _stream_result(_source_root_decision()),
        _stream_result(first),
        _stream_result(_enriched_tree()),
    ])
    review_report = {
        "ok": False,
        "issues": [
            {"kind": "bad_description", "path": "pkg/core", "detail": "too terse"}
        ],
    }
    # Codex always flags — even the strict re-review.
    codex = _FakeCodex(lambda p, o: _codex_result(review_report))
    _patch(monkeypatch, claude, codex)

    with pytest.raises(ExtractorReviewError):
        run_two_phase_extraction(
            repo,
            config=ExtractorConfig(fail_on_review_issues=True),
            on_event=None,
            artifacts_dir=None,
        )


# ── Telemetry ─────────────────────────────────────────────────────────────


def test_telemetry_sums_and_selects_primary_session(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    claude = _FakeClaude([
        _stream_result(_source_root_decision(), in_tok=5, out_tok=6),
        _stream_result(_enriched_tree(), in_tok=10, out_tok=20),
    ])
    codex = _FakeCodex(lambda p, o: _codex_result({"ok": True, "issues": []}))
    _patch(monkeypatch, claude, codex)
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
    # sessions.json lists both Claude stages + the review. Stage-3/4 records are
    # tagged with the shard key (`03_enrich[pkg__core]`) since enrichment is
    # sharded; this repo has exactly one top-level branch, so exactly one shard.
    sessions = json.loads((artifacts / "sessions.json").read_text())
    stages = [s["stage"] for s in sessions["sessions"]]
    assert "01_source_root" in stages
    assert [s for s in stages if s.startswith("03_enrich")] == ["03_enrich[pkg__core]"]
