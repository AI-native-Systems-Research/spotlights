"""End-to-end orchestrator tests using a FakeAgentRunner.

These tests cover the orchestration of all Stage-1 components and the
§14 scenarios in the implementation plan; the real Claude / Codex CLIs
are never invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from spotlights_engine.candidate_discovery.agents import AgentInvocation
from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    DiscoveryResult,
    discover,
)
from spotlights_engine.candidate_discovery.errors import (
    DiscoveryMutationError,
    DiscoverySetupError,
    DiscoveryValidationError,
)
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository


# ----- Test scaffolding -----------------------------------------------------


class FakeAgentRunner:
    """Stand-in for a real `AgentRunner`. Does NOT use subprocess.

    Scripted via `responses`: each element is either a JSON string (the
    `last_message.json` body to write) or a callable taking `(iter_dir,
    prompt)` returning a string (so a test can mutate the repo, raise, etc.).
    """

    def __init__(
        self,
        name: str,
        responses: list,
        repo_writer: Callable[[Path], None] | None = None,
    ) -> None:
        self.name = name
        self._responses = list(responses)
        self._call_index = 0
        self._repo_writer = repo_writer

    def invoke(self, *, prompt, iter_dir: Path, schema_path: Path) -> AgentInvocation:
        if self._call_index >= len(self._responses):
            raise AssertionError(
                f"FakeAgentRunner({self.name}) ran out of scripted responses"
            )
        entry = self._responses[self._call_index]
        self._call_index += 1

        body = entry(iter_dir, prompt) if callable(entry) else entry
        (iter_dir / "last_message.json").write_text(body, encoding="utf-8")
        (iter_dir / "raw_stdout.log").write_bytes(b"")
        (iter_dir / "raw_stderr.log").write_bytes(b"")

        if self._repo_writer is not None:
            self._repo_writer(iter_dir)

        return AgentInvocation(
            session_id=f"fake-{self.name}-{self._call_index - 1}",
            duration_s=0.001,
            cost_usd=0.01,
            input_tokens=100,
            output_tokens=200,
        )

    def parse_last_message(self, iter_dir: Path) -> str:
        path = iter_dir / "last_message.json"
        from spotlights_engine.candidate_discovery.agents import _SchemaParseError

        if not path.exists():
            raise _SchemaParseError("fake last_message.json missing")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise _SchemaParseError("fake last_message.json empty")
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            raise _SchemaParseError(f"fake last_message.json not JSON: {e}") from e
        return text


def _module(path: str = "src/v1/foo") -> Module:
    return Module(
        name="foo",
        path=path,
        description="hot path",
        main_files=[File(path="src/v1/foo/x.py", role="entry")],
    )


def _project_tree() -> ProjectTree:
    """Tree shaped so that qualified name `v1/foo` resolves to a `foo` leaf.

    The `module_qualified_name` is the module path relative to `source_root`
    (`"src"` here), so a leaf `foo` at `src/v1/foo` has qualified name
    `v1/foo`. Child paths must nest under their parent's path.
    """
    return ProjectTree(
        repository=Repository(name="demo", summary="demo repo", source_root="src"),
        modules=[
            Module(
                name="v1",
                path="src/v1",
                description="v1 namespace",
                submodules=[_module()],
            )
        ],
    )


def _seed_module_files(repo: Path) -> None:
    (repo / "src" / "v1" / "foo").mkdir(parents=True, exist_ok=True)
    for name in ("x.py", "y.py", "z.py"):
        (repo / "src" / "v1" / "foo" / name).write_text(
            "\n".join(f"line {i}" for i in range(1, 101)) + "\n"
        )


def _make_input() -> CandidateDiscoveryInput:
    return CandidateDiscoveryInput(
        project_tree=_project_tree(),
        module_qualified_name="v1/foo",
        context=SpotlightContext(objective="reduce latency"),
    )


def _make_config(repo: Path, artifacts: Path, num_reviews: int = 3) -> DiscoveryConfig:
    return DiscoveryConfig(
        repo_path=repo,
        artifacts_dir=artifacts,
        num_review_iterations=num_reviews,
    )


def _run(repo: Path, artifacts: Path, num_reviews: int = 3) -> DiscoveryResult:
    return discover(_make_input(), config=_make_config(repo, artifacts, num_reviews))


def _cands(ids_and_files, qn: str = "v1/foo") -> str:
    return json.dumps(
        {
            "module_qualified_name": qn,
            "candidates": [
                {
                    "id": cid,
                    "file": f,
                    "line_start": 1,
                    "line_end": 10,
                    "symbol": f"module.{cid}",
                    "kind": "function",
                    "description": f"work for {cid}",
                    "current_approach": "linear scan",
                    "evolve_rationale": f"hot {cid}; oracle is test_{cid}.py",
                    "estimated_impact": "medium",
                    "estimated_impact_explanation": f"cuts {cid}_latency_us; loop dominates",
                }
                for (cid, f) in ids_and_files
            ],
        }
    )


@pytest.fixture
def repo_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _seed_module_files(repo)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return repo, artifacts


def _install_runners(
    monkeypatch,
    claude: FakeAgentRunner,
    codex: FakeAgentRunner,
) -> None:
    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.orchestrator.ClaudeRunner",
        lambda _config: claude,
    )
    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.orchestrator.CodexRunner",
        lambda _config: codex,
    )


# ----- §14 scenarios --------------------------------------------------------


def test_happy_path_four_iterations(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0003", "src/v1/foo/z.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/y.py")]),
            _cands(
                [
                    ("cand-0001", "src/v1/foo/x.py"),
                    ("cand-0003", "src/v1/foo/z.py"),
                    ("cand-0004", "src/v1/foo/x.py"),
                ]
            ),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    result = _run(repo, artifacts, num_reviews=3)

    assert isinstance(result, DiscoveryResult)
    assert len(result.iterations) == 4
    assert [t.n for t in result.iterations] == [0, 1, 2, 3]
    assert [t.agent for t in result.iterations] == [
        "claude_code",
        "codex",
        "claude_code",
        "codex",
    ]

    final = artifacts / "candidates.json"
    assert final.exists()
    final_data = json.loads(final.read_text())
    final_ids = sorted(c["id"] for c in final_data["candidates"])
    # The agent emits bare `cand-NNNN`; persisted schema ids carry the module
    # segment (default slug of `v1/foo` -> `v1_foo`).
    assert final_ids == ["cand-v1_foo-0001", "cand-v1_foo-0003", "cand-v1_foo-0004"]

    last_iter_dir = artifacts / "candidate_discovery" / "iter_3_codex"
    assert (last_iter_dir / "candidates.json").read_bytes() == final.read_bytes()

    lines = (artifacts / "candidate_discovery" / "iterations.jsonl").read_text().splitlines()
    assert len(lines) == 4

    for n in (1, 2, 3):
        agent = "codex" if (n - 1) % 2 == 0 else "claude_code"
        assert (artifacts / "candidate_discovery" / f"iter_{n}_{agent}" / "diff_from_prev.md").exists()
    assert not (
        artifacts / "candidate_discovery" / "iter_0_bootstrap" / "diff_from_prev.md"
    ).exists()


def test_schema_parse_retry_succeeds(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            "not-json-on-first-attempt",  # forces a retry
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/y.py")]),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    result = _run(repo, artifacts, num_reviews=1)

    assert result.iterations[1].schema_retries == 1


def test_schema_parse_retry_fails_twice(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=["bad-1", "bad-2"],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=2)
    with pytest.raises(DiscoveryValidationError) as exc:
        discover(_make_input(), config=cfg)
    assert exc.value.context["iteration"] == 1


def test_mutation_guard_fires(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts

    def _write_into_repo(_iter_dir: Path) -> None:
        (repo / "src" / "v1" / "foo" / "x.py").write_text("mutated\n")

    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/v1/foo/x.py")])],
        repo_writer=_write_into_repo,
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoveryMutationError) as exc:
        discover(_make_input(), config=cfg)
    # Plan §6: raised iterations carry iteration/agent in context.
    assert exc.value.context["iteration"] == 0
    assert exc.value.context["agent"] == "claude_code"


def test_qualified_name_mismatch(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/v1/foo/x.py")], qn="wrong/name")],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoveryValidationError, match="qualified-name"):
        discover(_make_input(), config=cfg)


def test_containment_drop(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    (repo / "src" / "bar").mkdir()
    (repo / "src" / "bar" / "y.py").write_text("hello\n")

    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands(
                [
                    ("cand-0001", "src/v1/foo/x.py"),
                    ("cand-0002", "src/bar/y.py"),
                ]
            )
        ],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    result = _run(repo, artifacts, num_reviews=0)
    assert result.iterations[0].dropped_outside_module == 1
    assert [c.id for c in result.candidates.candidates] == ["cand-v1_foo-0001"]


def test_all_candidates_dropped(repo_artifacts, monkeypatch):
    """Per architecture, a discovery pass that ends with zero surviving
    candidates returns an empty `Candidates` (the manager will mark the
    module run `SKIPPED`); it is no longer fatal."""
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/v1/foo/missing.py")])],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    result = _run(repo, artifacts, num_reviews=0)
    assert result.candidates.candidates == []
    assert result.candidates.module_qualified_name == "v1/foo"


def test_id_monotonicity_violation(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0005", "src/v1/foo/y.py")]),
            _cands(
                [
                    ("cand-0001", "src/v1/foo/x.py"),
                    ("cand-0005", "src/v1/foo/y.py"),
                ]
            ),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands(
                [
                    ("cand-0001", "src/v1/foo/x.py"),
                    ("cand-0005", "src/v1/foo/y.py"),
                    ("cand-0003", "src/v1/foo/z.py"),  # 0003 <= 0005 max_seen — violation
                ]
            ),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=2)
    with pytest.raises(DiscoveryValidationError, match="strictly greater"):
        discover(_make_input(), config=cfg)


def test_dropped_candidate_can_be_readded(repo_artifacts, monkeypatch):
    """A candidate dropped by a later pass is fed back and can be re-added at
    its original id and file without tripping id-integrity."""
    repo, artifacts = repo_artifacts
    # bootstrap (claude, iter0): {0001, 0002}
    # review 1 (codex, iter1): drops 0002 -> {0001}
    # review 2 (claude, iter2): re-adds 0002 at its original file -> {0001, 0002}
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/y.py")]),
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/y.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),  # drops 0002
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    result = _run(repo, artifacts, num_reviews=2)

    final_ids = sorted(c.id for c in result.candidates.candidates)
    assert final_ids == ["cand-v1_foo-0001", "cand-v1_foo-0002"]

    # The dropped candidate must have been offered back to iter2's review prompt
    # (bare id space, like prev_json).
    iter2_prompt = (
        artifacts / "candidate_discovery" / "iter_2_claude_code" / "prompt.md"
    ).read_text(encoding="utf-8")
    assert "Previously removed candidates" in iter2_prompt
    assert "cand-0002" in iter2_prompt.split("Previously removed candidates", 1)[1]


def test_readd_with_changed_file_raises(repo_artifacts, monkeypatch):
    """Re-adding a previously dropped id but pointing at a different file is a
    collision, not a re-add, and must raise."""
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/y.py")]),
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/z.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),  # drops 0002
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=2)
    with pytest.raises(DiscoveryValidationError, match="re-added id changed file") as exc:
        discover(_make_input(), config=cfg)
    assert exc.value.context["id"] == "cand-v1_foo-0002"
    assert exc.value.context["prev_file"] == "src/v1/foo/y.py"
    assert exc.value.context["new_file"] == "src/v1/foo/z.py"


def test_within_iter_duplicate_ids_raises(repo_artifacts, monkeypatch):
    """§6.7 sub-check 1: ids must be unique within a single iteration."""
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands(
                [
                    ("cand-0001", "src/v1/foo/x.py"),
                    ("cand-0001", "src/v1/foo/y.py"),
                ]
            )
        ],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoveryValidationError, match="duplicate id"):
        discover(_make_input(), config=cfg)


def test_carry_over_id_must_retain_file(repo_artifacts, monkeypatch):
    """§6.7 sub-check 2: an id reused from the previous raw iter MUST keep its file."""
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/v1/foo/y.py")]),  # same id, different file → raise
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=1)
    with pytest.raises(DiscoveryValidationError, match="carry-over") as exc:
        discover(_make_input(), config=cfg)
    # Ids carry the module segment by the time the integrity check runs.
    assert exc.value.context["id"] == "cand-v1_foo-0001"
    assert exc.value.context["prev_file"] == "src/v1/foo/x.py"
    assert exc.value.context["new_file"] == "src/v1/foo/y.py"


def test_pre_existing_run_dir_raises(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    (artifacts / "candidate_discovery").mkdir()
    claude = FakeAgentRunner("claude_code", responses=[])
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoverySetupError, match="pre-existing"):
        discover(_make_input(), config=cfg)


def test_artifacts_dir_inside_repo_raises(repo_artifacts, monkeypatch):
    repo, _ = repo_artifacts
    inside = repo / "artifacts"
    inside.mkdir()
    claude = FakeAgentRunner("claude_code", responses=[])
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, inside, num_reviews=0)
    with pytest.raises(DiscoverySetupError, match="inside repo_path"):
        discover(_make_input(), config=cfg)


def test_schema_size_ceiling_raises_setup_error(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner("claude_code", responses=[])
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)
    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.orchestrator._MAX_SCHEMA_BYTES", 1
    )

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoverySetupError, match="--json-schema argv"):
        discover(_make_input(), config=cfg)
    # The check fires *before* the run dir is minted, so a retry is safe.
    assert not (artifacts / "candidate_discovery").exists()


def test_final_artifact_copy_byte_identical(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/y.py")]),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=1)
    discover(_make_input(), config=cfg)

    final = (artifacts / "candidates.json").read_bytes()
    iter_final = (
        artifacts / "candidate_discovery" / "iter_1_codex" / "candidates.json"
    ).read_bytes()
    assert final == iter_final


def test_total_duration_starts_at_run_entry(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/v1/foo/x.py")])],
    )
    codex = FakeAgentRunner("codex", responses=[])

    ticks = iter([0.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def fake_monotonic() -> float:
        return next(ticks)

    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.orchestrator.time.monotonic",
        fake_monotonic,
    )

    def make_claude(_config):
        fake_monotonic()
        return claude

    def make_codex(_config):
        fake_monotonic()
        return codex

    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.orchestrator.ClaudeRunner",
        make_claude,
    )
    monkeypatch.setattr(
        "spotlights_engine.candidate_discovery.orchestrator.CodexRunner",
        make_codex,
    )

    result = discover(_make_input(), config=_make_config(repo, artifacts, num_reviews=0))

    assert result.iterations[0].duration_s == 1.0
    assert result.total_duration_s == 9.0


# ----- §7 repo_context orchestration tests --------------------------------


def _make_config_with_context(
    repo: Path,
    artifacts: Path,
    repo_context_markdown: str | None,
    num_reviews: int = 1,
) -> DiscoveryConfig:
    return DiscoveryConfig(
        repo_path=repo,
        artifacts_dir=artifacts,
        repo_context_markdown=repo_context_markdown,
        num_review_iterations=num_reviews,
    )


def test_repo_context_persisted_when_supplied(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/v1/foo/x.py")])],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    ctx = "## Tests\n\n`pytest -q`\n"
    discover(_make_input(), config=_make_config_with_context(repo, artifacts, ctx, num_reviews=0))

    saved = artifacts / "candidate_discovery" / "repo_context.md"
    assert saved.is_file()
    assert saved.read_text(encoding="utf-8") == ctx


def test_repo_context_not_persisted_when_none(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/v1/foo/x.py")])],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    discover(_make_input(), config=_make_config_with_context(repo, artifacts, None, num_reviews=0))

    saved = artifacts / "candidate_discovery" / "repo_context.md"
    assert not saved.exists()


def test_repo_context_reaches_every_iteration_prompt(repo_artifacts, monkeypatch):
    """The repo-context markdown must thread through bootstrap *and* every
    review iteration. The orchestrator persists the attempt prompt to
    `<iter_dir>/prompt.md`; check that every such file contains the marker.
    """
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py")]),
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0003", "src/v1/foo/z.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/v1/foo/x.py"), ("cand-0002", "src/v1/foo/y.py")]),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    ctx = "## Test commands\n\n`pytest -q tests/foo/`\n\n## Bench\n\n`make bench`\n"
    discover(_make_input(), config=_make_config_with_context(repo, artifacts, ctx, num_reviews=2))

    prompt_files = sorted(
        (artifacts / "candidate_discovery").glob("iter_*/prompt.md")
    )
    assert len(prompt_files) == 3  # bootstrap + 2 reviews
    for pf in prompt_files:
        body = pf.read_text(encoding="utf-8")
        assert ctx in body, f"missing verbatim repo context in {pf}"
