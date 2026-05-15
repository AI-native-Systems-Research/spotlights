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
from spotlights_engine.schemas.modules import File, Module


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


def _module(path: str = "src/foo") -> Module:
    return Module(
        name="foo",
        path=path,
        description="hot path",
        main_files=[File(path="src/foo/x.py", role="entry")],
    )


def _seed_module_files(repo: Path) -> None:
    (repo / "src" / "foo").mkdir(parents=True, exist_ok=True)
    for name in ("x.py", "y.py", "z.py"):
        (repo / "src" / "foo" / name).write_text(
            "\n".join(f"line {i}" for i in range(1, 101)) + "\n"
        )


def _make_config(repo: Path, artifacts: Path, num_reviews: int = 3) -> DiscoveryConfig:
    return DiscoveryConfig(
        repo_path=repo,
        module_qualified_name="v1/foo",
        module=_module(),
        artifacts_dir=artifacts,
        num_review_iterations=num_reviews,
    )


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
                    "rationale": f"hot {cid}",
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
            _cands([("cand-0001", "src/foo/x.py")]),
            _cands([("cand-0001", "src/foo/x.py"), ("cand-0003", "src/foo/z.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/foo/x.py"), ("cand-0002", "src/foo/y.py")]),
            _cands(
                [
                    ("cand-0001", "src/foo/x.py"),
                    ("cand-0003", "src/foo/z.py"),
                    ("cand-0004", "src/foo/x.py"),
                ]
            ),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=3)
    result = discover(cfg)

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
    assert final_ids == ["cand-0001", "cand-0003", "cand-0004"]

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
            _cands([("cand-0001", "src/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            "not-json-on-first-attempt",  # forces a retry
            _cands([("cand-0001", "src/foo/x.py"), ("cand-0002", "src/foo/y.py")]),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=1)
    result = discover(cfg)

    assert result.iterations[1].schema_retries == 1


def test_schema_parse_retry_fails_twice(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=["bad-1", "bad-2"],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=2)
    with pytest.raises(DiscoveryValidationError) as exc:
        discover(cfg)
    assert exc.value.context["iteration"] == 1


def test_mutation_guard_fires(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts

    def _write_into_repo(_iter_dir: Path) -> None:
        (repo / "src" / "foo" / "x.py").write_text("mutated\n")

    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/foo/x.py")])],
        repo_writer=_write_into_repo,
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoveryMutationError) as exc:
        discover(cfg)
    # Plan §6: raised iterations carry iteration/agent in context.
    assert exc.value.context["iteration"] == 0
    assert exc.value.context["agent"] == "claude_code"


def test_qualified_name_mismatch(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/foo/x.py")], qn="wrong/name")],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoveryValidationError, match="qualified-name"):
        discover(cfg)


def test_containment_drop(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    (repo / "src" / "bar").mkdir()
    (repo / "src" / "bar" / "y.py").write_text("hello\n")

    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands(
                [
                    ("cand-0001", "src/foo/x.py"),
                    ("cand-0002", "src/bar/y.py"),
                ]
            )
        ],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    result = discover(cfg)
    assert result.iterations[0].dropped_outside_module == 1
    assert [c.id for c in result.candidates.candidates] == ["cand-0001"]


def test_all_candidates_dropped(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/foo/missing.py")])],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoveryValidationError, match="post-drop"):
        discover(cfg)


def test_id_monotonicity_violation(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/foo/x.py"), ("cand-0005", "src/foo/y.py")]),
            _cands(
                [
                    ("cand-0001", "src/foo/x.py"),
                    ("cand-0005", "src/foo/y.py"),
                ]
            ),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands(
                [
                    ("cand-0001", "src/foo/x.py"),
                    ("cand-0005", "src/foo/y.py"),
                    ("cand-0003", "src/foo/z.py"),  # 0003 <= 0005 max_seen — violation
                ]
            ),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=2)
    with pytest.raises(DiscoveryValidationError, match="strictly greater"):
        discover(cfg)


def test_within_iter_duplicate_ids_raises(repo_artifacts, monkeypatch):
    """§6.7 sub-check 1: ids must be unique within a single iteration."""
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands(
                [
                    ("cand-0001", "src/foo/x.py"),
                    ("cand-0001", "src/foo/y.py"),
                ]
            )
        ],
    )
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoveryValidationError, match="duplicate id"):
        discover(cfg)


def test_carry_over_id_must_retain_file(repo_artifacts, monkeypatch):
    """§6.7 sub-check 2: an id reused from the previous raw iter MUST keep its file."""
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/foo/y.py")]),  # same id, different file → raise
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=1)
    with pytest.raises(DiscoveryValidationError, match="carry-over") as exc:
        discover(cfg)
    assert exc.value.context["id"] == "cand-0001"
    assert exc.value.context["prev_file"] == "src/foo/x.py"
    assert exc.value.context["new_file"] == "src/foo/y.py"


def test_pre_existing_run_dir_raises(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    (artifacts / "candidate_discovery").mkdir()
    claude = FakeAgentRunner("claude_code", responses=[])
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=0)
    with pytest.raises(DiscoverySetupError, match="pre-existing"):
        discover(cfg)


def test_artifacts_dir_inside_repo_raises(repo_artifacts, monkeypatch):
    repo, _ = repo_artifacts
    inside = repo / "artifacts"
    inside.mkdir()
    claude = FakeAgentRunner("claude_code", responses=[])
    codex = FakeAgentRunner("codex", responses=[])
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, inside, num_reviews=0)
    with pytest.raises(DiscoverySetupError, match="inside repo_path"):
        discover(cfg)


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
        discover(cfg)
    # The check fires *before* the run dir is minted, so a retry is safe.
    assert not (artifacts / "candidate_discovery").exists()


def test_final_artifact_copy_byte_identical(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[
            _cands([("cand-0001", "src/foo/x.py")]),
        ],
    )
    codex = FakeAgentRunner(
        "codex",
        responses=[
            _cands([("cand-0001", "src/foo/x.py"), ("cand-0002", "src/foo/y.py")]),
        ],
    )
    _install_runners(monkeypatch, claude, codex)

    cfg = _make_config(repo, artifacts, num_reviews=1)
    discover(cfg)

    final = (artifacts / "candidates.json").read_bytes()
    iter_final = (
        artifacts / "candidate_discovery" / "iter_1_codex" / "candidates.json"
    ).read_bytes()
    assert final == iter_final


def test_total_duration_starts_at_run_entry(repo_artifacts, monkeypatch):
    repo, artifacts = repo_artifacts
    claude = FakeAgentRunner(
        "claude_code",
        responses=[_cands([("cand-0001", "src/foo/x.py")])],
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

    result = discover(_make_config(repo, artifacts, num_reviews=0))

    assert result.iterations[0].duration_s == 1.0
    assert result.total_duration_s == 9.0
