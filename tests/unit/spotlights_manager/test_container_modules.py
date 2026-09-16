"""`--skip-container-modules`: don't pay a discovery pass to analyze an `__init__.py`.

On the IOCR tree (30 modules) four targets are pure routing nodes:
`hrl_ocr/models/detection`, `hrl_ocr/models/recognition`, `hrl_ocr/pipeline` and
`hrl_ocr` itself, whose entire own content is a 9-18 line `__init__.py`
re-exporting submodules. Each one paid a full discovery pass (~$0.76 metered) to
analyze import plumbing, because every candidate it could produce has to live in
one of its *own* files -- `candidate_discovery.Validator` drops anything inside a
submodule -- and those files are re-exports.

The skip is coverage-safe by construction: it requires submodules, and the
content those submodules hold is analyzed as their own targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.schemas.project import File, Module
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.filters import container_module_reason
from tests.unit.spotlights_manager._fakes import (
    make_discovery_result,
    make_extractor_result,
    make_input,
    make_tree,
    patch_agent_proposals,
    patch_proposal_from_finding,
)

# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    a = tmp_path / "artifacts"
    a.mkdir()
    return a


def _write(repo: Path, rel: str, text: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _leaf(path: str, *files: str) -> Module:
    return Module(
        name=path.rsplit("/", 1)[-1],
        path=path,
        description="leaf",
        main_files=[File(path=f, role="entry") for f in files],
    )


# --- the predicate -----------------------------------------------------------


def test_plumbing_only_parent_is_a_container(repo: Path) -> None:
    """The real shape: a short `__init__.py` that only re-exports submodules."""
    _write(
        repo,
        "pkg/__init__.py",
        "from pkg.a import A\nfrom pkg.b import B\n\n__all__ = [A, B]\n",
    )
    module = Module(
        name="pkg",
        path="pkg",
        description="routing node",
        main_files=[File(path="pkg/__init__.py", role="entry")],
        submodules=[_leaf("pkg/a", "pkg/a/core.py"), _leaf("pkg/b", "pkg/b/core.py")],
    )
    reason = container_module_reason(module, repo)
    assert reason is not None
    # The reason has to name the evidence: it lands in a log line and a SKIPPED
    # status, where "why was this module never discovered" is the only question.
    assert "pkg/__init__.py" in reason
    assert "2 submodules" in reason


def test_no_own_files_at_all_is_a_container(repo: Path) -> None:
    """A namespace whose every `main_file` belongs to a submodule.

    Nothing is even readable for discovery to analyze here, so no line count
    applies -- and the tree fixture's `v1` is exactly this shape.
    """
    module = Module(
        name="pkg",
        path="pkg",
        description="namespace",
        main_files=[File(path="pkg/a/core.py", role="entry")],
        submodules=[_leaf("pkg/a", "pkg/a/core.py")],
    )
    reason = container_module_reason(module, repo)
    assert reason is not None
    assert "no files of its own" in reason


def test_a_leaf_is_never_a_container(repo: Path) -> None:
    """The opposite case: small, but possibly the whole point of the repo.

    Requiring submodules is what makes the skip coverage-safe, so a one-file
    leaf must be discovered however thin it looks.
    """
    _write(repo, "pkg/__init__.py", "x = 1\n")
    module = _leaf("pkg", "pkg/__init__.py")
    assert container_module_reason(module, repo) is None


def test_a_real_own_file_beside_the_init_disqualifies(repo: Path) -> None:
    _write(repo, "pkg/__init__.py", "from pkg.a import A\n")
    _write(repo, "pkg/engine.py", "def run():\n    return 1\n")
    module = Module(
        name="pkg",
        path="pkg",
        description="has its own code",
        main_files=[
            File(path="pkg/__init__.py", role="entry"),
            File(path="pkg/engine.py", role="entry"),
        ],
        submodules=[_leaf("pkg/a", "pkg/a/core.py")],
    )
    assert container_module_reason(module, repo) is None


def test_a_substantive_init_disqualifies(repo: Path) -> None:
    """The line bound is a guard, not the discriminator.

    An `__init__.py` carrying real implementation is rare but legal, and it is
    the one case where the filename alone would be wrong. Measured on the IOCR
    tree the containers hold 9-18 non-blank lines and the smallest non-container
    module holds 96, so any value in that gap selects the same four.
    """
    _write(repo, "pkg/__init__.py", "".join(f"x{i} = {i}\n" for i in range(60)))
    module = Module(
        name="pkg",
        path="pkg",
        description="implementation in __init__",
        main_files=[File(path="pkg/__init__.py", role="entry")],
        submodules=[_leaf("pkg/a", "pkg/a/core.py")],
    )
    assert container_module_reason(module, repo) is None


def test_blank_lines_do_not_count_toward_the_bound(repo: Path) -> None:
    _write(repo, "pkg/__init__.py", "from pkg.a import A\n" + "\n" * 200)
    module = Module(
        name="pkg",
        path="pkg",
        description="routing node, generously spaced",
        main_files=[File(path="pkg/__init__.py", role="entry")],
        submodules=[_leaf("pkg/a", "pkg/a/core.py")],
    )
    assert container_module_reason(module, repo) is not None


def test_an_unreadable_own_file_falls_back_to_discovery(repo: Path) -> None:
    """Ambiguity resolves toward spending the money.

    We cannot judge what we cannot read, and the cost of guessing wrong is a
    module silently omitted from the report -- far worse than one wasted pass.
    """
    module = Module(
        name="pkg",
        path="pkg",
        description="init not on disk",
        main_files=[File(path="pkg/__init__.py", role="entry")],
        submodules=[_leaf("pkg/a", "pkg/a/core.py")],
    )
    assert container_module_reason(module, repo) is None


def test_a_submodules_own_init_is_not_the_parents_file(repo: Path) -> None:
    """Own-file attribution is by path prefix, and a nested `__init__.py`
    belongs to the submodule that contains it, not to the parent."""
    _write(repo, "pkg/a/__init__.py", "from pkg.a.core import A\n")
    module = Module(
        name="pkg",
        path="pkg",
        description="namespace",
        main_files=[File(path="pkg/a/__init__.py", role="entry")],
        submodules=[_leaf("pkg/a", "pkg/a/__init__.py")],
    )
    reason = container_module_reason(module, repo)
    assert reason is not None
    assert "no files of its own" in reason


# --- the manager ------------------------------------------------------------

# `make_tree()` is two leaves under two namespaces: `v1` and `v1/attention` hold
# no own files, so with the flag on they are the containers and the two leaves
# are the only modules discovery should ever see.
CONTAINERS = ["v1", "v1/attention"]
LEAVES = ["v1/kv_offload", "v1/attention/paged_kv"]


def _cfg(artifacts: Path) -> SpotlightsManagerConfig:
    return SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(),
    )


def _patch_steps(monkeypatch, discovered: list[str]) -> None:
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(make_tree()),
    )

    def _discover(inp, *, config):
        discovered.append(inp.module_qualified_name)
        return make_discovery_result(inp.module_qualified_name)

    monkeypatch.setattr(orch, "discover", _discover)
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)


def _skipping_input(repo: Path):
    return make_input(repo, enable_deep_research=False).model_copy(
        update={"skip_container_modules": True}
    )


def test_container_modules_are_skipped_without_discovery(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    discovered: list[str] = []
    _patch_steps(monkeypatch, discovered)

    result = run_with_telemetry(_skipping_input(repo), config=_cfg(artifacts))

    assert sorted(discovered) == sorted(LEAVES)
    for qn in CONTAINERS:
        assert result.module_runs[qn].status == "SKIPPED"
    for qn in LEAVES:
        assert result.module_runs[qn].status == "SUCCEEDED"


def test_a_skipped_container_persists_an_empty_candidate_set(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """Status alone would silently undo the whole saving on the next resume.

    `_plan_module` reads a SKIPPED checkpoint *with* zero persisted candidates
    as settled, and one *without* them as "step 2 never finished" -- so it would
    re-run discovery on exactly the modules this flag exists to skip.
    """
    _patch_steps(monkeypatch, [])

    run_with_telemetry(_skipping_input(repo), config=_cfg(artifacts))

    for qn in CONTAINERS:
        mp = P.ManagerPaths(artifacts).for_module(qn)
        assert mp.candidates_path.exists()
        payload = json.loads(mp.candidates_path.read_text(encoding="utf-8"))
        assert payload["module_qualified_name"] == qn
        assert payload["candidates"] == []
        status = json.loads(mp.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "SKIPPED"
        assert status["last_step"] == "candidate_discovery"


def test_resume_does_not_rediscover_a_skipped_container(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _patch_steps(monkeypatch, [])

    inp = _skipping_input(repo)
    run_with_telemetry(inp, config=_cfg(artifacts))

    def _no_step(*args, **kwargs):  # pragma: no cover - asserted not called
        raise AssertionError("resume must not re-enter any module step")

    monkeypatch.setattr(orch, "discover", _no_step)
    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _no_step)
    monkeypatch.setattr(orch, "create_agent_proposals_with_telemetry", _no_step)

    result = run_with_telemetry(inp, config=_cfg(artifacts))
    for qn in CONTAINERS:
        assert result.module_runs[qn].status == "SKIPPED"


def test_the_flag_is_off_by_default(monkeypatch, repo: Path, artifacts: Path) -> None:
    """Off by default: it changes which modules get analyzed, and the saving is
    only ever a routing node whose submodules are targets in their own right."""
    discovered: list[str] = []
    _patch_steps(monkeypatch, discovered)

    inp = make_input(repo, enable_deep_research=False)
    assert inp.skip_container_modules is False
    result = run_with_telemetry(inp, config=_cfg(artifacts))

    assert sorted(discovered) == sorted(CONTAINERS + LEAVES)
    for qn in CONTAINERS:
        assert result.module_runs[qn].status != "SKIPPED"


def test_the_flag_is_recorded_in_the_run_manifest(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """A run has to say which policy produced it; the manifest is where."""
    _patch_steps(monkeypatch, [])

    run_with_telemetry(_skipping_input(repo), config=_cfg(artifacts))

    manifest = json.loads(
        P.ManagerPaths(artifacts).run_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["run_config"]["skip_container_modules"] is True
