# One-shot Claude Code Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `spotlights-engine fix` — a stage that turns one Spotlights candidate into one reviewable `fix.patch` plus `FIX-NOTES.md`, produced by a `claude -p` session in a throwaway git worktree — and a `/spotlights-fix-candidate` skill that drives the same prompt interactively.

**Architecture:** A new `src/spotlights_engine/one_shot_fix/` package imports the five evolver-agnostic modules from `prep_evolve` (`resolve`, `extract`, `spec`, `digest`, `validate_target`) to locate and validate a candidate, then adds its own git-worktree plumbing, prompt builder, `claude -p` runner, and artifact writer. The staleness gate runs **inside the worktree**, not against `--repo`, so the validated bytes are exactly the bytes the agent edits. Nothing is executed, tested, or benchmarked — the candidate's oracles are carried forward verbatim as a recipe for whoever has the hardware.

**Tech Stack:** Python 3.11+, pydantic v2, argparse, `subprocess` + `git` CLI, `claude` CLI, pytest.

## Global Constraints

- Python `>=3.11`; the package ships only `pydantic>=2` and `pyyaml>=6` as runtime deps — **add no new dependencies**.
- Ruff `line-length = 100`, `target-version = "py311"`, lint rules `["E", "F", "I", "W", "UP", "B"]`. Every new module starts with `from __future__ import annotations`.
- Every new module ends with an explicit `__all__` list, matching the existing package convention.
- The target repo is **never modified**. Worktrees are created with `--detach`; no branch is ever created in the target repo.
- **Nothing is executed against the candidate**: no tests, no benchmarks, no builds. The stage and the skill both state this explicitly in their output.
- `mypy` runs over `["src", "tests"]` with `strict = false`, `ignore_missing_imports = true`.
- Tests live under `tests/unit/one_shot_fix/`, mirroring `tests/unit/prep_evolve/`. Run with `uv run pytest` (or `pytest`) from the repo root; `testpaths = ["tests"]`, `addopts = "-ra"`.
- Artifact layout is fixed: `<base>/fix/<module-slug>/<candidate-id>/`, where `<base>` is `--out` when given, else the run directory, and `<module-slug>` is `spotlights_engine.utils.id_helpers.slug_for(qn)`.

---

## File Structure

**New package — `src/spotlights_engine/one_shot_fix/`:**

| File | Responsibility |
|---|---|
| `__init__.py` | Public re-exports (`one_shot_fix`, input/result models, errors). |
| `errors.py` | `OneShotFixError` base + `NotAGitRepoError`, `WorktreeError`. Resolution/staleness failures keep reusing `prep_evolve.errors`. |
| `prompts.py` | `build_fix_prompt(...)` — the fix prompt, single source of truth for both the stage and the skill. |
| `worktree.py` | git plumbing: `require_git_repo`, `create_worktree`, `remove_worktree`, `collect_patch`. Unit-testable, so `git add -N` cannot be silently forgotten. |
| `claude_exec.py` | `run_fix_claude(...)` — env-scrubbed `claude -p` subprocess in the worktree, with `max_turns`, wallclock cap, `stream-json` usage capture. |
| `notes.py` | `render_fix_notes(...)` — the `FIX-NOTES.md` body. |
| `api.py` | `one_shot_fix(input, config)` — resolve → worktree → validate → prompt → run → collect → clean up, plus the batch loop and `--print-prompt` mode. |
| `cli.py` | `spotlights-engine fix` argparse front end. |

**Modified:**

| File | Change |
|---|---|
| `src/spotlights_engine/cli.py:611-618` | Add a `fix` branch to the subcommand dispatch. |
| `README.md:24`, `README.md:100`, `README.md` Skills table, `README.md:209` | Nav link, `fix` subcommand mention, skill row, docs row. |
| `docs/prep-evolve.md` | A pointer framing one-shot fix as the cheap arm. |

**New non-code:**

| File | Responsibility |
|---|---|
| `templates/commands/fix-candidate/SKILL.md` | The bundled skill; installed by `spotlights-engine init` as `.claude/commands/spotlights-fix-candidate/SKILL.md`. |
| `docs/one-shot-fix.md` | User-facing doc: command, skill, artifacts, what is *not* verified. |
| `tests/unit/one_shot_fix/__init__.py` | Test package marker. |
| `tests/unit/one_shot_fix/_fixtures.py` | Fake `claude` binary builder + dirty-repo helper. Reuses `tests/unit/prep_evolve/_fixtures.py` for the result/repo payloads. |
| `tests/unit/one_shot_fix/test_prompts.py` | Task 2. |
| `tests/unit/one_shot_fix/test_worktree.py` | Task 3. |
| `tests/unit/one_shot_fix/test_claude_exec.py` | Task 4. |
| `tests/unit/one_shot_fix/test_notes.py` | Task 5. |
| `tests/unit/one_shot_fix/test_api.py` | Task 6. |
| `tests/unit/one_shot_fix/test_cli.py` | Task 7. |
| `tests/unit/one_shot_fix/test_skill_install.py` | Task 8. |

---

### Task 1: Package skeleton and error hierarchy

**Files:**
- Create: `src/spotlights_engine/one_shot_fix/__init__.py`
- Create: `src/spotlights_engine/one_shot_fix/errors.py`
- Create: `tests/unit/one_shot_fix/__init__.py`
- Test: `tests/unit/one_shot_fix/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `OneShotFixError(Exception)`, `NotAGitRepoError(OneShotFixError)`, `WorktreeError(OneShotFixError)`. Every later task's failure path raises one of these or a `prep_evolve.errors.PrepEvolveError` subclass.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/one_shot_fix/__init__.py` as an empty file, then `tests/unit/one_shot_fix/test_errors.py`:

```python
"""The one_shot_fix error hierarchy."""

from __future__ import annotations

import pytest

from spotlights_engine.one_shot_fix.errors import (
    NotAGitRepoError,
    OneShotFixError,
    WorktreeError,
)


@pytest.mark.parametrize("exc_type", [NotAGitRepoError, WorktreeError])
def test_subclasses_share_the_base(exc_type: type[Exception]) -> None:
    assert issubclass(exc_type, OneShotFixError)


def test_base_is_catchable_as_exception() -> None:
    with pytest.raises(OneShotFixError) as excinfo:
        raise NotAGitRepoError("not a git checkout: /tmp/x")
    assert "not a git checkout" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotlights_engine.one_shot_fix'`

- [ ] **Step 3: Write minimal implementation**

`src/spotlights_engine/one_shot_fix/errors.py`:

```python
"""Error hierarchy for the one-shot fix stage.

Resolution, repo-path, and staleness failures reuse
`spotlights_engine.prep_evolve.errors` — the fix stage runs the same resolver
and the same staleness gate, so it raises the same types. What is new here is
git-worktree failure: `fix` requires a real git checkout, which `prep-evolve`
does not.

The CLI catches `(PrepEvolveError, OneShotFixError)` once and maps both to a
clean stderr message plus a nonzero exit.
"""

from __future__ import annotations


class OneShotFixError(Exception):
    """Base class for one-shot-fix-specific failures."""


class NotAGitRepoError(OneShotFixError):
    """The target repo is not a git checkout, so no worktree can be made.

    Unlike `prep-evolve`, which tolerates a non-git target and records a `None`
    commit, `fix` fails loudly: the base commit is what makes the emitted patch
    applicable, and a worktree cannot exist without one.
    """


class WorktreeError(OneShotFixError):
    """A `git worktree` / `git diff` invocation failed."""


__all__ = [
    "NotAGitRepoError",
    "OneShotFixError",
    "WorktreeError",
]
```

`src/spotlights_engine/one_shot_fix/__init__.py` (placeholder; later tasks extend it):

```python
"""one-shot-fix: turn one Spotlights candidate into one reviewable patch.

The cheap arm next to `prep-evolve`: instead of generating a bundle for an
evolutionary search, run a single `claude -p` session in a throwaway git
worktree and hand back `fix.patch` + `FIX-NOTES.md`. Nothing is executed,
tested, or benchmarked here; the candidate's oracles travel with the patch as
the verification recipe for whoever has the hardware.

See `docs/superpowers/specs/2026-08-20-one-shot-claude-code-fix-design.md`.
"""

from __future__ import annotations

from spotlights_engine.one_shot_fix.errors import (
    NotAGitRepoError,
    OneShotFixError,
    WorktreeError,
)

__all__ = [
    "NotAGitRepoError",
    "OneShotFixError",
    "WorktreeError",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_errors.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/spotlights_engine/one_shot_fix/ tests/unit/one_shot_fix/
git commit -m "feat(one-shot-fix): add package skeleton and error hierarchy"
```

---

### Task 2: The fix prompt

**Files:**
- Create: `src/spotlights_engine/one_shot_fix/prompts.py`
- Test: `tests/unit/one_shot_fix/test_prompts.py`

**Interfaces:**
- Consumes: `spotlights_engine.prep_evolve.spec.EvolveSpec`, `spotlights_engine.prep_evolve.digest.render_digest`.
- Produces:
  - `build_fix_prompt(*, spec: EvolveSpec, worktree: Path) -> str`
  - `CHANGE_SUMMARY_NAME: str` = `"CHANGE-SUMMARY.md"` — the filename the prompt asks the agent to write its rationale to. Task 3 (`collect_patch`) reads and removes it; Task 5 (`render_fix_notes`) folds its text into the notes. All three must use this constant.

This is the module the design calls the single source of truth: the stage builds the prompt with it, and `--print-prompt` (Task 6) prints exactly this string for the skill to consume.

- [ ] **Step 1: Write the failing test**

`tests/unit/one_shot_fix/test_prompts.py`:

```python
"""The fix prompt: oracles verbatim, base SHA present, scope confined."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix.prompts import (
    CHANGE_SUMMARY_NAME,
    build_fix_prompt,
)
from spotlights_engine.prep_evolve.extract import build_spec, infer_direction
from spotlights_engine.prep_evolve.resolve import (
    find_candidate,
    load_result,
    resolve_findings,
    resolve_module,
)
from spotlights_engine.prep_evolve.spec import EvolveSpec, SourceRevision
from spotlights_engine.prep_evolve.validate_target import validate_candidate_target
from tests.unit.prep_evolve._fixtures import (
    CAND_END,
    CAND_FILE,
    CAND_START,
    make_repo,
    write_result,
)

BASE_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def spec(tmp_path: Path) -> EvolveSpec:
    result_path = write_result(tmp_path)
    repo = make_repo(tmp_path)
    loaded = load_result(result_path)
    sel = find_candidate(loaded, "cand-v1_attention-0002")
    validated = validate_candidate_target(repo, sel.candidate)
    return build_spec(
        loaded=loaded,
        module=resolve_module(loaded.project_tree, sel.qn),
        qn=sel.qn,
        candidate=sel.candidate,
        findings=resolve_findings(sel.run, sel.qn),
        repo_path=str(repo),
        validated=validated,
        revision=SourceRevision(
            git_commit=BASE_SHA, dirty=False, captured_at="2026-08-20T00:00:00+00:00"
        ),
        scope="candidate",
        direction=infer_direction(loaded.context.objective),
    )


def test_base_sha_is_present(spec: EvolveSpec, tmp_path: Path) -> None:
    prompt = build_fix_prompt(spec=spec, worktree=tmp_path / "wt")
    assert BASE_SHA in prompt


def test_oracles_appear_verbatim(spec: EvolveSpec, tmp_path: Path) -> None:
    prompt = build_fix_prompt(spec=spec, worktree=tmp_path / "wt")
    # The fixture rationale yields this correctness oracle and these metrics.
    assert "pytest tests/kernels/test_tile.py" in prompt
    assert "TPOT" in prompt and "TTFT" in prompt


def test_scope_is_confined_to_the_candidate_file_and_range(
    spec: EvolveSpec, tmp_path: Path
) -> None:
    prompt = build_fix_prompt(spec=spec, worktree=tmp_path / "wt")
    assert f"{CAND_FILE}:{CAND_START}-{CAND_END}" in prompt
    # The other module main file is NOT in scope for --scope candidate.
    assert "pkg/attn/launch.py" not in prompt


def test_findings_with_urls_and_proposals_are_included(
    spec: EvolveSpec, tmp_path: Path
) -> None:
    prompt = build_fix_prompt(spec=spec, worktree=tmp_path / "wt")
    assert "POD-Attention" in prompt
    assert "https://arxiv.org/abs/2410.18038" in prompt
    assert "Make tile size GQA-aware" in prompt


def test_prompt_forbids_running_tests_and_names_the_worktree(
    spec: EvolveSpec, tmp_path: Path
) -> None:
    worktree = tmp_path / "wt"
    prompt = build_fix_prompt(spec=spec, worktree=worktree)
    assert str(worktree) in prompt
    assert "Do not run tests" in prompt
    assert CHANGE_SUMMARY_NAME in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotlights_engine.one_shot_fix.prompts'`

- [ ] **Step 3: Write minimal implementation**

`src/spotlights_engine/one_shot_fix/prompts.py`:

```python
"""The one-shot fix prompt — the single source of truth.

`spotlights-engine fix` builds the prompt with `build_fix_prompt` and feeds it
to `claude -p`; `fix --print-prompt` prints the same string for
`/spotlights-fix-candidate` to work from in-session. The prompt is shared so
the two paths cannot drift on the part that matters.

The findings/proposals/target body is `prep_evolve.digest.render_digest`, the
same block every evolver bundle embeds. What this module adds around it is the
fix-specific contract: the worktree to edit, the hard scope boundary, the base
commit, the oracles as a recipe rather than a task, and the instruction to
write a rationale file the collector folds into FIX-NOTES.md.
"""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.prep_evolve.digest import render_digest
from spotlights_engine.prep_evolve.spec import EvolveSpec, Target

# The agent writes its rationale here, in the worktree root. `collect_patch`
# reads it and removes it *before* `git add -N`, so it never lands in the
# patch; `render_fix_notes` folds the text into FIX-NOTES.md.
CHANGE_SUMMARY_NAME = "CHANGE-SUMMARY.md"


def _candidate_target(spec: EvolveSpec) -> Target:
    for t in spec.targets:
        if t.scope_kind == "candidate":
            return t
    return spec.targets[0]


def _scope_block(spec: EvolveSpec) -> str:
    """The in-scope files, with validated line ranges. This is the hard bound.

    Written WITHOUT backticks around `file:start-end` on purpose: the range is
    a copy-pasteable location, and a trailing backtick between the path and the
    colon breaks both grep-ability and the tests that assert on it.
    """
    lines: list[str] = []
    for t in spec.targets:
        if t.scope_kind == "candidate" and t.line_start is not None:
            sym = f" — {t.symbol}" if t.symbol else ""
            lines.append(f"- {t.file}:{t.line_start}-{t.line_end}{sym}")
        else:
            role = f" — {t.role}" if t.role else ""
            lines.append(f"- {t.file} (whole file{role})")
    return "\n".join(lines)


def _oracle_block(target: Target) -> str:
    """The oracles verbatim. Recorded, never enforced here."""
    lines: list[str] = []
    if target.oracles.correctness:
        lines.append("Correctness oracle (recorded, NOT run here):")
        lines.extend(f"  {cmd}" for cmd in target.oracles.correctness)
    else:
        lines.append("Correctness oracle: (none recorded)")
    if target.oracles.performance:
        lines.append(
            "Performance oracle (recorded, NOT measured here): "
            f"{target.oracles.performance}"
        )
    else:
        lines.append("Performance oracle: (none recorded — see the objective)")
    return "\n".join(lines)


def build_fix_prompt(*, spec: EvolveSpec, worktree: Path) -> str:
    """Render the fix prompt for one candidate.

    `spec.run.repo_path` is the real target repo (identity); `worktree` is the
    detached, throwaway checkout the agent actually edits.
    """
    target = _candidate_target(spec)
    base_sha = spec.source_revision.git_commit or "(unknown)"
    return f"""You are implementing ONE proposed optimization in an isolated git worktree.

Working directory: {worktree}
This is a detached worktree of {spec.run.repo_path} at commit {base_sha}.
It is throwaway. It has no build artifacts, no virtualenv, and no compiled
extensions — nothing in it is runnable.

{render_digest(spec)}

## In-scope files — do not edit anything else
{_scope_block(spec)}

## Oracles
{_oracle_block(target)}

## What to do
1. Read the in-scope code, and any other file you need for context.
2. Implement the change described above, editing ONLY the in-scope files.
   Prefer the smallest faithful implementation of the proposal over a rewrite.
3. Write `{CHANGE_SUMMARY_NAME}` in the working-directory root: what you
   changed, why, which findings/proposals you drew on, and anything a reviewer
   should check by hand.

## Hard rules
- Do not run tests. Do not run benchmarks. Do not try to build or install
  anything. This worktree cannot execute them, and a fabricated result is
  worse than none. The oracles above are recorded for whoever has the
  hardware — they are not your task.
- Do not commit, branch, stash, or otherwise run git write commands. The patch
  is collected from your uncommitted working-tree changes.
- Do not edit files outside the in-scope list.
- If the change cannot be made faithfully within scope, make no edit and say
  so in `{CHANGE_SUMMARY_NAME}`, with the reason. A missing patch is a fine
  outcome; a patch that cannot be trusted is not.
""".strip()


__all__ = ["CHANGE_SUMMARY_NAME", "build_fix_prompt"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_prompts.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/spotlights_engine/one_shot_fix/ tests/unit/one_shot_fix/
git add src/spotlights_engine/one_shot_fix/prompts.py tests/unit/one_shot_fix/test_prompts.py
git commit -m "feat(one-shot-fix): add the shared fix prompt builder"
```

---

### Task 3: Git worktree plumbing

**Files:**
- Create: `src/spotlights_engine/one_shot_fix/worktree.py`
- Test: `tests/unit/one_shot_fix/test_worktree.py`

**Interfaces:**
- Consumes: `OneShotFixError` subclasses from Task 1; `CHANGE_SUMMARY_NAME` from Task 2.
- Produces:
  - `@dataclass Worktree` with fields `path: Path`, `base_sha: str`, `repo: Path`, `_parent: Path`
  - `require_git_repo(repo: Path) -> str` — returns the HEAD sha, raises `NotAGitRepoError`
  - `create_worktree(repo: Path, base_sha: str) -> Worktree`
  - `remove_worktree(wt: Worktree) -> None` — idempotent, never raises
  - `collect_patch(wt: Worktree) -> tuple[str, str | None]` — returns `(patch_text, change_summary_text_or_None)`

`collect_patch` is where the `git add -N` regression lives: without it a newly added file is invisible to `git diff` and silently dropped from the patch. That is the reason this plumbing is Python and not markdown.

- [ ] **Step 1: Write the failing test**

`tests/unit/one_shot_fix/test_worktree.py`:

```python
"""Worktree lifecycle and patch collection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix.errors import NotAGitRepoError
from spotlights_engine.one_shot_fix.prompts import CHANGE_SUMMARY_NAME
from spotlights_engine.one_shot_fix.worktree import (
    collect_patch,
    create_worktree,
    remove_worktree,
    require_git_repo,
)
from tests.unit.prep_evolve._fixtures import CAND_FILE, make_repo


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def test_require_git_repo_returns_head_sha(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sha = require_git_repo(repo)
    assert len(sha) == 40
    assert sha == _git(repo, "rev-parse", "HEAD").strip()


def test_require_git_repo_rejects_a_non_git_directory(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, git=False)
    with pytest.raises(NotAGitRepoError):
        require_git_repo(repo)


def test_worktree_is_detached_and_creates_no_branch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sha = require_git_repo(repo)
    branches_before = _git(repo, "branch", "--list")
    wt = create_worktree(repo, sha)
    try:
        assert (wt.path / CAND_FILE).is_file()
        assert _git(wt.path, "rev-parse", "HEAD").strip() == sha
        # --detach: no branch named after the worktree dir appears in the repo.
        assert _git(repo, "branch", "--list") == branches_before
    finally:
        remove_worktree(wt)


def test_remove_worktree_prunes_and_is_idempotent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    remove_worktree(wt)
    assert not wt.path.exists()
    assert _git(repo, "worktree", "list").count("\n") == 1  # only the main tree
    remove_worktree(wt)  # second call must not raise


def test_collect_patch_includes_a_modified_file(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        target = wt.path / CAND_FILE
        target.write_text(target.read_text() + "# appended\n", encoding="utf-8")
        patch, summary = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert "# appended" in patch
    assert CAND_FILE in patch
    assert summary is None


def test_collect_patch_includes_an_added_file(tmp_path: Path) -> None:
    """The `git add -N` regression: a new file must not be silently dropped."""
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / "pkg" / "attn" / "table.py").write_text(
            "TILE_TABLE = {128: 64}\n", encoding="utf-8"
        )
        patch, _ = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert "pkg/attn/table.py" in patch
    assert "TILE_TABLE" in patch


def test_collect_patch_extracts_and_excludes_the_change_summary(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / CHANGE_SUMMARY_NAME).write_text(
            "Swapped the heuristic for a table.\n", encoding="utf-8"
        )
        patch, summary = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert summary == "Swapped the heuristic for a table.\n"
    assert CHANGE_SUMMARY_NAME not in patch


def test_collect_patch_is_empty_when_nothing_changed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        patch, summary = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert patch == ""
    assert summary is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_worktree.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotlights_engine.one_shot_fix.worktree'`

- [ ] **Step 3: Write minimal implementation**

`src/spotlights_engine/one_shot_fix/worktree.py`:

```python
"""Throwaway git worktrees and patch collection.

The design puts this plumbing in Python rather than in the skill's markdown
for one concrete reason: `collect_patch` must run `git add -N` before
`git diff`, or a file the agent *added* is invisible to the diff and silently
dropped from the patch. A checklist in prose forgets that; a unit test does
not.

Worktrees are always created with `--detach`. Without it git creates a branch
named after the worktree path's basename *in the target repo*, which violates
the repo-untouched requirement.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.one_shot_fix.errors import NotAGitRepoError, WorktreeError
from spotlights_engine.one_shot_fix.prompts import CHANGE_SUMMARY_NAME

_GIT_TIMEOUT_S = 120


@dataclass
class Worktree:
    """A detached, throwaway checkout of `repo` at `base_sha`.

    `parent` is the temp directory holding `path`; it is removed alongside the
    worktree so no empty scaffolding is left behind.
    """

    path: Path
    base_sha: str
    repo: Path
    parent: Path


def _run_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            # Tolerant decoding: a target repo may hold source bytes that are
            # invalid under the locale's encoding. Strict decoding would raise
            # UnicodeDecodeError — a ValueError, so NOT caught below — and lose
            # the whole patch. A degraded patch beats no patch.
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorktreeError(f"git {' '.join(args)} failed in {cwd}: {exc}") from exc
    if check and completed.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed in {cwd} "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )
    return completed


def require_git_repo(repo: Path) -> str:
    """Return `repo`'s HEAD sha, or raise `NotAGitRepoError`.

    `prep-evolve` tolerates a non-git target and records a `None` commit. `fix`
    cannot: a worktree needs a commit, and the emitted patch is meaningless
    without a recorded base to apply it to.
    """
    completed = _run_git(repo, "rev-parse", "HEAD", check=False)
    sha = completed.stdout.strip()
    if completed.returncode != 0 or not sha:
        raise NotAGitRepoError(
            f"{repo} is not a git checkout (git rev-parse HEAD failed). "
            f"`fix` needs a real git repo: it creates a detached worktree at "
            f"the base commit and records that commit in the patch notes."
        )
    return sha


def create_worktree(repo: Path, base_sha: str) -> Worktree:
    """Create a detached worktree of `repo` at `base_sha` under a temp dir.

    `git worktree add` requires a non-existent path, so the temp directory is
    created first and the worktree goes in a child of it.
    """
    parent = Path(tempfile.mkdtemp(prefix="spotlights-fix-"))
    dest = parent / "worktree"
    try:
        _run_git(repo, "worktree", "add", "--detach", str(dest), base_sha)
    except WorktreeError:
        shutil.rmtree(parent, ignore_errors=True)
        raise
    return Worktree(path=dest, base_sha=base_sha, repo=repo, parent=parent)


def remove_worktree(wt: Worktree) -> None:
    """Remove the worktree and prune its metadata. Idempotent; never raises.

    This runs in a `finally`, including after a timeout or a crash. Without the
    prune, a failed run leaves a stale entry in `.git/worktrees` — and a sweep
    over N candidates leaves N of them.

    Each call needs its own `try/except`: `check=False` only suppresses
    `_run_git`'s return-code check, not its `except (OSError,
    SubprocessError)` branch, which re-raises a hung or un-launchable git as
    `WorktreeError`. Raising from a `finally` would mask the original
    exception — the very failure the timeout exists to contain.
    """
    for args in (
        ("worktree", "remove", "--force", str(wt.path)),
        ("worktree", "prune"),
    ):
        try:
            _run_git(wt.repo, *args, check=False)
        except WorktreeError:
            pass
    shutil.rmtree(wt.parent, ignore_errors=True)


def collect_patch(wt: Worktree) -> tuple[str, str | None]:
    """Return `(patch_text, change_summary)` from the worktree's dirty state.

    Order matters:
    1. read + delete `CHANGE-SUMMARY.md`, so the agent's rationale reaches
       FIX-NOTES.md but never appears in the patch;
    2. `git add -N .` so *added* files show up in the diff;
    3. `git diff` against the base commit already checked out.
    """
    summary_path = wt.path / CHANGE_SUMMARY_NAME
    summary: str | None = None
    if summary_path.is_file():
        summary = summary_path.read_text(encoding="utf-8", errors="replace")
        summary_path.unlink()

    _run_git(wt.path, "add", "-N", ".")
    patch = _run_git(wt.path, "diff").stdout
    return patch, summary


__all__ = [
    "Worktree",
    "collect_patch",
    "create_worktree",
    "remove_worktree",
    "require_git_repo",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_worktree.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/spotlights_engine/one_shot_fix/ tests/unit/one_shot_fix/
git add src/spotlights_engine/one_shot_fix/worktree.py tests/unit/one_shot_fix/test_worktree.py
git commit -m "feat(one-shot-fix): add worktree lifecycle and patch collection"
```

---

### Task 4: The `claude -p` runner

**Files:**
- Create: `src/spotlights_engine/one_shot_fix/claude_exec.py`
- Create: `tests/unit/one_shot_fix/_fixtures.py`
- Test: `tests/unit/one_shot_fix/test_claude_exec.py`

**Interfaces:**
- Consumes: `spotlights_engine.costing.usage.{AgentUsage, claude_usage_from_stream}`.
- Produces:
  - `@dataclass FixRunResult` with fields `candidate_id: str`, `duration_s: float`, `error: str | None = None`, `stdout: bytes = b""`, `stderr: bytes = b""`, `usage: AgentUsage | None = None`
  - `ensure_claude_available() -> None`
  - `run_fix_claude(*, candidate_id: str, prompt: str, worktree: Path, max_turns: int, wallclock_s: int) -> FixRunResult`
- Also produces (test helper, `_fixtures.py`): `write_fake_claude(bin_dir: Path, *, script: str) -> Path` and `FAKE_CLAUDE_SUCCESS` / `FAKE_CLAUDE_FAILURE` script bodies.

This mirrors `src/spotlights_engine/agent_proposals/claude_exec.py:35-196` — same env scrubbing, same `shutil.which` Windows handling, same stream-json usage capture. It differs in two ways: there is **no** `--json-schema` (the deliverable is edited files, not a JSON payload) and the permission mode is `acceptEdits`, not `plan`, because the whole point is that the agent edits the worktree.

- [ ] **Step 1: Write the failing test**

`tests/unit/one_shot_fix/_fixtures.py`:

```python
"""Fakes for one_shot_fix tests.

The `claude` runner is exercised against a fake executable on PATH rather than
a monkeypatched function, so the real subprocess/argv/stream-json path is under
test. Result payloads follow the same terminal-event shape
`claude_usage_from_stream` parses (see `costing/usage.py`).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

RESULT_EVENT = {
    "type": "result",
    "subtype": "success",
    "duration_api_ms": 1200,
    "total_cost_usd": 0.0123,
    "model": "claude-sonnet-5",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
    },
}

# Edits a file in $PWD (the worktree), then emits the terminal result event.
FAKE_CLAUDE_SUCCESS = f"""#!/bin/sh
printf '# touched by fake claude\\n' >> pkg/attn/tile.py
printf '%s\\n' '{json.dumps(RESULT_EVENT)}'
exit 0
"""

FAKE_CLAUDE_FAILURE = """#!/bin/sh
echo "boom: model unavailable" >&2
exit 3
"""

FAKE_CLAUDE_NO_RESULT_EVENT = """#!/bin/sh
printf '%s\\n' '{"type":"system","subtype":"init"}'
exit 0
"""

# Records argv so the test can assert on the flags passed to claude.
# The env var is deliberately NOT named SPOTLIGHTS_*: `_clean_env()` in
# claude_exec.py strips every SPOTLIGHTS_-prefixed variable, so the shim would
# never see it.
FAKE_CLAUDE_ARGV_RECORDER = f"""#!/bin/sh
printf '%s\\n' "$@" > "$TEST_ARGV_FILE"
printf '%s\\n' '{json.dumps(RESULT_EVENT)}'
exit 0
"""


def write_fake_claude(bin_dir: Path, *, script: str) -> Path:
    """Write an executable `claude` shim into `bin_dir` and return its path."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "claude"
    fake.write_text(script, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return fake


def prepend_to_path(monkeypatch, bin_dir: Path) -> None:
    """Put `bin_dir` first on PATH for the duration of a test."""
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


__all__ = [
    "FAKE_CLAUDE_ARGV_RECORDER",
    "FAKE_CLAUDE_FAILURE",
    "FAKE_CLAUDE_NO_RESULT_EVENT",
    "FAKE_CLAUDE_SUCCESS",
    "RESULT_EVENT",
    "prepend_to_path",
    "write_fake_claude",
]
```

`tests/unit/one_shot_fix/test_claude_exec.py`:

```python
"""`run_fix_claude` against a fake `claude` binary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.one_shot_fix.claude_exec import (
    ensure_claude_available,
    run_fix_claude,
)
from tests.unit.one_shot_fix._fixtures import (
    FAKE_CLAUDE_ARGV_RECORDER,
    FAKE_CLAUDE_FAILURE,
    FAKE_CLAUDE_NO_RESULT_EVENT,
    FAKE_CLAUDE_SUCCESS,
    prepend_to_path,
    write_fake_claude,
)
from tests.unit.prep_evolve._fixtures import CAND_FILE, make_repo

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="fake claude shim is a POSIX shell script"
)


def test_successful_run_edits_the_worktree_and_captures_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    prepend_to_path(monkeypatch, write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_SUCCESS).parent)

    result = run_fix_claude(
        candidate_id="cand-v1_attention-0002",
        prompt="do the thing",
        worktree=repo,
        max_turns=5,
        wallclock_s=30,
    )

    assert result.error is None
    assert "# touched by fake claude" in (repo / CAND_FILE).read_text(encoding="utf-8")
    assert result.usage is not None
    assert result.usage.input == 100
    assert result.usage.output == 50
    assert result.duration_s >= 0


def test_nonzero_exit_is_reported_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    prepend_to_path(monkeypatch, write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_FAILURE).parent)

    result = run_fix_claude(
        candidate_id="c1", prompt="p", worktree=repo, max_turns=5, wallclock_s=30
    )

    assert result.error is not None
    assert "exit=3" in result.error
    assert b"boom" in result.stderr


def test_missing_result_event_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike agent_proposals, the deliverable is edited files, not a payload.

    A stream with no terminal result event costs us the usage numbers, not the
    fix — so it degrades to `usage is None` rather than failing the run.
    """
    repo = make_repo(tmp_path)
    prepend_to_path(
        monkeypatch, write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_NO_RESULT_EVENT).parent
    )

    result = run_fix_claude(
        candidate_id="c1", prompt="p", worktree=repo, max_turns=5, wallclock_s=30
    )

    assert result.error is None
    assert result.usage is None


def test_argv_carries_the_prompt_and_edit_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("TEST_ARGV_FILE", str(argv_file))
    prepend_to_path(
        monkeypatch, write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_ARGV_RECORDER).parent
    )

    run_fix_claude(
        candidate_id="c1",
        prompt="THE-PROMPT",
        worktree=repo,
        max_turns=7,
        wallclock_s=30,
    )

    argv = argv_file.read_text(encoding="utf-8").splitlines()
    assert "-p" in argv
    assert "THE-PROMPT" in argv
    assert "acceptEdits" in argv  # must be able to edit; not --permission-mode plan
    assert "7" in argv  # --max-turns


def test_timeout_is_reported_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    prepend_to_path(
        monkeypatch,
        write_fake_claude(tmp_path / "bin", script="#!/bin/sh\nsleep 5\n").parent,
    )

    result = run_fix_claude(
        candidate_id="c1", prompt="p", worktree=repo, max_turns=5, wallclock_s=1
    )

    assert result.error is not None
    assert "timed out" in result.error


def test_ensure_claude_available_raises_when_not_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(AgentProposalsSetupError):
        ensure_claude_available()
```

> **Why the recorder env var is not `SPOTLIGHTS_`-prefixed:** `_clean_env()`
> drops every variable with a `SPOTLIGHTS_` prefix before launching the shim,
> so a `SPOTLIGHTS_TEST_ARGV_FILE` would be scrubbed away and the shim would
> write to an empty path. The scrub list is inherited from
> `agent_proposals/claude_exec.py` and stays as-is; the test variable is named
> `TEST_ARGV_FILE` to sit outside it.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_claude_exec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotlights_engine.one_shot_fix.claude_exec'`

- [ ] **Step 3: Write minimal implementation**

`src/spotlights_engine/one_shot_fix/claude_exec.py`:

```python
"""Thin subprocess wrapper around `claude -p` for the one-shot fix session.

Modeled on `agent_proposals.claude_exec` — same env scrubbing, same
`shutil.which` resolution so a Windows `.CMD` shim is found, same stream-json
usage capture for costing. Two deliberate differences:

- **No `--json-schema`.** The deliverable is edited files in the worktree, not
  a structured payload, so there is nothing to validate against a schema. A
  stream that never reached its terminal `result` event costs us the token
  numbers, not the fix: `usage` degrades to `None` and the run still succeeds.
- **`--permission-mode acceptEdits`, not `plan`.** Step 5 of the pipeline asks
  an agent to *propose*; this stage asks it to *implement*. The blast radius is
  a throwaway detached worktree under a temp dir, never the target repo.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.costing.usage import AgentUsage, claude_usage_from_stream

_DROP_EXACT = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "VIRTUAL_ENV",
    }
)
_DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in _DROP_EXACT or key.startswith(_DROP_PREFIX):
            env.pop(key)
    return env


@dataclass
class FixRunResult:
    """Outcome of one fix session.

    `error is None` means the session completed; it does **not** mean the
    worktree changed. An agent that concluded the change could not be made in
    scope exits cleanly and leaves an empty diff — a legitimate outcome the
    caller records as "no patch produced", not as a failure.
    """

    candidate_id: str
    duration_s: float
    error: str | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    usage: AgentUsage | None = None


def ensure_claude_available() -> None:
    """Setup-time check: the `claude` CLI must be on PATH."""
    if shutil.which("claude") is None:
        raise AgentProposalsSetupError(
            "required CLI not on PATH: claude",
            executable="claude",
        )


def run_fix_claude(
    *,
    candidate_id: str,
    prompt: str,
    worktree: Path,
    max_turns: int,
    wallclock_s: int,
) -> FixRunResult:
    """Run one `claude -p` fix session with `worktree` as the working directory."""
    claude_resolved = shutil.which("claude") or "claude"
    argv = [
        claude_resolved,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--max-turns",
        str(max_turns),
    ]
    env = _clean_env()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            env=env,
            cwd=str(worktree),
            timeout=wallclock_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        stdout = exc.stdout or b""
        return FixRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"claude timed out after {duration:.1f}s",
            stdout=stdout,
            stderr=exc.stderr or b"",
            usage=claude_usage_from_stream(stdout),
        )
    except OSError as exc:
        duration = time.monotonic() - start
        return FixRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"could not launch claude: {exc}",
        )

    duration = time.monotonic() - start
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    usage = claude_usage_from_stream(stdout)

    if completed.returncode != 0:
        stderr_tail = stderr[-500:].decode("utf-8", "replace")
        return FixRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"claude exit={completed.returncode}: stderr={stderr_tail!r}",
            stdout=stdout,
            stderr=stderr,
            usage=usage,
        )

    return FixRunResult(
        candidate_id=candidate_id,
        duration_s=duration,
        stdout=stdout,
        stderr=stderr,
        usage=usage,
    )


__all__ = [
    "FixRunResult",
    "ensure_claude_available",
    "run_fix_claude",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_claude_exec.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/spotlights_engine/one_shot_fix/ tests/unit/one_shot_fix/
git add src/spotlights_engine/one_shot_fix/claude_exec.py tests/unit/one_shot_fix/_fixtures.py tests/unit/one_shot_fix/test_claude_exec.py
git commit -m "feat(one-shot-fix): add the claude -p fix-session runner"
```

---

### Task 5: `FIX-NOTES.md` rendering

**Files:**
- Create: `src/spotlights_engine/one_shot_fix/notes.py`
- Test: `tests/unit/one_shot_fix/test_notes.py`

**Interfaces:**
- Consumes: `EvolveSpec` (Task 2's fixture builds one the same way), `CHANGE_SUMMARY_NAME`.
- Produces: `render_fix_notes(*, spec: EvolveSpec, candidate_id: str, module_qn: str, base_sha: str, repo: Path, change_summary: str | None, patch_produced: bool, agent_error: str | None) -> str`

Recording the base commit is load-bearing: `git diff` embeds no base, and applied to the wrong commit a patch either fails or misapplies.

- [ ] **Step 1: Write the failing test**

`tests/unit/one_shot_fix/test_notes.py`:

```python
"""FIX-NOTES.md content: oracles verbatim, base commit, no-verification statement."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix.notes import render_fix_notes
from spotlights_engine.prep_evolve.extract import build_spec, infer_direction
from spotlights_engine.prep_evolve.resolve import (
    find_candidate,
    load_result,
    resolve_findings,
    resolve_module,
)
from spotlights_engine.prep_evolve.spec import EvolveSpec, SourceRevision
from spotlights_engine.prep_evolve.validate_target import validate_candidate_target
from tests.unit.prep_evolve._fixtures import (
    CAND_END,
    CAND_FILE,
    CAND_START,
    make_repo,
    write_result,
)

BASE_SHA = "0123456789abcdef0123456789abcdef01234567"
CAND_ID = "cand-v1_attention-0002"


@pytest.fixture
def spec_and_repo(tmp_path: Path) -> tuple[EvolveSpec, Path]:
    result_path = write_result(tmp_path)
    repo = make_repo(tmp_path)
    loaded = load_result(result_path)
    sel = find_candidate(loaded, CAND_ID)
    spec = build_spec(
        loaded=loaded,
        module=resolve_module(loaded.project_tree, sel.qn),
        qn=sel.qn,
        candidate=sel.candidate,
        findings=resolve_findings(sel.run, sel.qn),
        repo_path=str(repo),
        validated=validate_candidate_target(repo, sel.candidate),
        revision=SourceRevision(
            git_commit=BASE_SHA, dirty=False, captured_at="2026-08-20T00:00:00+00:00"
        ),
        scope="candidate",
        direction=infer_direction(loaded.context.objective),
    )
    return spec, repo


def _notes(spec_and_repo, **overrides) -> str:
    spec, repo = spec_and_repo
    kwargs = {
        "spec": spec,
        "candidate_id": CAND_ID,
        "module_qn": "v1/attention",
        "base_sha": BASE_SHA,
        "repo": repo,
        "change_summary": "Replaced the heuristic with a lookup table.",
        "patch_produced": True,
        "agent_error": None,
    }
    kwargs.update(overrides)
    return render_fix_notes(**kwargs)


def test_records_identity_and_base_commit(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert CAND_ID in notes
    assert "v1/attention" in notes
    assert BASE_SHA in notes


def test_records_in_scope_files_with_line_ranges(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert f"{CAND_FILE}:{CAND_START}-{CAND_END}" in notes


def test_carries_the_oracles_verbatim(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "pytest tests/kernels/test_tile.py" in notes
    assert "TPOT" in notes


def test_states_that_nothing_was_verified(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "Nothing in this directory was verified" in notes


def test_includes_the_apply_and_verify_recipe(spec_and_repo) -> None:
    spec, repo = spec_and_repo
    notes = _notes(spec_and_repo)
    assert f"git -C {repo} checkout {BASE_SHA}" in notes
    # Every git invocation must be `-C {repo}`-qualified, so the recipe works
    # from wherever the reviewer saved the artifacts. Assert the qualified form
    # AND that the bare form is absent — a reader who copies an unqualified
    # `git apply` either gets "not a git repository" or, if their cwd happens to
    # sit inside another repo, silently patches the wrong one.
    assert f'git -C {repo} apply --check "$PWD/fix.patch"' in notes
    assert f'git -C {repo} apply -3 "$PWD/fix.patch"' in notes
    # Both previously-shipped forms are broken and must stay absent:
    # bare `git apply fix.patch` targets whatever repo the cwd sits in, and
    # `git -C <repo> apply fix.patch` resolves the RELATIVE path under <repo>
    # rather than the artifact dir, failing with "can't open patch".
    assert "git apply --check fix.patch" not in notes
    assert f"git -C {repo} apply --check fix.patch" not in notes
    # `patch` has no -C equivalent; its prose says "from the repo root" instead.
    assert "patch -p1 < fix.patch" in notes


def test_includes_findings_with_urls(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "https://arxiv.org/abs/2410.18038" in notes


def test_agent_summary_is_folded_in(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "Replaced the heuristic with a lookup table." in notes


def test_no_patch_case_is_stated_explicitly(spec_and_repo) -> None:
    notes = _notes(
        spec_and_repo,
        patch_produced=False,
        change_summary="The proposal needs a change in a file outside scope.",
    )
    assert "No patch was produced" in notes
    assert "outside scope" in notes
    assert "git apply" not in notes


def test_agent_error_is_recorded(spec_and_repo) -> None:
    notes = _notes(
        spec_and_repo, patch_produced=False, change_summary=None, agent_error="claude exit=3"
    )
    assert "claude exit=3" in notes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_notes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotlights_engine.one_shot_fix.notes'`

- [ ] **Step 3: Write minimal implementation**

`src/spotlights_engine/one_shot_fix/notes.py`:

```python
"""`FIX-NOTES.md` — the patch's travelling documentation.

The notes make the patch self-contained for a reviewer on another machine:
what was proposed, what changed and why, which research backed it, **which
commit the patch applies to**, and the recorded oracles as a verification
recipe. Recording the base commit is load-bearing — `git diff` embeds no base,
and a patch applied to the wrong commit either fails or, worse, misapplies.

The notes never claim a result. Nothing was executed here (see the design's
"No verification here"), and saying so plainly is part of the deliverable.
"""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.prep_evolve.spec import EvolveSpec, Target


def _candidate_target(spec: EvolveSpec) -> Target:
    for t in spec.targets:
        if t.scope_kind == "candidate":
            return t
    return spec.targets[0]


def _scope_lines(spec: EvolveSpec) -> str:
    """Same no-backtick rule as `prompts._scope_block` — see the note there."""
    lines: list[str] = []
    for t in spec.targets:
        if t.scope_kind == "candidate" and t.line_start is not None:
            sym = f" — {t.symbol}" if t.symbol else ""
            lines.append(f"- {t.file}:{t.line_start}-{t.line_end}{sym}")
        else:
            role = f" — {t.role}" if t.role else ""
            lines.append(f"- {t.file} (whole file{role})")
    return "\n".join(lines)


def _findings_lines(spec: EvolveSpec) -> str:
    if not spec.findings:
        return "_(none linked to this candidate)_"
    return "\n".join(
        f"- {f.title} ({f.source_type}) — {f.url}\n  {f.technique_summary}"
        for f in spec.findings
    )


def _oracle_lines(target: Target) -> str:
    lines: list[str] = []
    if target.oracles.correctness:
        lines.append("**Correctness:**")
        lines.extend(f"- `{cmd}`" for cmd in target.oracles.correctness)
    else:
        lines.append("**Correctness:** _(none recorded)_")
    perf = target.oracles.performance or "_(none recorded — see the objective)_"
    lines.append(f"\n**Performance:** {perf}")
    return "\n".join(lines)


def _outcome_section(
    *,
    repo: Path,
    base_sha: str,
    change_summary: str | None,
    patch_produced: bool,
    agent_error: str | None,
    target: Target,
) -> str:
    summary = change_summary.strip() if change_summary else "_(the agent left no summary)_"
    if agent_error:
        summary = f"{summary}\n\n**Agent session error:** `{agent_error}`"

    if not patch_produced:
        return (
            "## Outcome\n\n"
            "**No patch was produced.** The agent made no in-scope edit. Its "
            "reasoning follows; a missing patch is a legitimate outcome and is "
            "preferable to one that cannot be trusted.\n\n"
            f"{summary}\n"
        )

    oracle_cmd = (
        target.oracles.correctness[0]
        if target.oracles.correctness
        else "# (no correctness oracle was recorded for this candidate)"
    )
    return f"""## What changed and why

{summary}

## Applying and verifying this patch

Run this from the directory containing `fix.patch` — `-C {repo}` chdirs before
resolving the patch path, so a bare relative `fix.patch` would resolve under
`{repo}` instead and fail with "can't open patch".

```bash
git -C {repo} checkout {base_sha}
git -C {repo} apply --check "$PWD/fix.patch" && git -C {repo} apply "$PWD/fix.patch"

# the recorded correctness oracle — run it on a machine that can:
{oracle_cmd}
```

If the patch does not apply cleanly, `git -C {repo} apply -3 fix.patch` falls
back to a three-way merge. Without git, `patch -p1 < fix.patch` works, run from
the repo root.
"""


def render_fix_notes(
    *,
    spec: EvolveSpec,
    candidate_id: str,
    module_qn: str,
    base_sha: str,
    repo: Path,
    change_summary: str | None,
    patch_produced: bool,
    agent_error: str | None,
) -> str:
    """Render `FIX-NOTES.md` for one candidate."""
    target = _candidate_target(spec)
    return f"""# Fix notes — {candidate_id}

- **Candidate:** `{candidate_id}`
- **Module:** `{module_qn}`
- **Repo:** `{repo}`
- **Base commit:** `{base_sha}`
- **Objective:** {spec.objective.goal} (direction: {spec.objective.direction})

## In-scope files

{_scope_lines(spec)}

## The proposal

{target.description}

**Current approach:** {target.current_approach}

**Why it was worth changing:** {target.evolve_rationale}

## Research findings used

{_findings_lines(spec)}

## Recorded oracles

These are carried forward **verbatim** from the candidate. They are the
verification recipe for a machine that can run them.

{_oracle_lines(target)}

> **Nothing in this directory was verified.** No test was run, no benchmark was
> measured, no build was attempted. The patch was produced in a fresh detached
> worktree with no virtualenv and no compiled extensions, on a machine that may
> lack the hardware the performance oracle needs. Treat this as a proposal
> faithfully implemented — not as a measured win.

{_outcome_section(
    repo=repo,
    base_sha=base_sha,
    change_summary=change_summary,
    patch_produced=patch_produced,
    agent_error=agent_error,
    target=target,
)}"""


__all__ = ["render_fix_notes"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_notes.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/spotlights_engine/one_shot_fix/ tests/unit/one_shot_fix/
git add src/spotlights_engine/one_shot_fix/notes.py tests/unit/one_shot_fix/test_notes.py
git commit -m "feat(one-shot-fix): render FIX-NOTES.md with oracles and base commit"
```

---

### Task 6: The orchestrator

**Files:**
- Create: `src/spotlights_engine/one_shot_fix/api.py`
- Modify: `src/spotlights_engine/one_shot_fix/__init__.py`
- Test: `tests/unit/one_shot_fix/test_api.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5, plus `prep_evolve.{resolve, extract, spec, validate_target}`.
- Produces:
  - `OneShotFixInput(BaseModel)`: `result: Path`, `candidate: str | None`, `module: str | None`, `out: Path | None`, `top_n: int | None`, `index: Path | None`, `repo: str | None`, `direction: Direction | None`, `print_prompt: bool = False`, `max_turns: int = 40`, `wallclock_s: int = 1800`
  - `OneShotFixConfig(BaseModel)`: `captured_at: str | None = None`
  - `FixArtifact(BaseModel)`: `candidate_id: str`, `module_qualified_name: str`, `path: str`, `files: list[str]`, `patch_produced: bool`, `usage: AgentUsage | None`
  - `PromptPreview(BaseModel)`: `candidate_id: str`, `module_qualified_name: str`, `worktree: str`, `prompt: str`
  - `SkippedFix(BaseModel)`: `candidate_id: str | None`, `module_qualified_name: str | None`, `reason: str`
  - `OneShotFixResult(BaseModel)`: `fixes: list[FixArtifact]`, `prompts: list[PromptPreview]`, `skipped: list[SkippedFix]`, `warnings: list[str]`
  - `one_shot_fix(input: OneShotFixInput, config: OneShotFixConfig | None = None, *, claude_runner=None) -> OneShotFixResult`

`claude_runner` defaults to `run_fix_claude` and is injectable, matching the `agent_proposals` pattern — that is what lets the orchestrator tests run without a subprocess.

- [ ] **Step 1: Write the failing test**

`tests/unit/one_shot_fix/test_api.py`:

```python
"""Orchestration: worktree-scoped validation, batch semantics, cleanup."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix.api import (
    OneShotFixInput,
    one_shot_fix,
)
from spotlights_engine.one_shot_fix.claude_exec import FixRunResult
from spotlights_engine.one_shot_fix.errors import NotAGitRepoError
from spotlights_engine.prep_evolve.errors import StalenessError
from tests.unit.prep_evolve._fixtures import (
    CAND_FILE,
    make_repo,
    write_index,
    write_result,
    write_sorted,
)

CAND_ID = "cand-v1_attention-0002"


def _worktrees(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.startswith("worktree ")]


def _runner(*, edit: str | None = "# agent edit\n", summary: str | None = "did the thing",
            error: str | None = None, new_file: str | None = None):
    """A fake claude_runner that edits the worktree it is handed."""

    def _run(*, candidate_id: str, prompt: str, worktree: Path, max_turns: int,
             wallclock_s: int) -> FixRunResult:
        if edit is not None:
            target = worktree / CAND_FILE
            target.write_text(target.read_text(encoding="utf-8") + edit, encoding="utf-8")
        if new_file is not None:
            (worktree / new_file).write_text("NEW = 1\n", encoding="utf-8")
        if summary is not None:
            (worktree / "CHANGE-SUMMARY.md").write_text(summary, encoding="utf-8")
        return FixRunResult(candidate_id=candidate_id, duration_s=0.01, error=error)

    return _run


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    """(run_dir, repo) with result.json + index.md + a committed repo."""
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path)
    write_index(run_dir, repo)
    return run_dir, repo


def test_single_candidate_writes_patch_and_notes(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.fixes) == 1
    fix = result.fixes[0]
    assert fix.candidate_id == CAND_ID
    out_dir = Path(fix.path)
    assert out_dir == run_dir / "fix" / "v1_attention" / CAND_ID
    assert sorted(fix.files) == ["FIX-NOTES.md", "fix.patch"]
    assert "# agent edit" in (out_dir / "fix.patch").read_text(encoding="utf-8")
    assert "did the thing" in (out_dir / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert fix.patch_produced is True


def test_added_file_appears_in_the_patch(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file="pkg/attn/table.py"),
    )
    patch = (Path(result.fixes[0].path) / "fix.patch").read_text(encoding="utf-8")
    assert "pkg/attn/table.py" in patch


def test_validation_runs_against_the_worktree_not_a_dirty_repo(run) -> None:
    """A dirty --repo must not affect the outcome: the gate reads worktree bytes."""
    run_dir, repo = run
    # Break the candidate's symbol in the *working tree* only; HEAD is still good.
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.fixes) == 1
    assert result.skipped == []


def test_staleness_at_head_raises_for_an_explicit_candidate(run) -> None:
    run_dir, repo = run
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "gut"],
        cwd=repo,
        check=True,
    )

    with pytest.raises(StalenessError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )


def test_staleness_in_a_batch_is_skipped_with_a_reason(run) -> None:
    run_dir, repo = run
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "gut"],
        cwd=repo,
        check=True,
    )

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo)),  # no --candidate → batch
        claude_runner=_runner(),
    )

    assert result.fixes == []
    assert len(result.skipped) == 1
    assert result.skipped[0].candidate_id == CAND_ID
    assert "stale" in result.skipped[0].reason.lower()


def test_worktree_is_removed_on_the_success_path(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)
    one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    assert _worktrees(repo) == before


def test_worktree_is_removed_when_the_agent_fails(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary=None, error="claude exit=3"),
    )
    assert _worktrees(repo) == before
    notes = (Path(result.fixes[0].path) / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert "claude exit=3" in notes
    assert result.fixes[0].patch_produced is False


def test_worktree_is_removed_when_the_runner_raises(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)

    def _boom(**kwargs):
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_boom,
        )
    assert _worktrees(repo) == before


def test_no_edit_produces_notes_but_no_patch(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    fix = result.fixes[0]
    assert fix.patch_produced is False
    assert fix.files == ["FIX-NOTES.md"]
    assert not (Path(fix.path) / "fix.patch").exists()


def test_non_git_repo_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path, git=False)

    with pytest.raises(NotAGitRepoError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )


def test_print_prompt_leaves_the_worktree_and_runs_no_agent(run) -> None:
    run_dir, repo = run
    called: list[str] = []

    def _spy(**kwargs):
        called.append("ran")
        raise AssertionError("claude must not run under --print-prompt")

    result = one_shot_fix(
        OneShotFixInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_spy,
    )

    assert called == []
    assert result.fixes == []
    assert len(result.prompts) == 1
    preview = result.prompts[0]
    worktree = Path(preview.worktree)
    assert worktree.is_dir()
    assert (worktree / CAND_FILE).is_file()
    assert CAND_FILE in preview.prompt
    assert len(_worktrees(repo)) == 2  # main tree + the one left for the caller

    # Not our job to clean up under --print-prompt, but don't leak in the test.
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=True
    )


def test_top_n_over_a_ranking_selects_the_ranked_prefix(run) -> None:
    run_dir, repo = run
    sorted_dir = write_sorted(run_dir, [CAND_ID])

    result = one_shot_fix(
        OneShotFixInput(result=sorted_dir, repo=str(repo), top_n=1),
        claude_runner=_runner(),
    )

    assert len(result.fixes) == 1
    # Artifacts land under the run dir, not inside sorted/.
    assert Path(result.fixes[0].path) == run_dir / "fix" / "v1_attention" / CAND_ID


def test_out_overrides_the_artifact_base(run, tmp_path: Path) -> None:
    run_dir, repo = run
    out = tmp_path / "elsewhere"
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID, out=out),
        claude_runner=_runner(),
    )
    assert Path(result.fixes[0].path) == out / "fix" / "v1_attention" / CAND_ID
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotlights_engine.one_shot_fix.api'`

- [ ] **Step 3: Write minimal implementation**

`src/spotlights_engine/one_shot_fix/api.py`:

```python
"""`one_shot_fix(input, config) -> OneShotFixResult`.

Per candidate: resolve → capture base sha → worktree → **validate inside the
worktree** → build prompt → run `claude -p` → collect patch + notes → remove
the worktree in a `finally`.

Two orchestration decisions carry the design's weight:

- **Validation runs against the worktree, not `--repo`.** The bytes validated
  are then exactly the bytes the agent edits, the recorded excerpt hash is
  truthful, and your own checkout may be dirty while a fix runs. `prep-evolve`
  has no such concern because it validates and points the evolver at the same
  path; the worktree introduces the asymmetry. With no tests being run, this
  gate is the only correctness check in the design.
- **`print_prompt` runs steps 1-5 and stops**, leaving the worktree in place.
  That is what `/spotlights-fix-candidate` consumes: it gets a validated
  worktree and the prompt in one call, so the staleness gate is never
  reimplemented in markdown.

Failure semantics match `prep_evolve`'s batch loop: a single explicit
`--candidate` raises; a sweep records a per-candidate skip and continues.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.one_shot_fix.claude_exec import FixRunResult, run_fix_claude
from spotlights_engine.one_shot_fix.errors import OneShotFixError
from spotlights_engine.one_shot_fix.notes import render_fix_notes
from spotlights_engine.one_shot_fix.prompts import build_fix_prompt
from spotlights_engine.one_shot_fix.worktree import (
    Worktree,
    collect_patch,
    create_worktree,
    remove_worktree,
    require_git_repo,
)
from spotlights_engine.prep_evolve.errors import PrepEvolveError, SelectionError
from spotlights_engine.prep_evolve.extract import Direction, build_spec, infer_direction
from spotlights_engine.prep_evolve.resolve import (
    CandidateSelection,
    LoadedResult,
    find_candidate,
    iter_candidates,
    load_ranking,
    load_result,
    resolve_findings,
    resolve_module,
    resolve_repo_path,
    resolve_result_location,
)
from spotlights_engine.prep_evolve.spec import EvolveSpec, SourceRevision
from spotlights_engine.prep_evolve.validate_target import (
    ensure_repo_dir,
    validate_candidate_target,
)
from spotlights_engine.utils.id_helpers import slug_for

ClaudeRunner = Callable[..., FixRunResult]

PATCH_NAME = "fix.patch"
NOTES_NAME = "FIX-NOTES.md"


class OneShotFixInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    result: Path
    candidate: str | None = None  # omit to sweep every candidate
    module: str | None = None
    out: Path | None = None  # base dir; defaults to the run directory
    top_n: int | None = Field(default=None, ge=1)  # ranked sources only
    index: Path | None = None
    repo: str | None = None
    direction: Direction | None = None
    print_prompt: bool = False
    max_turns: int = Field(default=40, ge=1)
    wallclock_s: int = Field(default=1800, ge=1)


class OneShotFixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captured_at: str | None = None  # injected timestamp; default = now (UTC)


class FixArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    module_qualified_name: str
    path: str
    files: list[str]
    patch_produced: bool
    base_sha: str
    usage: AgentUsage | None = None


class PromptPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    module_qualified_name: str
    worktree: str
    base_sha: str
    prompt: str


class SkippedFix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    candidate_id: str | None = None
    module_qualified_name: str | None = None


class OneShotFixResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixes: list[FixArtifact] = Field(default_factory=list)
    prompts: list[PromptPreview] = Field(default_factory=list)
    skipped: list[SkippedFix] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _fix_dir(base: Path, qn: str, candidate_id: str) -> Path:
    """`fix/<module-slug>/<candidate-id>/`, mirroring the evolve/ layout."""
    return base / "fix" / slug_for(qn) / candidate_id


def _select(
    input: OneShotFixInput,
    loaded: LoadedResult,
    ranking: Path | None,
    skipped: list[SkippedFix],
) -> list[CandidateSelection]:
    """Resolve the candidate selection, mirroring prep_evolve's three modes."""
    if input.candidate is not None:
        sel = find_candidate(loaded, input.candidate)
        if input.module is not None and input.module != sel.qn:
            raise SelectionError(
                f"--module {input.module!r} does not match candidate "
                f"{input.candidate!r}, which is in module {sel.qn!r}; omit --module"
            )
        return [sel]

    if ranking is not None:
        ranked_ids = load_ranking(ranking)
        if input.top_n is not None:
            ranked_ids = ranked_ids[: input.top_n]
        selections: list[CandidateSelection] = []
        for cid in ranked_ids:
            try:
                selections.append(find_candidate(loaded, cid))
            except SelectionError as exc:
                skipped.append(SkippedFix(reason=str(exc), candidate_id=cid))
        return selections

    if input.top_n is not None:
        raise SelectionError(
            "--top-n requires a ranked source; point --result at a sorted/ "
            "directory or a sorted_candidates.json (a plain result.json has no ranking)"
        )
    return iter_candidates(loaded)


def _build_spec_in_worktree(
    *,
    sel: CandidateSelection,
    loaded: LoadedResult,
    repo_path: Path,
    worktree: Worktree,
    captured_at: str,
    direction: Direction,
) -> EvolveSpec:
    """Validate against the worktree, then assemble the spec.

    `validate_candidate_target(worktree.path, ...)` — not `repo_path` — is the
    whole point: validated bytes == edited bytes, and a dirty checkout is
    irrelevant.
    """
    validated = validate_candidate_target(worktree.path, sel.candidate)
    return build_spec(
        loaded=loaded,
        module=resolve_module(loaded.project_tree, sel.qn),
        qn=sel.qn,
        candidate=sel.candidate,
        findings=resolve_findings(sel.run, sel.qn),
        repo_path=str(repo_path),
        validated=validated,
        revision=SourceRevision(
            git_commit=worktree.base_sha, dirty=False, captured_at=captured_at
        ),
        scope="candidate",
        direction=direction,
    )


def _write_artifacts(
    *,
    out_dir: Path,
    spec: EvolveSpec,
    sel: CandidateSelection,
    repo_path: Path,
    base_sha: str,
    patch: str,
    change_summary: str | None,
    agent_error: str | None,
) -> tuple[list[str], bool]:
    """Write `fix.patch` (when non-empty) and `FIX-NOTES.md`. Returns (files, produced)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    patch_produced = bool(patch.strip())

    if patch_produced:
        header = (
            f"# spotlights one-shot fix\n"
            f"# candidate: {sel.candidate.id}\n"
            f"# module:    {sel.qn}\n"
            f"# repo:      {repo_path}\n"
            f"# base:      {base_sha}\n"
            f"# apply with: git -C {repo_path} checkout {base_sha} && "
            f"git -C {repo_path} apply {PATCH_NAME}\n"
        )
        (out_dir / PATCH_NAME).write_text(header + patch, encoding="utf-8")
        files.append(PATCH_NAME)

    notes = render_fix_notes(
        spec=spec,
        candidate_id=sel.candidate.id,
        module_qn=sel.qn,
        base_sha=base_sha,
        repo=repo_path,
        change_summary=change_summary,
        patch_produced=patch_produced,
        agent_error=agent_error,
    )
    (out_dir / NOTES_NAME).write_text(notes, encoding="utf-8")
    files.append(NOTES_NAME)

    return sorted(files), patch_produced


def _process_candidate(
    *,
    sel: CandidateSelection,
    loaded: LoadedResult,
    repo_path: Path,
    base: Path,
    base_sha: str,
    input: OneShotFixInput,
    captured_at: str,
    direction: Direction,
    claude_runner: ClaudeRunner,
    result: OneShotFixResult,
) -> None:
    """Run the full per-candidate pipeline, appending to `result`."""
    worktree = create_worktree(repo_path, base_sha)
    keep_worktree = False
    try:
        spec = _build_spec_in_worktree(
            sel=sel,
            loaded=loaded,
            repo_path=repo_path,
            worktree=worktree,
            captured_at=captured_at,
            direction=direction,
        )
        prompt = build_fix_prompt(spec=spec, worktree=worktree.path)

        if input.print_prompt:
            keep_worktree = True
            result.prompts.append(
                PromptPreview(
                    candidate_id=sel.candidate.id,
                    module_qualified_name=sel.qn,
                    worktree=str(worktree.path),
                    base_sha=base_sha,
                    prompt=prompt,
                )
            )
            return

        run = claude_runner(
            candidate_id=sel.candidate.id,
            prompt=prompt,
            worktree=worktree.path,
            max_turns=input.max_turns,
            wallclock_s=input.wallclock_s,
        )
        patch, change_summary = collect_patch(worktree)
    finally:
        if not keep_worktree:
            remove_worktree(worktree)

    files, patch_produced = _write_artifacts(
        out_dir=_fix_dir(base, sel.qn, sel.candidate.id),
        spec=spec,
        sel=sel,
        repo_path=repo_path,
        base_sha=base_sha,
        patch=patch,
        change_summary=change_summary,
        agent_error=run.error,
    )
    result.fixes.append(
        FixArtifact(
            candidate_id=sel.candidate.id,
            module_qualified_name=sel.qn,
            path=str(_fix_dir(base, sel.qn, sel.candidate.id)),
            files=files,
            patch_produced=patch_produced,
            base_sha=base_sha,
            usage=run.usage,
        )
    )


def one_shot_fix(
    input: OneShotFixInput,
    config: OneShotFixConfig | None = None,
    *,
    claude_runner: ClaudeRunner | None = None,
) -> OneShotFixResult:
    """Turn one (or every) candidate into a patch plus its notes."""
    config = config or OneShotFixConfig()
    captured_at = config.captured_at or _now_iso()
    runner = claude_runner or run_fix_claude
    result = OneShotFixResult()

    # 1. locate + load the run.
    location = resolve_result_location(input.result)
    loaded = load_result(location.result_json)

    # 2. resolve the repo, and require a real git checkout.
    repo_path = resolve_repo_path(input.repo, input.index or location.index)
    ensure_repo_dir(repo_path)
    base_sha = require_git_repo(repo_path)

    base = input.out or location.run_dir

    direction = input.direction or infer_direction(loaded.context.objective)
    if input.direction is None:
        result.warnings.append(
            f"direction inferred as {direction!r} from the objective; pass --direction to override"
        )

    batch = input.candidate is None
    selections = _select(input, loaded, location.ranking, result.skipped)

    for sel in selections:
        try:
            _process_candidate(
                sel=sel,
                loaded=loaded,
                repo_path=repo_path,
                base=base,
                base_sha=base_sha,
                input=input,
                captured_at=captured_at,
                direction=direction,
                claude_runner=runner,
                result=result,
            )
        except (PrepEvolveError, OneShotFixError) as exc:
            if not batch:
                raise
            result.skipped.append(
                SkippedFix(
                    reason=str(exc),
                    candidate_id=sel.candidate.id,
                    module_qualified_name=sel.qn,
                )
            )

    return result


__all__ = [
    "FixArtifact",
    "OneShotFixConfig",
    "OneShotFixInput",
    "OneShotFixResult",
    "PromptPreview",
    "SkippedFix",
    "one_shot_fix",
]
```

Replace `src/spotlights_engine/one_shot_fix/__init__.py` with:

```python
"""one-shot-fix: turn one Spotlights candidate into one reviewable patch.

The cheap arm next to `prep-evolve`: instead of generating a bundle for an
evolutionary search, run a single `claude -p` session in a throwaway git
worktree and hand back `fix.patch` + `FIX-NOTES.md`. Nothing is executed,
tested, or benchmarked here; the candidate's oracles travel with the patch as
the verification recipe for whoever has the hardware.

See `docs/superpowers/specs/2026-08-20-one-shot-claude-code-fix-design.md`.
"""

from __future__ import annotations

from spotlights_engine.one_shot_fix.api import (
    FixArtifact,
    OneShotFixConfig,
    OneShotFixInput,
    OneShotFixResult,
    PromptPreview,
    SkippedFix,
    one_shot_fix,
)
from spotlights_engine.one_shot_fix.errors import (
    NotAGitRepoError,
    OneShotFixError,
    WorktreeError,
)
from spotlights_engine.one_shot_fix.prompts import build_fix_prompt

__all__ = [
    "FixArtifact",
    "NotAGitRepoError",
    "OneShotFixConfig",
    "OneShotFixError",
    "OneShotFixInput",
    "OneShotFixResult",
    "PromptPreview",
    "SkippedFix",
    "WorktreeError",
    "build_fix_prompt",
    "one_shot_fix",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_api.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Run the whole new suite and lint**

Run: `uv run pytest tests/unit/one_shot_fix/ -v && uv run ruff check src/spotlights_engine/one_shot_fix/ tests/unit/one_shot_fix/`
Expected: PASS (41 passed), ruff reports `All checks passed!`

> **Scope ruff to the new package.** `uv run ruff check src/ tests/` reports
> ~89 errors on this repo at baseline, all pre-existing in `signal_pipeline`,
> `schemas`, `doctor`, `validation`, and older test modules that this work never
> touches. A repo-wide run cannot distinguish your regressions from that
> backlog; the scoped run can.

- [ ] **Step 6: Commit**

```bash
git add src/spotlights_engine/one_shot_fix/ tests/unit/one_shot_fix/test_api.py
git commit -m "feat(one-shot-fix): add the fix orchestrator with worktree-scoped validation"
```

---

### Task 7: The `fix` CLI subcommand

**Files:**
- Create: `src/spotlights_engine/one_shot_fix/cli.py`
- Modify: `src/spotlights_engine/cli.py:611-618`
- Test: `tests/unit/one_shot_fix/test_cli.py`

**Interfaces:**
- Consumes: `one_shot_fix`, `OneShotFixInput`, `OneShotFixConfig` from Task 6.
- Produces: `main(argv: list[str] | None = None) -> int` in `one_shot_fix.cli`, dispatched from `spotlights_engine.cli.main` on `argv[0] == "fix"`.

Exit codes follow `prep_evolve/cli.py:101-165`: `2` for a resolution/validation failure or a bad `--top-n`, `1` when nothing at all was produced, `0` otherwise.

- [ ] **Step 1: Write the failing test**

`tests/unit/one_shot_fix/test_cli.py`:

```python
"""`spotlights-engine fix` argument handling and output shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.cli import main as engine_main
from spotlights_engine.one_shot_fix.cli import main as fix_main
from tests.unit.prep_evolve._fixtures import CAND_FILE, make_repo, write_index, write_result

CAND_ID = "cand-v1_attention-0002"


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path)
    write_index(run_dir, repo)
    return run_dir, repo


def test_bad_top_n_exits_2(run, capsys: pytest.CaptureFixture) -> None:
    run_dir, repo = run
    code = fix_main(
        ["--result", str(run_dir), "--repo", str(repo), "--top-n", "zero"]
    )
    assert code == 2
    assert "--top-n" in capsys.readouterr().err


def test_zero_top_n_exits_2(run) -> None:
    run_dir, repo = run
    assert fix_main(["--result", str(run_dir), "--repo", str(repo), "--top-n", "0"]) == 2


def test_unknown_candidate_exits_2(run, capsys: pytest.CaptureFixture) -> None:
    run_dir, repo = run
    code = fix_main(
        ["--result", str(run_dir), "--repo", str(repo), "--candidate", "cand-nope-0001"]
    )
    assert code == 2
    assert "fix:" in capsys.readouterr().err


def test_non_git_repo_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path, git=False)
    code = fix_main(
        ["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID]
    )
    assert code == 2
    assert "git" in capsys.readouterr().err


def test_print_prompt_writes_the_prompt_and_worktree_to_stdout(
    run, capsys: pytest.CaptureFixture
) -> None:
    run_dir, repo = run
    code = fix_main(
        [
            "--result",
            str(run_dir),
            "--repo",
            str(repo),
            "--candidate",
            CAND_ID,
            "--print-prompt",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "WORKTREE:" in out
    assert CAND_FILE in out

    # The worktree is left in place for the caller; clean it up.
    import subprocess

    worktree = next(
        ln.split("WORKTREE:", 1)[1].strip() for ln in out.splitlines() if "WORKTREE:" in ln
    )
    assert Path(worktree).is_dir()
    subprocess.run(["git", "worktree", "remove", "--force", worktree], cwd=repo, check=True)


def test_engine_dispatch_routes_fix_to_the_subcommand(
    run, capsys: pytest.CaptureFixture
) -> None:
    run_dir, repo = run
    code = engine_main(
        ["fix", "--result", str(run_dir), "--repo", str(repo), "--top-n", "zero"]
    )
    assert code == 2
    assert "--top-n" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spotlights_engine.one_shot_fix.cli'`

- [ ] **Step 3: Write minimal implementation**

`src/spotlights_engine/one_shot_fix/cli.py`:

```python
"""CLI for `spotlights-engine fix`.

Wired into the top-level dispatch in `spotlights_engine.cli.main`. Argument
shape deliberately mirrors `prep-evolve` (`--result`, `--repo`, `--index`,
`--candidate`, `--module`, `--out`, `--top-n`, `--direction`) so the two stages
are interchangeable at the call site.

`--print-prompt` runs resolution, worktree creation, and validation, then
prints the worktree path and the prompt and exits — leaving the worktree in
place for `/spotlights-fix-candidate` to work in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spotlights_engine.one_shot_fix.api import (
    OneShotFixConfig,
    OneShotFixInput,
    one_shot_fix,
)
from spotlights_engine.one_shot_fix.errors import OneShotFixError
from spotlights_engine.prep_evolve.errors import PrepEvolveError


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine fix",
        description=(
            "Implement one candidate with a single Claude Code session in a "
            "throwaway git worktree, and write fix.patch + FIX-NOTES.md. "
            "Runs no tests and no benchmarks; the target repo is never modified."
        ),
    )
    p.add_argument(
        "--result",
        type=Path,
        required=True,
        help=(
            "A finished run's result.json, or the run directory / index.md / "
            "sorted/ dir / sorted_candidates.{json,md} that self-locates it."
        ),
    )
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Rendered index.md; used only as a --repo fallback.",
    )
    p.add_argument(
        "--repo",
        default=None,
        help=(
            "Target repo path; wins over --index. Must be a git checkout. "
            "One of --repo/--index must resolve."
        ),
    )
    p.add_argument(
        "--module",
        default=None,
        help="Slash-form qualified name (optional; inferred from --candidate).",
    )
    p.add_argument(
        "--candidate",
        default=None,
        help="Candidate id, e.g. cand-....; omit to fix every candidate.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Artifacts base dir (optional; defaults to the run directory).",
    )
    p.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default=None,
        help="Override the inferred optimization direction.",
    )
    p.add_argument(
        "--top-n",
        dest="top_n",
        default="all",
        help=(
            "For a sorted --result: fix only the top N ranked candidates "
            "('all' = every ranked candidate, the default)."
        ),
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=40,
        help="Cap on agent turns per candidate (default: 40).",
    )
    p.add_argument(
        "--wallclock",
        dest="wallclock_s",
        type=int,
        default=1800,
        help="Wall-clock cap in seconds per candidate (default: 1800).",
    )
    p.add_argument(
        "--print-prompt",
        action="store_true",
        help=(
            "Resolve, create and validate the worktree, then print the "
            "worktree path and the prompt and exit. Runs no agent, writes no "
            "artifacts, and leaves the worktree in place for the caller."
        ),
    )
    return p


def _parse_top_n(raw: str) -> int | None:
    """`'all'` -> None; a positive int -> that int. Raises ValueError otherwise."""
    text = str(raw).strip().lower()
    if text == "all":
        return None
    n = int(text)  # ValueError propagates
    if n < 1:
        raise ValueError(f"--top-n must be >= 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    try:
        top_n = _parse_top_n(args.top_n)
    except ValueError as exc:
        print(
            f"fix: --top-n must be a positive integer or 'all', got "
            f"{args.top_n!r} ({exc})",
            file=sys.stderr,
        )
        return 2

    inp = OneShotFixInput(
        result=args.result,
        index=args.index,
        repo=args.repo,
        module=args.module,
        candidate=args.candidate,
        out=args.out,
        direction=args.direction,
        top_n=top_n,
        max_turns=args.max_turns,
        wallclock_s=args.wallclock_s,
        print_prompt=args.print_prompt,
    )

    try:
        result = one_shot_fix(inp, OneShotFixConfig())
    except (PrepEvolveError, OneShotFixError) as exc:
        print(f"fix: {exc}", file=sys.stderr)
        return 2

    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)

    for preview in result.prompts:
        print(f"CANDIDATE: {preview.candidate_id}")
        print(f"MODULE:    {preview.module_qualified_name}")
        print(f"BASE:      {preview.base_sha}")
        print(f"WORKTREE:  {preview.worktree}")
        print("PROMPT:")
        print(preview.prompt)

    for fix in result.fixes:
        state = "patch + notes" if fix.patch_produced else "notes only (no patch)"
        print(f"{fix.candidate_id}: {fix.path} ({state})")

    if result.fixes:
        print(
            "fix: nothing was verified — no tests and no benchmarks were run. "
            "See FIX-NOTES.md for the recorded oracles.",
            file=sys.stderr,
        )
    for s in result.skipped:
        who = f"{s.module_qualified_name}/{s.candidate_id} " if s.candidate_id else ""
        print(f"  skipped {who}: {s.reason}", file=sys.stderr)

    produced = result.fixes or result.prompts or result.skipped
    return 0 if produced else 1


if __name__ == "__main__":
    sys.exit(main())
```

Then edit `src/spotlights_engine/cli.py`. Find this block (around line 611):

```python
    if raw_argv and raw_argv[0] == "prep-evolve":
        from spotlights_engine.prep_evolve.cli import main as prep_main

        return prep_main(raw_argv[1:])
```

and insert immediately after it:

```python
    if raw_argv and raw_argv[0] == "fix":
        from spotlights_engine.one_shot_fix.cli import main as fix_main

        return fix_main(raw_argv[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_cli.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Verify the subcommand is reachable end-to-end**

Run: `uv run spotlights-engine fix --help`
Expected: the `spotlights-engine fix` usage block, including `--print-prompt` and the "Runs no tests and no benchmarks" line.

- [ ] **Step 6: Commit**

```bash
uv run ruff check src/spotlights_engine/one_shot_fix/ src/spotlights_engine/cli.py tests/unit/one_shot_fix/
git add src/spotlights_engine/one_shot_fix/cli.py src/spotlights_engine/cli.py tests/unit/one_shot_fix/test_cli.py
git commit -m "feat(one-shot-fix): wire the fix subcommand into the engine CLI"
```

---

### Task 8: The `/spotlights-fix-candidate` skill

**Files:**
- Create: `templates/commands/fix-candidate/SKILL.md`
- Test: `tests/unit/one_shot_fix/test_skill_install.py`

**Interfaces:**
- Consumes: `spotlights-engine fix --print-prompt` (Task 7).
- Produces: an installed skill at `.claude/commands/spotlights-fix-candidate/SKILL.md`. The `spotlights-` prefix is applied at install time by `init_skills._plan_install_items` (`src/spotlights_engine/init_skills.py:144-167`) — the template directory is named `fix-candidate`, unprefixed, like `share-candidates`.

The skill's body is untested (consistent with the other bundled skills); what *is* tested is that `init` picks the directory up and installs it under the prefixed name.

- [ ] **Step 1: Write the failing test**

`tests/unit/one_shot_fix/test_skill_install.py`:

```python
"""`spotlights-engine init` installs the fix-candidate skill."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.init_skills import _plan_install_items, install_skills

REL = ".claude/commands/spotlights-fix-candidate/SKILL.md"


def test_fix_candidate_is_in_the_install_plan() -> None:
    rel_paths = {item.rel_path for item in _plan_install_items()}
    assert REL in rel_paths


def test_init_writes_the_skill_into_claude_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert install_skills(scope="project") == 0
    installed = tmp_path / REL
    assert installed.is_file()
    text = installed.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: fix-candidate" in text
    assert "--print-prompt" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/one_shot_fix/test_skill_install.py -v`
Expected: FAIL — `assert '.claude/commands/spotlights-fix-candidate/SKILL.md' in rel_paths` (the template directory does not exist yet)

- [ ] **Step 3: Write minimal implementation**

`templates/commands/fix-candidate/SKILL.md`:

```markdown
---
name: fix-candidate
description: Use when implementing ONE Spotlights candidate as a reviewable patch in-session — "fix this candidate", "implement candidate cand-...", "one-shot fix". Runs `spotlights-engine fix --print-prompt` to get a validated throwaway worktree plus the fix prompt, does the work interactively, and collects fix.patch + FIX-NOTES.md. Runs no tests and no benchmarks, and never modifies the target repo.
---

# Fix Candidate

Turns one candidate from a finished Spotlights run into a reviewable patch,
implemented **in this session** so you can steer it, interrupt it, and ask why
mid-change. The batch equivalent is `spotlights-engine fix`, which runs a
nested `claude -p` you cannot influence — reach for this skill when you want to
be in the loop on one candidate.

## What this does not do

**No verification happens here.** No tests are run, no benchmarks are measured,
no build is attempted. Three reasons, all of them load-bearing:

- The machine may lack the hardware. A recorded `performance_oracle: TTFT, TPOT`
  is a metric *name* — no harness, no workload, no baseline. A fabricated
  number is worse than none.
- The recorded correctness oracle is a best-effort regex over the candidate's
  LLM-written rationale. `pytest tests/…` reads as authoritative and is a regex
  hit on a sentence; nothing has checked that the file exists.
- The worktree is a fresh detached checkout with no virtualenv, no build
  artifacts, and no compiled extensions. For a repo like vLLM it cannot execute
  anything.

The oracles still travel with the patch, verbatim, as the verification recipe
for whoever has the hardware. Never claim a result you did not measure.

## Procedure

1. **Resolve the run, repo, and candidate.**
   - Ask for the run directory (the folder holding `result.json`) and the target
     repo path if they are not already obvious from the conversation.
   - If the user did not name a candidate, read `<run>/sorted/sorted_candidates.json`
     (or `sorted_candidates.md`) and show the ranked list, then ask which one.
     **This is the only interactive step before the work starts.**

2. **Get the prompt and a validated worktree** — one command does both:

   ```bash
   spotlights-engine fix --print-prompt \
     --result "<run-dir>" --repo "<repo>" --candidate "<cand-id>"
   ```

   It prints `CANDIDATE:`, `MODULE:`, `BASE:`, `WORKTREE:`, then `PROMPT:`
   followed by the prompt body. The worktree it names is a detached checkout at
   `BASE`, already validated against the candidate's recorded symbol and line
   range, and is **left in place for you**.

   If the command fails with a staleness error, stop. Tell the user the repo has
   drifted from the run and print the commit to check out — the base commit is
   in the run's `run_manifest.json` under `target.commit_sha`. Do not work around
   the gate; with no tests being run it is the only correctness check there is.

3. **Implement the change in the worktree**, following the printed prompt.
   Edit only the in-scope files it lists. `cd` into `WORKTREE` — everything you
   do happens there, never in the user's checkout, which may be dirty and is
   none of your business.

4. **Collect the artifacts.** From the worktree:

   ```bash
   cd "<WORKTREE>"
   git add -N .                 # REQUIRED: without it, files you ADDED vanish from the diff
   git diff > "<run-dir>/fix/<module-slug>/<cand-id>/fix.patch"
   ```

   Create the output directory first. `<module-slug>` is the module's slash-form
   qualified name with `/` replaced by `_` (e.g. `v1/attention` → `v1_attention`).
   Then write `FIX-NOTES.md` beside the patch containing:

   - candidate id, module, objective, and the **base commit** from `BASE:`
   - the in-scope files with their line ranges
   - what you changed and why
   - the findings you used, with their URLs
   - **the oracles verbatim**, correctness commands and performance metrics
   - an explicit statement that nothing was verified here
   - the apply-and-verify recipe:

     ```bash
     git -C <repo> checkout <BASE>
     git -C <repo> apply --check "$PWD/fix.patch" && git -C <repo> apply "$PWD/fix.patch"
     <the recorded correctness oracle>
     ```

   Recording the base commit is not optional: `git diff` embeds no base, and
   applied to the wrong commit the patch either fails or misapplies.

5. **Remove the worktree** — always, including when you produced no patch:

   ```bash
   git -C "<repo>" worktree remove --force "<WORKTREE>"
   git -C "<repo>" worktree prune
   ```

   Skipping the prune leaves a stale entry in `.git/worktrees`.

6. **Report** where the artifacts landed, whether a patch was produced, and the
   apply-and-verify commands.

## If the change cannot be made

Write `FIX-NOTES.md` explaining why — the scope is wrong, the proposal needs a
file outside it, the research does not actually support the change — and
produce no patch. A missing patch is a fine outcome. A patch that cannot be
trusted is not. There is no plan-approval step in this skill; the patch itself
is the reviewable artifact, and nothing is applied until a human applies it.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/one_shot_fix/test_skill_install.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add templates/commands/fix-candidate/ tests/unit/one_shot_fix/test_skill_install.py
git commit -m "feat(one-shot-fix): add the /spotlights-fix-candidate bundled skill"
```

---

### Task 9: Documentation

**Files:**
- Create: `docs/one-shot-fix.md`
- Modify: `README.md:24` (nav), `README.md:100` (command table), `README.md` Skills table, `README.md:209` (More list)
- Modify: `docs/prep-evolve.md` (pointer to the cheap arm)

**Interfaces:**
- Consumes: the CLI surface from Task 7 and the skill from Task 8. No code changes.

- [ ] **Step 1: Write `docs/one-shot-fix.md`**

```markdown
# One-shot fix (`fix`)

`prep-evolve` is the expensive arm: it hands a candidate to an evolutionary
search, which needs a project-specific fitness function before it produces
anything meaningful. Many candidates don't warrant a search. They warrant *one
attempt*: read the candidate, read the research behind it, implement the
change, hand back something reviewable.

`spotlights-engine fix` is that arm. Per candidate it creates a throwaway
detached git worktree at the base commit, validates the candidate against it,
runs one `claude -p` session inside it, and writes `fix.patch` plus
`FIX-NOTES.md`. Your checkout is never modified, and a dirty working tree is
irrelevant.

← Back to [README](../README.md) · The other arm: [prep-evolve](prep-evolve.md)

## What is not verified

**Nothing.** No tests are run, no benchmarks are measured, no build is
attempted. This is a design decision, not a gap:

- **The machine may not be able to.** A candidate whose recorded
  `performance_oracle` is `TTFT, TPOT` names *metrics* — there is no harness,
  no workload, and no baseline. Measuring them needs a GPU and a serving
  benchmark. A laptop cannot produce that number, and a fabricated one is worse
  than none.
- **The correctness oracle is a prose regex.** It is scraped from the
  candidate's LLM-written `evolve_rationale` by matching `pytest`-shaped strings
  and test-file-shaped paths. Nothing checks that the file exists.
  `pytest tests/v1/worker/test_gpu_model_runner.py` reads as authoritative and
  is a regex hit on a sentence.
- **A fresh worktree cannot run them anyway.** It has no venv, no build
  artifacts, and no compiled extensions.

The oracles still travel with the patch, verbatim, in `FIX-NOTES.md` — as the
verification recipe for whoever has the hardware. The deliverable is *a
proposal faithfully implemented and documented*, not a measured win. Use an
evolver when you need the measured win.

## Usage

```bash
# one candidate
spotlights-engine fix --result ./spotlights-out --repo ../vllm \
  --candidate cand-vllm_v1_kv_offload-0002

# the top 5 of a ranking (artifacts still land under the run dir)
spotlights-engine fix --result ./spotlights-out/sorted --repo ../vllm --top-n 5

# every candidate in the run
spotlights-engine fix --result ./spotlights-out --repo ../vllm
```

`--result` accepts any run artifact and self-locates the rest — a run
directory, a `result.json`, an `index.md`, a `sorted/` directory, or a
`sorted_candidates.{json,md}` — exactly as `prep-evolve` does.

Unlike `prep-evolve`, `--repo` **must** point at a real git checkout. `fix`
fails loudly otherwise: a worktree needs a commit, and a patch without a
recorded base is not applicable.

| Flag | Purpose |
|---|---|
| `--result` | The finished run (any artifact of it). Required. |
| `--repo` | Target repo path. Must be a git checkout. Wins over `--index`. |
| `--candidate` | One candidate id. Omit to sweep every candidate. |
| `--out` | Artifacts base dir. Default: the run directory. |
| `--top-n` | With a sorted `--result`: only the top N ranked candidates. |
| `--max-turns` | Agent turn cap per candidate (default 40). |
| `--wallclock` | Wall-clock cap in seconds per candidate (default 1800). |
| `--print-prompt` | Create and validate the worktree, print it and the prompt, exit. |

### Failure semantics

A single explicit `--candidate` fails loudly. A sweep records a per-candidate
skip with a reason and continues — the same shape `prep-evolve` uses. Worktrees
are created and removed one at a time, and removal happens in a `finally`, so a
timed-out or crashed run leaves nothing behind in `.git/worktrees`.

## The staleness gate

Before the agent sees anything, the candidate is validated **against the
worktree**: path containment, file existence, line bounds, excerpt sha256, and
a two-part symbol heuristic (an identifier token of the recorded symbol within
5 lines of the recorded range, and every container of a qualified name still
present in the file).

Validating inside the worktree rather than at `--repo` is deliberate — the
bytes validated are then exactly the bytes the agent edits, the recorded hash
is truthful, and your checkout may be dirty while a fix runs.

With no tests being run, **this gate is the only correctness check in the
design.** It is what stops the agent editing the wrong function after a file
has drifted. If it fails, check out the commit the run targeted (recorded in
`run_manifest.json` under `target.commit_sha`) and try again.

## Artifacts

```
<run-dir>/fix/<module-slug>/<candidate-id>/
├── fix.patch        # git diff against the base commit, SHA in a header comment
└── FIX-NOTES.md     # the travelling documentation
```

This mirrors the `evolve/<module-slug>/<candidate-id>/<evolver>/` layout.
`fix.patch` is a `git diff`, not `format-patch`: the latter needs a commit, and
the repo stays untouched. When the agent makes no in-scope edit — a legitimate
outcome — only `FIX-NOTES.md` is written, saying why.

`FIX-NOTES.md` records the candidate, module, objective, **base commit**,
in-scope files with line ranges, what changed and why, the findings used with
URLs, the oracles verbatim, an explicit no-verification statement, and:

```bash
git -C <repo> checkout <base-sha>
git -C <repo> apply --check "$PWD/fix.patch" && git -C <repo> apply "$PWD/fix.patch"
pytest tests/v1/worker/test_gpu_model_runner.py   # ← the recorded oracle
```

`git -C <repo> apply -3 "$PWD/fix.patch"` falls back to a three-way merge if the patch does not apply
cleanly; `patch -p1 < fix.patch` works without git.

## The interactive front door: `/spotlights-fix-candidate`

The stage runs a nested `claude -p` you cannot influence. When you want to be
in the loop on one candidate — steer it, interrupt it, ask why mid-change —
use the bundled skill instead (install it with `spotlights-engine init`):

```
/spotlights-fix-candidate
```

It picks a candidate, calls `spotlights-engine fix --print-prompt` to get the
prompt *and* a created, validated worktree in one call, does the work
in-session, then collects the same two artifacts. The prompt is shared between
the two paths, so they cannot drift on the part that matters.
```

- [ ] **Step 2: Update `README.md`**

Nav (line 24) — add after the `prep-evolve` link:

```html
  <a href="docs/one-shot-fix.md">one-shot fix</a> ·
```

Command table (line 100) — replace that row with:

```markdown
| `spotlights-engine` | Main engine: structural map → candidates → proposals. Subcommands: `doctor`, `init`, `prep-evolve`, `fix`. |
```

Skills table — add a row after the `/spotlights-share-candidates` row:

```markdown
| `/spotlights-fix-candidate` | Implements one candidate as a reviewable patch, in-session, in a throwaway git worktree. Produces `fix.patch` + `FIX-NOTES.md`. Runs no tests and no benchmarks; your checkout is never modified. |
```

More list (line 209) — add immediately after the `prep-evolve` bullet:

```markdown
- **[One-shot fix (`fix`)](docs/one-shot-fix.md)** — the cheap arm: one Claude Code session turns a candidate into a reviewable patch plus its verification recipe. No fitness loop, no evaluator to write.
```

- [ ] **Step 3: Add the pointer in `docs/prep-evolve.md`**

Immediately after the `← Back to [README](../README.md)` line, insert:

```markdown
> [!TIP]
> **Not every candidate needs a search.** Every bundle below ships with a
> deliberately unfinished evaluator (the "evaluation gap"), and completing it is
> project-specific work. When a candidate warrants *one attempt* rather than an
> evolutionary search, [`spotlights-engine fix`](one-shot-fix.md) is the cheap
> arm: one Claude Code session in a throwaway worktree produces `fix.patch` plus
> the recorded verification recipe — a proposal faithfully implemented, with no
> fitness loop. Use an evolver when you need a *measured* win.
```

- [ ] **Step 4: Verify the docs are consistent with the shipped CLI**

Run: `uv run spotlights-engine fix --help`
Expected: every flag in the `docs/one-shot-fix.md` table appears in the help output, with matching defaults (`--max-turns` 40, `--wallclock` 1800).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest; uv run ruff check src/spotlights_engine/one_shot_fix/ src/spotlights_engine/cli.py tests/unit/one_shot_fix/`

Expected: no regressions in `tests/unit/prep_evolve/` or the init-skills tests, and ruff reports `All checks passed!` on the scoped paths.

> **Two known baselines, so you can tell a regression from the status quo:**
> `uv run pytest` reports **3 pre-existing failures** in `module_knowledge` and
> `module_deep_research`, unrelated to this work — exactly those three are
> expected. `uv run ruff check src/ tests/` reports ~89 pre-existing errors in
> `signal_pipeline`, `schemas`, `doctor`, `validation`, and older test modules.
> Neither baseline is this feature's to fix; both are why the commands above are
> scoped and not `&&`-chained.

- [ ] **Step 6: Commit**

```bash
git add docs/one-shot-fix.md docs/prep-evolve.md README.md
git commit -m "docs(one-shot-fix): document the fix stage, the skill, and the cheap-arm framing"
```

---

## Self-Review

**Spec coverage:**

| Design section | Covered by |
|---|---|
| Architecture: `one_shot_fix/prompts.py` as single source of truth | Task 2 |
| Reuse of `prep_evolve` (`resolve`, `extract`, `spec`, `digest`, `validate_target`) | Tasks 2, 5, 6 — imports only, no move to `candidate_context/` |
| Shared parts / collection duplicated in the skill | Task 3 (Python) + Task 8 (skill markdown), both spelled out |
| `fix` step 1 — resolve | Task 6, `_select` + `resolve_result_location` |
| `fix` step 2 — capture base SHA, fail without git | Task 3 `require_git_repo`, Task 6, tested in Tasks 3, 6, 7 |
| `fix` step 3 — `worktree add --detach` | Task 3, `test_worktree_is_detached_and_creates_no_branch` |
| `fix` step 4 — validate against the worktree | Task 6, `test_validation_runs_against_the_worktree_not_a_dirty_repo` |
| `fix` step 5 — build the prompt | Task 2 |
| `fix` step 6 — `claude -p` with env scrub, max_turns, wallclock, stream-json usage | Task 4 |
| `fix` step 7 — `git add -N` → diff → notes | Task 3 + Task 5 + Task 6 |
| `fix` step 8 — cleanup in a `finally` + prune | Task 3, Task 6 (`test_worktree_is_removed_when_the_runner_raises`) |
| Failure semantics (raise vs skip) | Task 6 |
| `--print-prompt` runs steps 1-5, leaves the worktree | Tasks 6, 7 |
| Batching `--candidate` / `--top-n` | Tasks 6, 7 |
| `/spotlights-fix-candidate` skill, 6 steps, no plan approval | Task 8 |
| Artifacts layout, `fix.patch` header, `FIX-NOTES.md` contents | Tasks 5, 6 |
| Tests listed in the design (prompts, validation ordering, batch loop, collection, claude_exec) | Tasks 2, 3, 4, 6 |
| Docs: `docs/one-shot-fix.md`, README row, `prep-evolve.md` pointer | Task 9 |

**Type consistency:** `CHANGE_SUMMARY_NAME` is defined once in `prompts.py` and imported by `worktree.py`; `collect_patch` returns `(patch, summary)` everywhere; `FixRunResult` is the runner's return type in the real runner, the fake in `_fixtures.py`, and the injected fakes in `test_api.py`; `Worktree.parent` (not `_parent`) is the field name used by `create_worktree` and `remove_worktree`; `render_fix_notes` keyword names match `_write_artifacts`'s call site exactly.

**Known wrinkle carried into Task 4 deliberately:** `_clean_env()` strips every `SPOTLIGHTS_`-prefixed variable, so the argv-recorder shim must use `TEST_ARGV_FILE`, not `SPOTLIGHTS_TEST_ARGV_FILE`. Task 4 Step 2 makes that reconciliation an explicit step rather than a silent trap.
