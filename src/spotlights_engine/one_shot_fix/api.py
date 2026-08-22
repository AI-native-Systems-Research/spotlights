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

import shlex
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.one_shot_fix.claude_exec import (
    FixRunResult,
    ensure_claude_available,
    run_fix_claude,
)
from spotlights_engine.one_shot_fix.errors import (
    ArtifactWriteError,
    ClaudeUnavailableError,
    OneShotFixError,
)
from spotlights_engine.one_shot_fix.notes import render_fix_notes
from spotlights_engine.one_shot_fix.prompts import build_fix_prompt
from spotlights_engine.one_shot_fix.scope import out_of_scope_files
from spotlights_engine.one_shot_fix.worktree import (
    FileChange,
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
    out_of_scope_files: list[str] = Field(default_factory=list)


class PromptPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    module_qualified_name: str
    worktree: str
    worktree_parent: str
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
    patch: bytes,
    manifest: list[FileChange],
    change_summary: str | None,
    agent_error: str | None,
) -> tuple[list[str], bool, list[str]]:
    """Write `fix.patch` (when non-empty) and `FIX-NOTES.md`.

    Returns `(files, produced, out_of_scope)` — `out_of_scope` is computed
    once here, from `manifest` against `spec.targets`, and threaded both into
    the notes and back to the caller for `FixArtifact`, so the notes, the
    artifact, and the CLI warning are one verdict rather than three.

    Every filesystem write is wrapped: an `OSError` becomes an
    `ArtifactWriteError`, which the batch loop already knows how to record as
    a per-candidate skip. Left bare, it would propagate past that loop's
    handler and abandon every candidate still queued.
    """
    patch_produced = bool(patch.strip())
    out_of_scope = out_of_scope_files(spec, manifest) if patch_produced else []
    notes = render_fix_notes(
        spec=spec,
        candidate_id=sel.candidate.id,
        module_qn=sel.qn,
        base_sha=base_sha,
        repo=repo_path,
        change_summary=change_summary,
        patch_produced=patch_produced,
        agent_error=agent_error,
        manifest=manifest,
        out_of_scope=out_of_scope,
    )

    # Shell-quoted for the same reason as the recipe in FIX-NOTES.md: these
    # header lines are meant to be copy-pasted, and a repo path containing a
    # space would otherwise be split by the shell.
    repo_arg = shlex.quote(str(repo_path))
    header = (
        f"# spotlights one-shot fix\n"
        f"# candidate: {sel.candidate.id}\n"
        f"# module:    {sel.qn}\n"
        f"# repo:      {repo_path}\n"
        f"# base:      {base_sha}\n"
        f"# apply with (from the directory containing this patch):\n"
        f"#   git -C {repo_arg} checkout {base_sha}\n"
        f'#   git -C {repo_arg} apply "$PWD/{PATCH_NAME}"\n'
    )

    files: list[str] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if patch_produced:
            # `write_bytes`, not `write_text`: `patch` is undecoded git output
            # and must land on disk exactly as git emitted it (see
            # `worktree._run_git_bytes`). Only the header is ours to encode.
            (out_dir / PATCH_NAME).write_bytes(header.encode("utf-8") + patch)
            files.append(PATCH_NAME)
        else:
            # A rerun that concludes no fix must not leave a previous session's
            # patch behind — the notes below explicitly deny one exists.
            (out_dir / PATCH_NAME).unlink(missing_ok=True)

        (out_dir / NOTES_NAME).write_text(notes, encoding="utf-8")
        files.append(NOTES_NAME)
    except OSError as exc:
        raise ArtifactWriteError(
            f"could not write fix artifacts for {sel.candidate.id} to {out_dir}: {exc}"
        ) from exc

    return sorted(files), patch_produced, out_of_scope


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
    # `create_worktree` is called *inside* the `try`, with `worktree` pre-set
    # to None, so there is no window where a worktree exists but the `finally`
    # that removes it has not been registered yet. A KeyboardInterrupt landing
    # in that window used to leak both a stale `.git/worktrees` entry in the
    # target repo and the temp directory holding it — once per interrupted
    # candidate.
    worktree: Worktree | None = None
    keep_worktree = False
    try:
        worktree = create_worktree(repo_path, base_sha)
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
            preview = PromptPreview(
                candidate_id=sel.candidate.id,
                module_qualified_name=sel.qn,
                worktree=str(worktree.path),
                worktree_parent=str(worktree.parent),
                base_sha=base_sha,
                prompt=prompt,
            )
            result.prompts.append(preview)
            # Only set once the handoff has actually succeeded, so a leaked
            # worktree can never coexist with a swallowed exception.
            keep_worktree = True
            return

        run = claude_runner(
            candidate_id=sel.candidate.id,
            prompt=prompt,
            worktree=worktree.path,
            max_turns=input.max_turns,
            wallclock_s=input.wallclock_s,
        )
        collection = collect_patch(worktree)
    finally:
        if worktree is not None and not keep_worktree:
            remove_worktree(worktree)

    out_dir = _fix_dir(base, sel.qn, sel.candidate.id)
    files, patch_produced, out_of_scope = _write_artifacts(
        out_dir=out_dir,
        spec=spec,
        sel=sel,
        repo_path=repo_path,
        base_sha=base_sha,
        patch=collection.patch,
        manifest=collection.manifest,
        change_summary=collection.change_summary,
        agent_error=run.error,
    )
    result.fixes.append(
        FixArtifact(
            candidate_id=sel.candidate.id,
            module_qualified_name=sel.qn,
            path=str(out_dir),
            files=files,
            patch_produced=patch_produced,
            base_sha=base_sha,
            usage=run.usage,
            out_of_scope_files=out_of_scope,
        )
    )


def one_shot_fix(
    input: OneShotFixInput,
    config: OneShotFixConfig | None = None,
    *,
    claude_runner: ClaudeRunner | None = None,
) -> OneShotFixResult:
    """Turn one (or every) candidate into a patch plus its notes."""
    if input.print_prompt and input.candidate is None:
        raise SelectionError(
            "--print-prompt requires --candidate: without one, a sweep would "
            "leave one worktree per candidate registered in the target repo "
            "with nothing to clean them up"
        )

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

    # Setup-time check, mirroring agent_proposals/proposal_from_finding_creator.
    # Skipped for injected runners (every test) and --print-prompt (runs no
    # agent) — neither needs the `claude` binary on PATH. Runs after the
    # cheaper, more fundamental checks above (result location, repo dir,
    # git-ness) so those errors are never masked by this one — but still
    # before any worktree is created.
    if claude_runner is None and not input.print_prompt:
        try:
            ensure_claude_available()
        except AgentProposalsSetupError as exc:
            raise ClaudeUnavailableError(str(exc)) from exc

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
