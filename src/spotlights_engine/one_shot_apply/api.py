"""`one_shot_apply(input, config) -> OneShotApplyResult`.

Per candidate: resolve → capture base sha → worktree → **validate inside the
worktree** → build prompt → run `claude -p` → collect patch + notes → remove
the worktree in a `finally`.

Two orchestration decisions carry the design's weight:

- **Validation runs against the worktree, not `--repo`.** The bytes validated
  are then exactly the bytes the agent edits, the recorded excerpt hash is
  truthful, and your own checkout may be dirty while an apply runs. `prep-evolve`
  has no such concern because it validates and points the evolver at the same
  path; the worktree introduces the asymmetry. With no tests being run, this
  gate is the only correctness check in the design.
- **`print_prompt` runs steps 1-5 and stops**, leaving the worktree in place.
  That is what `/spotlights-apply-candidate` consumes: it gets a validated
  worktree and the prompt in one call, so the staleness gate is never
  reimplemented in markdown.

Failure semantics follow `prep_evolve`'s batch loop in shape — a single
explicit `--candidate` raises; a sweep records a per-candidate skip and
continues — but the sweep's net is deliberately wider: it records *any*
`Exception`, not only the two expected error hierarchies. Every candidate in a
`apply` sweep costs a paid agent session, so an unexpected exception type on
candidate 2 must not discard the eight that have not run yet. `prep_evolve`
can afford the narrow handler because re-running it is free.
"""

from __future__ import annotations

import contextlib
import shlex
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.one_shot_apply.claude_exec import (
    ApplyRunResult,
    ensure_claude_available,
    run_apply_claude,
)
from spotlights_engine.one_shot_apply.errors import (
    ArtifactWriteError,
    ClaudeUnavailableError,
    OneShotApplyError,
)
from spotlights_engine.one_shot_apply.notes import render_apply_notes
from spotlights_engine.one_shot_apply.prompts import build_apply_prompt
from spotlights_engine.one_shot_apply.scope import out_of_scope_files
from spotlights_engine.one_shot_apply.worktree import (
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

ClaudeRunner = Callable[..., ApplyRunResult]

PATCH_NAME = "apply.patch"
NOTES_NAME = "APPLY-NOTES.md"


class OneShotApplyInput(BaseModel):
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


class OneShotApplyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captured_at: str | None = None  # injected timestamp; default = now (UTC)


class ApplyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    module_qualified_name: str
    path: str
    files: list[str]
    patch_produced: bool
    base_sha: str
    usage: AgentUsage | None = None
    out_of_scope_files: list[str] = Field(default_factory=list)
    # Set when the diff could not be taken at all. `patch_produced` is False in
    # that case too, but for a completely different reason — "the agent made no
    # edit" vs "what the agent did is unknown and unrecoverable" — so a reader
    # (and the CLI) must not treat the two as the same outcome.
    collection_error: str | None = None


class PromptPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    module_qualified_name: str
    worktree: str
    worktree_parent: str
    base_sha: str
    prompt: str


class SkippedApply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    candidate_id: str | None = None
    module_qualified_name: str | None = None


class OneShotApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patches: list[ApplyArtifact] = Field(default_factory=list)
    prompts: list[PromptPreview] = Field(default_factory=list)
    skipped: list[SkippedApply] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _apply_dir(base: Path, qn: str, candidate_id: str) -> Path:
    """`apply/<module-slug>/<candidate-id>/`, mirroring the evolve/ layout."""
    return base / "apply" / slug_for(qn) / candidate_id


def _select(
    input: OneShotApplyInput,
    loaded: LoadedResult,
    ranking: Path | None,
    skipped: list[SkippedApply],
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
                skipped.append(SkippedApply(reason=str(exc), candidate_id=cid))
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
    collect_error: str | None = None,
) -> tuple[list[str], bool, list[str]]:
    """Write `apply.patch` (when non-empty) and `APPLY-NOTES.md`.

    Returns `(files, produced, out_of_scope)` — `out_of_scope` is computed
    once here, from `manifest` against `spec.targets`, and threaded both into
    the notes and back to the caller for `ApplyArtifact`, so the notes, the
    artifact, and the CLI warning are one verdict rather than three.

    `collect_error` is written even though it produces no patch: a session
    whose diff failed still gets a `APPLY-NOTES.md`, saying exactly that. It is
    the one case where the notes must not read "no patch was produced" — the
    agent may well have edited files, and they are gone with the worktree.

    Every filesystem write is wrapped: an `OSError` becomes an
    `ArtifactWriteError`, which the batch loop already knows how to record as
    a per-candidate skip. Left bare, it would propagate past that loop's
    handler and abandon every candidate still queued.
    """
    patch_produced = bool(patch.strip())
    out_of_scope = out_of_scope_files(spec, manifest) if patch_produced else []
    notes = render_apply_notes(
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
        collect_error=collect_error,
    )

    # Shell-quoted for the same reason as the recipe in APPLY-NOTES.md: these
    # header lines are meant to be copy-pasted, and a repo path containing a
    # space would otherwise be split by the shell.
    repo_arg = shlex.quote(str(repo_path))
    header = (
        f"# spotlights one-shot apply\n"
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
            # A rerun that concludes no change must not leave a previous session's
            # patch behind — the notes below explicitly deny one exists.
            (out_dir / PATCH_NAME).unlink(missing_ok=True)

        (out_dir / NOTES_NAME).write_text(notes, encoding="utf-8")
        files.append(NOTES_NAME)
    except OSError as exc:
        # Neither `write_bytes` nor `write_text` is atomic: an `ENOSPC` partway
        # through a multi-MB patch leaves a truncated `apply.patch` on disk, and
        # the same failure on the notes leaves a truncated `APPLY-NOTES.md`.
        # Raising over either and leaving it there is the worst of the options.
        # The patch is the sharper edge — the directory would hold a file named
        # `apply.patch` that `git apply` will reject, or (with bad luck at a hunk
        # boundary) apply *partially* — but the notes matter too: a
        # half-written one can lose the very "Applying and verifying" section
        # that says which commit the patch belongs to, and this directory is
        # also where a *previous* run's artifacts sit. Leaving last run's notes
        # next to no patch is a directory that documents a patch which is not
        # there. Remove both, so the failure reads as "nothing here" — the same
        # invariant the empty-patch branch above maintains.
        for name in (PATCH_NAME, NOTES_NAME):
            with contextlib.suppress(OSError):
                (out_dir / name).unlink(missing_ok=True)
        raise ArtifactWriteError(
            f"could not write apply artifacts for {sel.candidate.id} to {out_dir}: {exc}"
        ) from exc

    return sorted(files), patch_produced, out_of_scope


def _process_candidate(
    *,
    sel: CandidateSelection,
    loaded: LoadedResult,
    repo_path: Path,
    base: Path,
    base_sha: str,
    input: OneShotApplyInput,
    captured_at: str,
    direction: Direction,
    claude_runner: ClaudeRunner,
    result: OneShotApplyResult,
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
        prompt = build_apply_prompt(spec=spec, worktree=worktree.path)

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

    out_dir = _apply_dir(base, sel.qn, sel.candidate.id)
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
        collect_error=collection.collect_error,
    )
    result.patches.append(
        ApplyArtifact(
            candidate_id=sel.candidate.id,
            module_qualified_name=sel.qn,
            path=str(out_dir),
            files=files,
            patch_produced=patch_produced,
            base_sha=base_sha,
            usage=run.usage,
            out_of_scope_files=out_of_scope,
            collection_error=collection.collect_error,
        )
    )


def one_shot_apply(
    input: OneShotApplyInput,
    config: OneShotApplyConfig | None = None,
    *,
    claude_runner: ClaudeRunner | None = None,
) -> OneShotApplyResult:
    """Turn one (or every) candidate into a patch plus its notes."""
    if input.print_prompt and input.candidate is None:
        raise SelectionError(
            "--print-prompt requires --candidate: without one, a sweep would "
            "leave one worktree per candidate registered in the target repo "
            "with nothing to clean them up"
        )

    config = config or OneShotApplyConfig()
    captured_at = config.captured_at or _now_iso()
    runner = claude_runner or run_apply_claude
    result = OneShotApplyResult()

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
        except (PrepEvolveError, OneShotApplyError) as exc:
            if not batch:
                raise
            result.skipped.append(
                SkippedApply(
                    reason=str(exc),
                    candidate_id=sel.candidate.id,
                    module_qualified_name=sel.qn,
                )
            )
        except Exception as exc:  # noqa: BLE001 — see below
            # An unexpected exception type (a pydantic ValidationError, an
            # OSError from somewhere not already wrapped, a bug in this module)
            # must not abandon the candidates still queued: each one of those
            # costs a paid agent session, and in a sweep they have not run yet.
            # `prep_evolve`'s loop is deliberately narrow because re-running it
            # is free; a `apply` sweep that dies on candidate 2 of 10 throws away
            # eight sessions' worth of work that were never started, and the
            # one already-finished patch is written before this point, so it
            # survives.
            #
            # `Exception`, not `BaseException`: KeyboardInterrupt and
            # SystemExit still abort the sweep immediately, which is what a
            # Ctrl-C must do. And still `raise` for a single `--candidate`,
            # where there is nothing to protect and the traceback is the
            # useful output.
            if not batch:
                raise
            result.skipped.append(
                SkippedApply(
                    # Worded so it cannot be read as a routine skip: an
                    # unexpected type here means a bug, not a candidate that
                    # legitimately could not be fixed.
                    reason=(
                        f"unexpected {type(exc).__name__} — this is a bug, not a "
                        f"normal skip; the remaining candidates were still "
                        f"attempted: {exc}"
                    ),
                    candidate_id=sel.candidate.id,
                    module_qualified_name=sel.qn,
                )
            )

    return result


__all__ = [
    "ApplyArtifact",
    "OneShotApplyConfig",
    "OneShotApplyInput",
    "OneShotApplyResult",
    "PromptPreview",
    "SkippedApply",
    "one_shot_apply",
]
