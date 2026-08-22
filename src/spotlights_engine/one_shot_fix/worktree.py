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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from spotlights_engine.one_shot_fix.errors import NotAGitRepoError, WorktreeError
from spotlights_engine.one_shot_fix.prompts import CHANGE_SUMMARY_NAME

_GIT_TIMEOUT_S = 120

ChangeKind = Literal["added", "modified", "deleted", "renamed"]


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


def _spawn_git(cwd: Path, args: tuple[str, ...], *, text: bool) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=text,
            errors="replace" if text else None,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorktreeError(f"git {' '.join(args)} failed in {cwd}: {exc}") from exc


def _run_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git and return the completed process, stdout decoded lossily.

    `errors="replace"` keeps a surprising byte from crashing a *metadata*
    read (`rev-parse`, `--name-status`, `--numstat`), where a mangled
    character costs at most a cosmetically wrong manifest row. It must never
    be used for the patch itself — see `_run_git_bytes`.
    """
    completed = _spawn_git(cwd, args, text=True)
    if check and completed.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed in {cwd} "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )
    return completed


def _run_git_bytes(cwd: Path, *args: str, check: bool = True) -> bytes:
    """Run git and return stdout as raw, undecoded bytes.

    Required for `git diff`. A diff's context lines reproduce the file's bytes
    verbatim, and a source file may legally contain bytes that are not valid
    UTF-8. Decoding with `errors="replace"` turns each such byte into U+FFFD
    (`ef bf bd`); writing that back out produces a `fix.patch` whose context
    no longer matches the file it came from, and `git apply` rejects it with
    "patch does not apply". `errors="strict"` is no better — it raises, losing
    the patch entirely. The only correct handling is not to decode: collect
    the diff as bytes and write it with `write_bytes`.
    """
    completed = _spawn_git(cwd, args, text=False)
    if check and completed.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed in {cwd} (exit {completed.returncode}): "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return completed.stdout


def require_git_repo(repo: Path) -> str:
    """Return `repo`'s HEAD sha, or raise `NotAGitRepoError`.

    `prep-evolve` tolerates a non-git target and records a `None` commit. `fix`
    cannot: a worktree needs a commit, and the emitted patch is meaningless
    without a recorded base to apply it to.

    `check=False` suppresses only the return-code check. `_spawn_git` still
    raises `WorktreeError` when the subprocess cannot run at all — git absent
    from `PATH` being the case that matters, since `ensure_repo_dir` has
    already established that `repo` exists. That is the same "no usable git
    checkout here" conclusion, so it is reported as `NotAGitRepoError` rather
    than leaking a lower-level type past this function's documented contract.
    The message distinguishes the two causes, because "not a git checkout" on
    its own would be actively misleading when the checkout is fine and git is
    what is missing.
    """
    try:
        completed = _run_git(repo, "rev-parse", "HEAD", check=False)
    except WorktreeError as exc:
        raise NotAGitRepoError(
            f"could not run git in {repo}: {exc}. `fix` needs a real git repo "
            f"and a working `git` on PATH: it creates a detached worktree at "
            f"the base commit and records that commit in the patch notes."
        ) from exc
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

    `check=False` only suppresses `_run_git`'s return-code check; it does not
    stop `_run_git` from raising `WorktreeError` when the underlying
    `subprocess.run` itself blows up (`OSError`, or a `SubprocessError` such as
    a timeout). Each git call is therefore wrapped individually so no such
    exception escapes this function.
    """
    try:
        _run_git(wt.repo, "worktree", "remove", "--force", str(wt.path), check=False)
    except WorktreeError:
        pass
    try:
        _run_git(wt.repo, "worktree", "prune", check=False)
    except WorktreeError:
        pass
    shutil.rmtree(wt.parent, ignore_errors=True)


@dataclass(frozen=True)
class FileChange:
    """One file's change, derived from the actual diff — never from a declared scope.

    `insertions`/`deletions` are `None` for binary files: `git diff --numstat`
    reports `-` for both rather than a count. `old_path` is set only when
    `change_kind == "renamed"`.

    `counts_known` distinguishes "this file is binary, so line counts do not
    apply" (`binary=True`) from "line counts could not be determined" (a
    degraded record — `counts_known=False`, `binary=False`). Both leave
    `insertions`/`deletions` as `None`; conflating them made a degraded
    record render as a false "binary" claim (see Important 3).
    """

    path: str
    change_kind: ChangeKind
    insertions: int | None
    deletions: int | None
    binary: bool
    counts_known: bool = True
    old_path: str | None = None


@dataclass(frozen=True)
class PatchCollection:
    """Result of `collect_patch`: the raw diff, the agent's rationale, and a manifest.

    `manifest` is the per-file breakdown of exactly what `patch` contains —
    derived from `git diff --name-status`/`--numstat`, not from any declared
    scope, so it can catch a patch that disagrees with what was intended.

    `patch` is deliberately `bytes`, not `str`: the diff must reach disk
    byte-for-byte as git emitted it or it stops applying (see
    `_run_git_bytes`). `change_summary` is `str` — it is the agent's prose,
    destined for a markdown document, where a lossy decode is the right
    trade.

    `collect_error` is set when the diff itself could not be taken (e.g. the
    `_GIT_TIMEOUT_S` timeout fired). It is *not* the same as an empty `patch`:
    an empty patch means the agent made no edit, while `collect_error` means
    what the agent did is unknown and now unrecoverable. Conflating the two
    would make `FIX-NOTES.md` claim "the agent made no in-scope edit" about a
    session whose edits were merely lost.
    """

    patch: bytes
    change_summary: str | None
    manifest: list[FileChange] = field(default_factory=list)
    collect_error: str | None = None


def _split_z(raw: str) -> list[str]:
    """Split `-z`-terminated git output into tokens, dropping the trailing empty one."""
    tokens = raw.split("\0")
    if tokens and tokens[-1] == "":
        tokens = tokens[:-1]
    return tokens


def _parse_name_status_z(raw: str) -> list[tuple[str, str, str | None]]:
    """Parse `git diff --name-status -z` into `(status, path, old_path)` triples.

    With `-z`, a rename/copy record is `status<NUL>old_path<NUL>new_path<NUL>`;
    every other record is `status<NUL>path<NUL>`. `-z` also disables git's
    quoting of paths with spaces or non-ASCII bytes, so no unquoting is needed.

    A trailing record whose path tokens are missing is *dropped*, not
    unpacked. This is the truncation guard: a partial stream (git killed
    mid-write, or a `_GIT_TIMEOUT_S` timeout whose captured stdout stops
    mid-record) used to raise `IndexError` here, and `collect_patch`'s
    fallback re-parses the same raw string — so the second attempt raised
    too and the manifest degraded all the way to `[]`, silently switching
    off the out-of-scope check for that patch. Returning the records that
    did parse keeps the scope check alive for every complete record.
    """
    tokens = _split_z(raw)
    records: list[tuple[str, str, str | None]] = []
    i = 0
    while i < len(tokens):
        status = tokens[i]
        needed = 3 if status[:1] in ("R", "C") else 2
        if i + needed > len(tokens):
            break
        if needed == 3:
            old_path, new_path = tokens[i + 1], tokens[i + 2]
            records.append((status, new_path, old_path))
        else:
            records.append((status, tokens[i + 1], None))
        i += needed
    return records


def _parse_count(token: str) -> tuple[int | None, bool]:
    """Parse one numstat count field. Returns `(value, ok)`.

    `-` is git's own binary sentinel: mapped to `(None, True)` — a known,
    deliberate absence. Anything else that fails to parse as an int degrades
    to `(None, False)` — unknown, never raises. This is the hardening half of
    Critical 1: a malformed count field must degrade a single record, not
    crash the whole collection.
    """
    if token == "-":
        return None, True
    try:
        return int(token), True
    except ValueError:
        return None, False


def _parse_numstat_z(raw: str) -> list[tuple[int | None, int | None, bool, bool]]:
    """Parse `git diff --numstat -z` into `(added, deleted, binary, counts_known)`
    tuples, in record order.

    Binary files report `-` for both counts (`binary=True`, `counts_known=True`
    — the absence is expected, not degraded). A rename/copy record has an
    empty path field followed by two more NUL-separated path tokens (old,
    then new) — this only needs the counts, so those extra tokens are
    skipped; `_parse_name_status_z` is the source of truth for paths.

    `-z` disables git's C-quoting, so a path containing a literal tab (legal
    on POSIX) makes the count line itself contain a tab
    (`"1\\t0\\ttab\\tname.py"`). `split("\\t", maxsplit=2)` is load-bearing:
    it keeps exactly 3 fields regardless of how many tabs are embedded in the
    path, where an unbounded `split("\\t")` raised `ValueError: too many
    values to unpack`. The path field itself is discarded here regardless
    (see above), so no unquoting or further handling of that tab is needed.
    """
    tokens = _split_z(raw)
    records: list[tuple[int | None, int | None, bool, bool]] = []
    i = 0
    while i < len(tokens):
        parts = tokens[i].split("\t", 2)
        if len(parts) != 3:
            # Should be unreachable — every numstat record has at least the
            # two tab-separated counts — but a malformed record degrades
            # rather than crashes, and by itself carries no reliable
            # rename/copy path-token count, so it cannot be paired past this
            # single token.
            records.append((None, None, False, False))
            i += 1
            continue
        added_s, deleted_s, path = parts
        added, added_ok = _parse_count(added_s)
        deleted, deleted_ok = _parse_count(deleted_s)
        binary = added_s == "-" and deleted_s == "-"
        # `counts_known` must imply both counts are present, not merely that
        # each field parsed. Git always writes `-` for *both* counts of a
        # binary file, so a half-`-` record is not reachable from real git
        # output — but `_parse_count("-")` reports `ok=True` (a deliberate
        # absence), so the earlier `binary or (added_ok and deleted_ok)` let
        # a one-sided `-` through as "known" with a `None` count, which
        # `notes._change_row` then rendered as the nonsense `+None/-5`.
        counts_known = binary or (
            added_ok and deleted_ok and added is not None and deleted is not None
        )
        if not counts_known:
            added, deleted = None, None
        records.append((added, deleted, binary, counts_known))
        i += 1 if path != "" else 3
    return records


def _change_kind(status: str) -> ChangeKind:
    code = status[:1]
    if code == "A":
        return "added"
    if code == "D":
        return "deleted"
    if code == "R":
        return "renamed"
    if code == "C":
        # A copy creates a new path; the source is untouched. There is no
        # "copied" kind in the manifest's vocabulary, so this is the closest
        # honest label.
        return "added"
    return "modified"


def _build_manifest(name_status_raw: str, numstat_raw: str) -> list[FileChange]:
    statuses = _parse_name_status_z(name_status_raw)
    counts = _parse_numstat_z(numstat_raw)
    if len(counts) != len(statuses):
        # Shouldn't happen — both come from the same worktree state, queried
        # back to back with no intervening change — but a degraded manifest
        # (paths and kinds right, counts unknown) beats crashing the whole
        # collection over a git output surprise.
        counts = [(None, None, False, False)] * len(statuses)

    manifest: list[FileChange] = []
    # `strict=False`, spelled out rather than implied (ruff's B905 requires an
    # explicit value): the guard above has already equalized the two lengths in
    # *both* of its branches, so a strict `zip` here could never raise. Keeping
    # it would advertise a diagnostic this function does not actually offer,
    # and invite a later reader to delete the guard on the strength of it.
    # The guard is the single defence, and it does not swallow the failure:
    # every degraded record surfaces in `notes._manifest_section` as an
    # explicit "line counts unavailable for N files" line beside the totals,
    # which is what a reader of the notes needs and an exception here — inside
    # the block that must not lose an already-collected patch — is not.
    for (status, path, old_path), (added, deleted, binary, counts_known) in zip(
        statuses, counts, strict=False
    ):
        manifest.append(
            FileChange(
                path=path,
                change_kind=_change_kind(status),
                insertions=added,
                deletions=deleted,
                binary=binary,
                counts_known=counts_known,
                old_path=old_path,
            )
        )
    return manifest


def _summary_is_the_agents(wt: Worktree) -> bool:
    """Did the agent write `CHANGE-SUMMARY.md`, or did the repo supply it?

    The file's existence on disk says nothing on its own: a repo that *tracks*
    a path by that name hands a copy to every worktree created from it, before
    the agent has run at all. Reading that copy would put the repo's own
    content into `FIX-NOTES.md`'s "What changed and why" as though it were the
    agent's rationale — a false attribution in the one document whose entire
    value is being trustworthy about what was and was not done.

    Deliberately compares against `wt.base_sha` rather than using
    `git status`, which compares against HEAD: `claude_exec` permits `git
    reset` inside the worktree, and a `--soft` reset moves HEAD without
    touching the working tree (see `collect_patch`'s docstring). Every
    comparison in this module is anchored to the recorded base for that
    reason.

    Answers "yes" whenever git cannot say. A file present while the base
    commit does not track that path is overwhelmingly the common case, and
    dropping a real rationale is the worse trade: the wrong guess needs three
    coincidences at once — the repo tracks the path, git failed, *and* the
    agent left nothing.
    """
    try:
        tracked = _run_git(
            wt.path, "cat-file", "-e", f"{wt.base_sha}:{CHANGE_SUMMARY_NAME}", check=False
        )
    except WorktreeError:
        return True
    if tracked.returncode != 0:
        # Not in the base commit, so nothing but the agent put it on disk.
        return True
    try:
        # Tracked: only a difference from the committed copy is the agent's
        # doing. `--quiet` exits 1 when they differ, 0 when they match; any
        # other code means git could not tell us, which the `!= 0` treats as
        # "the agent's" per the docstring above.
        diff = _run_git(
            wt.path, "diff", "--quiet", wt.base_sha, "--", CHANGE_SUMMARY_NAME, check=False
        )
    except WorktreeError:
        return True
    return diff.returncode != 0


def collect_patch(wt: Worktree) -> PatchCollection:
    """Return the patch, the agent's rationale, and a per-file manifest.

    Never raises for a git failure: a diff that cannot be taken comes back as
    `collect_error` on the result, because the caller's `finally` destroys the
    worktree and an exception here would take the candidate's whole record with
    it (see the comment on the diff below).

    Order matters:
    1. read + delete `CHANGE-SUMMARY.md` *if the agent wrote it* (see
       `_summary_is_the_agents`), so the agent's rationale reaches FIX-NOTES.md
       but never appears in the patch (or the manifest), then restore it from
       `base_sha` in case the repo tracked that path;
    2. `git add -N .` so *added* files show up in the diff;
    3. `git diff <base_sha>` (patch bytes) plus `git diff <base_sha>
       --name-status -z` and `git diff <base_sha> --numstat -z` (the
       manifest) — diffed against `wt.base_sha`, the exact commit the
       worktree was created at and the same value recorded in `fix.patch`'s
       header and `FIX-NOTES.md`.

    `wt.base_sha` (not a bare `git diff`, which is index-vs-worktree only,
    and not `HEAD`) is load-bearing on two counts:

    - `git add -N .`'s pathspec (`.`) does not just intent-to-add new
      paths — for a path that matches and no longer exists on disk, plain
      `git add` (this is standard `add` semantics since git 2.0, unrelated to
      `-N`) stages its deletion *fully*, not as intent-to-add. A bare
      `git diff` (index vs worktree) then finds nothing to show for that
      path, so a deleted file — and the delete-half of a rename, which is a
      deletion plus an addition of matching content — silently vanishes.
      Diffing against a commit compares the worktree directly against that
      commit regardless of what got staged along the way, so it sees the
      deletion (and the rename) correctly while still matching a bare
      `git diff` byte-for-byte for the modified/added cases that were
      already covered. This still requires `git add -N .` first: `git diff
      <commit>` does not surface untracked files on its own.
    - `claude_exec.py` deliberately permits `git reset` from inside the
      worktree (see its module docstring): on a detached worktree it only
      rewrites that worktree's own private HEAD/index, so it "cannot
      escape". That reasoning holds only if the diff base doesn't move with
      it. `git reset --soft` moves HEAD without touching the working tree,
      so a plain `git diff HEAD` after such a reset silently changes what
      the diff (and thus `fix.patch`) is relative to — while the header and
      `FIX-NOTES.md` still (correctly) claim `base_sha`. Diffing against
      `wt.base_sha` directly is immune to HEAD moving underneath it.
    """
    summary_path = wt.path / CHANGE_SUMMARY_NAME
    summary: str | None = None
    # `_summary_is_the_agents` and not a bare `is_file()`: see that function.
    # When the file is the repo's own untouched copy, every step below is
    # skipped — there is nothing to read, and nothing for the collector to
    # remove and then put back.
    if summary_path.is_file() and _summary_is_the_agents(wt):
        try:
            summary = summary_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            # Not left as `None`: that renders as "the agent left no summary",
            # a claim about the agent's behaviour that is false here. Nor
            # raised — see the note on losing the whole record below.
            summary = f"_({CHANGE_SUMMARY_NAME} exists but could not be read: {exc})_"
        try:
            summary_path.unlink()
        except OSError:
            # The file stays and the diff reports it as a change. A spurious
            # hunk that the manifest flags is a far smaller loss than raising,
            # and there is no third option available here.
            pass
        else:
            # Removing the rationale file keeps it out of the patch — unless
            # the target repo *tracks* a file by that name, in which case the
            # unlink is a deletion the diff below reports faithfully, and
            # `fix.patch` ships a spurious hunk deleting one of the repo's own
            # files. That is precisely the class of unintended edit the
            # manifest exists to catch, introduced by the collector itself.
            # Restoring the committed copy undoes exactly the collector's own
            # edit and nothing else.
            #
            # `check=False` suppresses only the return-code check — in the
            # normal case the file was untracked, so this pathspec matches
            # nothing and git exits 1, a no-op rather than a failure. It does
            # not stop `_spawn_git` from raising when the subprocess itself
            # cannot run (an `OSError`, or a `_GIT_TIMEOUT_S` timeout), so the
            # call is wrapped for the same reason `remove_worktree` wraps
            # each of its own: this function must not raise for a git failure
            # (see its docstring). Letting one escape here would destroy the
            # whole record of an agent session that has already finished and
            # already been paid for, over a best-effort restore.
            try:
                _run_git(
                    wt.path, "checkout", wt.base_sha, "--", CHANGE_SUMMARY_NAME, check=False
                )
            except WorktreeError:
                pass

    # `git add -N .` and the diff are the two steps that can fail outright
    # (a 120 s `_GIT_TIMEOUT_S` timeout on a huge tree, a git that dies). The
    # caller runs this inside the `try` whose `finally` destroys the worktree,
    # so raising here would take the whole candidate's record down with the
    # worktree: no `FIX-NOTES.md`, no recorded `run.error`, no usage numbers,
    # and in a sweep, one skip line for a session that was already paid for.
    # Report the failure instead, keeping the agent's own prose (already read
    # above), and let the caller write notes that say plainly what is and is
    # not known.
    try:
        _run_git(wt.path, "add", "-N", ".")
        patch = _run_git_bytes(wt.path, "diff", wt.base_sha)
    except WorktreeError as exc:
        return PatchCollection(
            patch=b"", change_summary=summary, manifest=[], collect_error=str(exc)
        )

    # The patch above is the deliverable; nothing past this point may be
    # allowed to lose it. A manifest-parsing surprise (git output nobody
    # anticipated) degrades to the best manifest still recoverable — paths
    # and kinds from the simpler, NUL-delimited name-status output, counts
    # marked unknown — rather than raising and discarding the patch that was
    # already collected (see Critical 1).
    manifest: list[FileChange] = []
    name_status_raw = ""
    try:
        name_status_raw = _run_git(
            wt.path, "diff", wt.base_sha, "--name-status", "-z"
        ).stdout
        numstat_raw = _run_git(wt.path, "diff", wt.base_sha, "--numstat", "-z").stdout
        manifest = _build_manifest(name_status_raw, numstat_raw)
    except Exception:
        try:
            manifest = [
                FileChange(
                    path=path,
                    change_kind=_change_kind(status),
                    insertions=None,
                    deletions=None,
                    binary=False,
                    counts_known=False,
                    old_path=old_path,
                )
                for status, path, old_path in _parse_name_status_z(name_status_raw)
            ]
        except Exception:
            manifest = []

    return PatchCollection(patch=patch, change_summary=summary, manifest=manifest)


__all__ = [
    "FileChange",
    "PatchCollection",
    "Worktree",
    "collect_patch",
    "create_worktree",
    "remove_worktree",
    "require_git_repo",
]
