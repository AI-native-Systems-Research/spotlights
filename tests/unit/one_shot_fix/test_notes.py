"""FIX-NOTES.md content: oracles verbatim, base commit, no-verification statement."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix.notes import out_of_scope_files, render_fix_notes
from spotlights_engine.one_shot_fix.worktree import FileChange
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
        "manifest": [
            FileChange(
                path=CAND_FILE,
                change_kind="modified",
                insertions=3,
                deletions=1,
                binary=False,
            )
        ],
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
    assert f'git -C {repo} apply --check "$PWD/fix.patch"' in notes
    assert f'git -C {repo} apply "$PWD/fix.patch"' in notes
    assert f"git -C {repo} apply -3" in notes
    assert "patch -p1 < fix.patch" in notes
    assert "repo root" in notes
    # Prose anchoring $PWD to where fix.patch (and this file) actually live.
    assert "directory containing" in notes


def test_apply_recipe_is_not_location_dependent(spec_and_repo) -> None:
    """Regression guard against BOTH previously-shipped broken forms.

    Form 1 (bare, no `-C`): `git apply --check fix.patch` either fails with
    "not a git repository" when run from wherever FIX-NOTES.md was saved, or
    — worse — silently applies against whatever unrelated git repo happens to
    contain that directory.

    Form 2 (`-C` but a bare patch path): `git -C {repo} apply --check
    fix.patch` chdirs into `{repo}` first, so git then resolves the
    *relative* `fix.patch` under `{repo}` — not under the artifact directory
    where the file actually lives. Verified in a scratch repo: this fails
    with "can't open patch 'fix.patch': No such file or directory" (exit
    128) when run from the artifact directory. This was the state after the
    first "fix" of this finding — neither form worked.

    The correct form keeps `-C {repo}` for repo targeting and anchors the
    patch path with `$PWD` so it resolves regardless of git's chdir.
    """
    spec, repo = spec_and_repo
    notes = _notes(spec_and_repo)
    assert "git apply --check fix.patch" not in notes
    assert "git apply fix.patch" not in notes
    assert f"git -C {repo} apply --check fix.patch" not in notes
    assert f"git -C {repo} apply fix.patch" not in notes


def test_the_emitted_apply_recipe_actually_applies_from_the_artifact_directory(
    spec_and_repo, tmp_path: Path
) -> None:
    """Prove the recipe works: execute the emitted lines against a real repo.

    A recipe nobody executed is how this bug survived two review rounds. This
    builds an independent target repo, produces a real patch against it,
    lands the patch in a separate artifact directory (never the repo itself),
    extracts the `git -C ... checkout` / `git -C ... apply` lines verbatim
    from the rendered notes, and runs them via `subprocess` with the artifact
    directory as cwd — exactly how a human would follow the recipe.
    """
    spec, _unused_repo = spec_and_repo

    target_repo = tmp_path / "target-repo"
    target_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=target_repo, check=True)
    tracked = target_repo / "f.txt"
    tracked.write_text("line1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=target_repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=target_repo,
        check=True,
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=target_repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    tracked.write_text("line1\nline2\n", encoding="utf-8")
    patch_text = subprocess.run(
        ["git", "diff"], cwd=target_repo, capture_output=True, text=True, check=True
    ).stdout
    subprocess.run(["git", "checkout", "-q", "--", "f.txt"], cwd=target_repo, check=True)
    assert tracked.read_text(encoding="utf-8") == "line1\n"

    notes = render_fix_notes(
        spec=spec,
        candidate_id=CAND_ID,
        module_qn="v1/attention",
        base_sha=base_sha,
        repo=target_repo,
        change_summary="did the thing",
        patch_produced=True,
        agent_error=None,
        manifest=[],
    )

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "fix.patch").write_text(patch_text, encoding="utf-8")

    bash_block = notes.split("```bash\n", 1)[1].split("```", 1)[0]
    apply_lines = [
        ln for ln in bash_block.splitlines() if ln.startswith("git -C")
    ]
    assert len(apply_lines) == 2  # the checkout line + the check-and-apply line
    script = "\n".join(apply_lines)

    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=artifact_dir,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"emitted apply recipe failed:\nSCRIPT:\n{script}\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert tracked.read_text(encoding="utf-8") == "line1\nline2\n"


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


def test_manifest_section_lists_in_scope_changes_with_counts_and_totals(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "## Files changed" in notes
    assert CAND_FILE in notes
    assert "modified" in notes
    assert "+3/-1" in notes
    assert "in scope" in notes
    assert "**Totals:** 1 file changed, +3/-1." in notes
    # No false alarm when everything is in scope.
    assert "outside the declared scope" not in notes


def test_manifest_section_flags_out_of_scope_files_prominently(spec_and_repo) -> None:
    manifest = [
        FileChange(
            path=CAND_FILE, change_kind="modified", insertions=3, deletions=1, binary=False
        ),
        FileChange(
            path="pkg/attn/extra.py",
            change_kind="added",
            insertions=12,
            deletions=0,
            binary=False,
        ),
    ]
    notes = _notes(spec_and_repo, manifest=manifest)
    assert "pkg/attn/extra.py" in notes
    assert "**OUT OF SCOPE**" in notes
    assert "**Totals:** 2 files changed, +15/-1." in notes
    # The callout must be prominent prose, not just a table cell.
    assert "**This patch touches files outside the declared scope.**" in notes
    assert "pkg/attn/extra.py" in notes.split("outside the declared scope", 1)[1]


def test_out_of_scope_files_helper_matches_the_notes(spec_and_repo) -> None:
    spec, _repo = spec_and_repo
    manifest = [
        FileChange(
            path=CAND_FILE, change_kind="modified", insertions=1, deletions=0, binary=False
        ),
        FileChange(
            path="pkg/attn/extra.py",
            change_kind="added",
            insertions=1,
            deletions=0,
            binary=False,
        ),
    ]
    assert out_of_scope_files(spec, manifest) == ["pkg/attn/extra.py"]


def test_manifest_section_renders_a_binary_file_without_line_counts(spec_and_repo) -> None:
    manifest = [
        FileChange(
            path="pkg/attn/blob.bin",
            change_kind="added",
            insertions=None,
            deletions=None,
            binary=True,
        )
    ]
    notes = _notes(spec_and_repo, manifest=manifest)
    assert "pkg/attn/blob.bin" in notes
    assert "binary" in notes


def test_manifest_section_is_coherent_when_no_patch_was_produced(spec_and_repo) -> None:
    notes = _notes(
        spec_and_repo,
        patch_produced=False,
        change_summary="The proposal needs a change in a file outside scope.",
        manifest=[],
    )
    assert "## Files changed" in notes
    # It must not claim a change happened when none did.
    assert "**Totals:**" not in notes
    assert "OUT OF SCOPE" not in notes


def test_manifest_section_never_claims_binary_or_a_false_total_for_unknown_counts(
    spec_and_repo,
) -> None:
    """Important 3: the length-mismatch fallback used to set

    `insertions=deletions=None`, and `binary` was *derived* from
    `insertions is None and deletions is None` — so "counts unknown" was
    indistinguishable from "is a binary file", and the totals line summed
    `insertions or 0` for the unknown file too, silently reporting a total
    that understates the real change. Both are false statements about the
    one artifact a reviewer is told to trust.

    `counts_known=False` (a real file, not binary, whose line counts could
    not be determined) must render as neither "binary" nor a clean numeric
    total.
    """
    manifest = [
        FileChange(
            path=CAND_FILE,
            change_kind="modified",
            insertions=None,
            deletions=None,
            binary=False,
            counts_known=False,
        ),
        FileChange(
            path="pkg/attn/extra.py",
            change_kind="added",
            insertions=5,
            deletions=0,
            binary=False,
            counts_known=True,
        ),
    ]
    notes = _notes(spec_and_repo, manifest=manifest)

    # The unknown-counts file must not be mislabeled "binary" — it is a real
    # file whose counts just could not be determined.
    row_a = next(ln for ln in notes.splitlines() if f"`{CAND_FILE}`" in ln)
    assert "binary" not in row_a
    assert "?" in row_a

    # The totals line must not silently sum in a 0 for the unknown file —
    # that understates the real (unknown) change. It must say plainly that
    # some counts are unavailable, and must not print a bare "+0/-0"-style
    # total that omits the unknown file's contribution.
    totals_line = next(ln for ln in notes.splitlines() if ln.startswith("**Totals:**"))
    assert "unavailable" in totals_line
    assert "1" in totals_line  # one file with unknown counts
    assert "+0/-0" not in totals_line


def test_declared_scope_normalizes_dot_and_dotdot_segments(spec_and_repo) -> None:
    """Important 4: `_declared_scope` compared `t.file` verbatim against the

    diff's canonical POSIX paths. `validate_target.py` only checks
    containment and existence, so a declared target spelled
    `./pkg/attn/tile.py` (or with a `..` segment) validates and is stored
    as-is — then the diff's canonical `pkg/attn/tile.py` fails to match it,
    producing a false "OUT OF SCOPE" callout on the candidate's own file.
    """
    spec, _repo = spec_and_repo
    spec.targets[0].file = f"./{CAND_FILE}"
    manifest = [
        FileChange(path=CAND_FILE, change_kind="modified", insertions=1, deletions=0, binary=False)
    ]
    notes = _notes(spec_and_repo, manifest=manifest)
    assert "**OUT OF SCOPE**" not in notes
    assert out_of_scope_files(spec, manifest) == []


def test_render_fix_notes_requires_manifest(spec_and_repo) -> None:
    """Minor 7: `manifest` must be a required keyword, not `()`-defaulted.

    A caller that passes `patch_produced=True` but forgets `manifest` used to
    get a notes file claiming "no patch was produced" — actively lying about
    a patch that *was* produced. Making the parameter required turns that
    mistake into a `TypeError` at the call site instead of a wrong artifact.
    """
    import inspect

    sig = inspect.signature(render_fix_notes)
    assert sig.parameters["manifest"].default is inspect.Parameter.empty


def test_change_row_escapes_a_pipe_in_the_path(spec_and_repo) -> None:
    """Minor 8: GFM tables don't treat backticks as pipe-escapes — a path

    containing `|` breaks the row into extra columns.
    """
    manifest = [
        FileChange(
            path="pkg/attn/weird|name.py",
            change_kind="added",
            insertions=1,
            deletions=0,
            binary=False,
        )
    ]
    notes = _notes(spec_and_repo, manifest=manifest)
    assert "weird\\|name.py" in notes


def test_agent_error_is_recorded(spec_and_repo) -> None:
    notes = _notes(
        spec_and_repo, patch_produced=False, change_summary=None, agent_error="claude exit=3"
    )
    assert "claude exit=3" in notes
