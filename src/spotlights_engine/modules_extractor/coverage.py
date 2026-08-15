"""Cross-artifact validation and coverage computation for the two-phase extractor.

Everything here checks paths against the real filesystem — the existing
`ProjectTree` validators only check lexical nesting, normalized names, and
qualified-name collisions. This module adds:

- `validate_source_root_decision` — Stage-1 filesystem/exclusion validation.
- `validate_enriched_tree` — Stage-3 cross-artifact validation (paths exist and
  resolve inside the repo without symlinks, shape rules, fold rules, and a
  provisional `ProjectTree` pass).
- `compute_coverage` — the small, deliberate coverage equation
  (`missing = required - emitted - folded`) plus the optional-path ledger.

See `design/module_extraction_fix_impl_plan.md` (Stages 1, 3, 5).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.constants import MAX_MAIN_FILES
from spotlights_engine.modules_extractor.skeleton import is_source_file
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    Skeleton,
    SourceRootDecision,
)
from spotlights_engine.schemas.project import (
    ProjectTree,
    Repository,
)

# How many Rule 3/4 violations one error message may list before it summarizes
# the rest. See the end of `validate_enriched_tree`.
_MAX_REPORTED_PROBLEMS = 20


class CrossArtifactError(ValueError):
    """A cross-artifact (filesystem/shape/fold) validation failure.

    Distinct from a missing-coverage failure: this means the tree is malformed,
    not merely incomplete.
    """


class CoverageReport(BaseModel):
    """The coverage ledger written to `coverage.json`."""

    model_config = ConfigDict(extra="forbid")

    required: list[str] = Field(default_factory=list)
    emitted: list[str] = Field(default_factory=list)
    folded: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    invalid_folds: list[str] = Field(default_factory=list)
    optional_inventory: list[str] = Field(default_factory=list)
    optional_emitted: list[str] = Field(default_factory=list)
    optional_folded: list[str] = Field(default_factory=list)
    optional_unaccounted: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.invalid_folds


# ── Filesystem helpers ────────────────────────────────────────────────────


def _resolves_inside(repo_path: Path, rel: str) -> bool:
    """True when `repo_path/rel` resolves to a path inside `repo_path`.

    Guards against a component symlink escaping the repository.
    """
    repo_resolved = repo_path.resolve(strict=False)
    target = (repo_path / rel).resolve(strict=False)
    try:
        target.relative_to(repo_resolved)
    except ValueError:
        return False
    return True


def _has_symlink_component(repo_path: Path, rel: str) -> bool:
    """True when any component of `rel` (below the repo) is a symlink."""
    current = repo_path
    for seg in PurePosixPath(rel).parts:
        current = current / seg
        if current.is_symlink():
            return True
    return False


def _is_real_dir(repo_path: Path, rel: str) -> bool:
    p = repo_path / rel
    return p.is_dir() and not _has_symlink_component(repo_path, rel)


def _is_real_source_file(repo_path: Path, rel: str) -> bool:
    p = repo_path / rel
    return (
        p.is_file()
        and not _has_symlink_component(repo_path, rel)
        and is_source_file(p.name)
    )


def _is_real_file(repo_path: Path, rel: str) -> bool:
    """A real non-symlink file, of any extension (source or not)."""
    p = repo_path / rel
    return p.is_file() and not _has_symlink_component(repo_path, rel)


def _dir_has_direct_source_file(repo_path: Path, module_path: str) -> bool:
    """True when the module's own directory holds ≥1 direct source-extension file.

    Distinguishes a module that simply hasn't cited its source (strict gate
    applies) from one that *has no* source-extension file to cite — e.g. a
    `docker/` directory of Dockerfiles + `.hcl` + `.json`, whose only real
    source lives in a child directory. The latter is allowed to cite its own
    non-source files (Dockerfile, bake config) as `main_files`.

    Symlinks are ignored to match the non-symlink invariant enforced elsewhere.
    """
    d = repo_path / module_path
    try:
        entries = list(d.iterdir())
    except (OSError, NotADirectoryError):
        return False
    return any(
        entry.is_file()
        and not entry.is_symlink()
        and is_source_file(entry.name)
        for entry in entries
    )


def _dir_has_direct_file(repo_path: Path, module_path: str) -> bool:
    """True when the module's own directory holds ≥1 direct non-symlink file of
    any extension.

    False means a *pure container* directory — only sub-directories, like a Go
    `cmd/` or a namespace package — which is the one shape allowed to cite no
    `main_files` at all: everything under it is owned by a child.
    """
    d = repo_path / module_path
    try:
        entries = list(d.iterdir())
    except (OSError, NotADirectoryError):
        return False
    return any(entry.is_file() and not entry.is_symlink() for entry in entries)


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    """Physical-path ancestry (`ancestor` strictly contains `descendant`)."""
    return descendant == ancestor or descendant.startswith(ancestor + "/")


# ── Stage 1 — source-root decision validation ─────────────────────────────


def forced_repository_level_files(
    source_root: str, detected_source_files: list[str]
) -> list[str]:
    """Source files sitting *directly* at the selected root.

    A file at the root of the modeled tree cannot itself be a directory module
    (the schema forbids a module path equal to `source_root`, and a bare file
    has no directory to represent it), so it can only ever be excluded as a
    `repository_level_file`. That makes the set mechanically determined rather
    than a judgment call: `validate_source_root_decision` *requires* each such
    file to be excluded, so the extractor injects them deterministically instead
    of trusting the Stage-1 model to enumerate every one.

    Returned paths are sorted and repo-relative. Semantic exclusions (a tests,
    docs, or benchmarks *directory*, or source outside a non-empty root) are not
    included here — those remain the model's decision.
    """
    forced: list[str] = []
    if source_root:
        prefix = source_root + "/"
        for f in detected_source_files:
            if f.startswith(prefix) and "/" not in f[len(prefix):]:
                forced.append(f)
    else:
        for f in detected_source_files:
            if "/" not in f:
                forced.append(f)
    return sorted(forced)


def validate_source_root_decision(
    decision: SourceRootDecision,
    repo_path: Path,
    detected_source_files: list[str],
) -> None:
    """Filesystem validation of a Stage-1 decision. Raises `CrossArtifactError`.

    `detected_source_files` is the repo-wide safe source scan (sorted,
    repo-relative). The nested `Repository` has already normalized `source_root`.
    """
    source_root = decision.repository.source_root

    # Root resolves to an existing, non-symlink directory inside the repo.
    if source_root:
        if not _is_real_dir(repo_path, source_root):
            raise CrossArtifactError(
                f"source_root {source_root!r} is not an existing non-symlink "
                "directory inside the repository"
            )

    # Exclusions: exist inside the repo, form an antichain, don't equal/contain
    # the repo or source root, and each covers ≥1 detected source file.
    exclusions = [e.path for e in decision.excluded_source_paths]
    if len(set(exclusions)) != len(exclusions):
        raise CrossArtifactError(f"duplicate excluded_source_paths: {exclusions}")

    for ex in exclusions:
        if not _resolves_inside(repo_path, ex):
            raise CrossArtifactError(f"exclusion escapes repository: {ex!r}")
        if not (repo_path / ex).exists():
            raise CrossArtifactError(f"exclusion path does not exist: {ex!r}")
        if source_root and (ex == source_root):
            raise CrossArtifactError(
                f"exclusion cannot equal source_root: {ex!r}"
            )
        if source_root and _is_ancestor(ex, source_root):
            raise CrossArtifactError(
                f"exclusion {ex!r} contains the source_root {source_root!r}"
            )

    # Antichain: no exclusion nested under another.
    for a in exclusions:
        for b in exclusions:
            if a is not b and a != b and _is_ancestor(b, a):
                raise CrossArtifactError(
                    f"exclusions must form an antichain; {a!r} is nested under {b!r}"
                )

    # Each exclusion covers ≥1 detected source file.
    for ex in exclusions:
        covered = any(
            f == ex or f.startswith(ex + "/") for f in detected_source_files
        )
        if not covered:
            raise CrossArtifactError(
                f"exclusion {ex!r} covers no detected source file"
            )


# ── Stage 3 — cross-artifact validation ────────────────────────────────────


def validate_enriched_tree(
    enriched: EnrichedTree,
    repo_path: Path,
    repository: Repository,
    skeleton: Skeleton,
    *,
    rule4_exempt_paths: set[str] | None = None,
) -> None:
    """Validate the enriched tree against the filesystem, repo metadata, and
    skeleton inventory. Raises `CrossArtifactError` on any structural failure.

    Coverage (missing required paths) is computed separately by
    `compute_coverage`; this function establishes the invariants that make the
    coverage equation meaningful.

    The keyword parameter exists only so a *sharded* Stage-3 enrichment
    (see `sharding.py`) can run this same validator against one shard's subtree
    without forking it. It defaults to today's exact behavior, so the merged
    Stage-5 call is byte-for-byte the unsharded validation:

    - `rule4_exempt_paths` — when set, the zero-or-≥2-children rule is
      *deferred* (not weakened) for exactly those emitted paths. Used only by a
      spine shard for its own root, whose ≥2 guaranteed children are pruned
      from its subtree; the merged tree enforces Rule 4 in full.
    """
    inventory = skeleton.all_paths()

    all_modules = list(enriched.iter_all_modules())
    emitted_paths = [_norm(m.path) for m in all_modules]

    # Rule 1: every emitted path is an inventoried directory that still resolves
    # inside the repo without a symlink component.
    for path in emitted_paths:
        if path not in inventory:
            raise CrossArtifactError(
                f"emitted module path {path!r} is not in the skeleton inventory"
            )
        if not _is_real_dir(repo_path, path):
            raise CrossArtifactError(
                f"emitted module path {path!r} is not a real non-symlink directory"
            )

    if len(set(emitted_paths)) != len(emitted_paths):
        raise CrossArtifactError(f"duplicate emitted module paths: {emitted_paths}")

    # Rule 2: provisional ProjectTree runs the lexical/name/qn validators, and
    # object-tree ancestry must agree with physical-path ancestry.
    provisional_dict = {
        "repository": repository.model_dump(),
        "modules": enriched.modules_as_project_tree_dicts(),
    }
    try:
        provisional = ProjectTree.model_validate(provisional_dict)
    except Exception as exc:  # noqa: BLE001 — surfaced as a cross-artifact error
        raise CrossArtifactError(
            f"provisional ProjectTree validation failed: {exc}"
        ) from exc

    _check_object_tree_matches_physical(provisional)

    # Rules 3 and 4 are collected, not raised one at a time: they constrain the
    # same choice (which directories to emit and what each one cites), so a tree
    # that violates both must be shown both. A repair told only about Rule 3
    # rearranges the emitted set, trips Rule 4, and burns the single bounded
    # repair pass on half a fix.
    problems: list[str] = []

    # Rule 3: main_files.
    owned_by: dict[str, str] = {}
    emitted_set = set(emitted_paths)
    for module_path, main_files in _iter_module_mainfiles(enriched):
        # A module directory holding no direct source-extension file (e.g. a
        # `docker/` of Dockerfiles + `.hcl` + `.json`, whose only real source
        # lives in a child) has nothing source-typed to cite, so it may cite its
        # own real non-symlink files of any extension. A directory that *does*
        # hold source keeps the strict source-extension gate.
        allow_any_file = not _dir_has_direct_source_file(repo_path, module_path)
        if not main_files:
            # A pure container directory — one holding no direct file at all,
            # only sub-directories (a Go `cmd/`, a namespace package) — has
            # nothing of its own to cite: every file beneath it belongs to a
            # child. Requiring ≥1 main_file there is unsatisfiable as soon as it
            # emits its children, and the only escape would be to fold real
            # structure away. So it, and only it, may cite nothing.
            if _dir_has_direct_file(repo_path, module_path):
                problems.append(
                    f"module {module_path!r} cites no main_files, but its "
                    f"directory holds direct files; cite 1–{MAX_MAIN_FILES} "
                    "of them"
                )
            continue
        for f in main_files:
            fpath = f.path
            # A main file must live under the module directory. (`fpath ==
            # module_path` — a directory cited as a file — falls through to the
            # real-file checks below, which reject it precisely.)
            if fpath != module_path and not fpath.startswith(module_path + "/"):
                problems.append(
                    f"main_file {fpath!r} is not under module {module_path!r}"
                )
                continue
            if allow_any_file:
                if not _is_real_file(repo_path, fpath):
                    problems.append(
                        f"main_file {fpath!r} is not a real non-symlink file"
                    )
                    continue
            elif not _is_real_source_file(repo_path, fpath):
                problems.append(
                    f"main_file {fpath!r} is not a real non-symlink source file"
                )
                continue
            # Not owned by a separately emitted descendant module.
            owner = _nearest_emitted_owner(fpath, emitted_set)
            if owner != module_path:
                problems.append(
                    f"main_file {fpath!r} belongs to emitted descendant "
                    f"{owner!r}, not {module_path!r}"
                )
                continue
            if fpath in owned_by:
                problems.append(
                    f"main_file {fpath!r} claimed by both {owned_by[fpath]!r} "
                    f"and {module_path!r}"
                )
                continue
            owned_by[fpath] = module_path

    # Rule 4: every parent has zero or ≥2 emitted children. Report *all*
    # offenders in one error — a subtree can carry several clustered
    # single-child parents (e.g. organizational dirs each wrapping one child),
    # and the single bounded repair pass can only fix what it is shown. Raising
    # on the first offender would surface them one-per-attempt and exhaust the
    # repair budget before the last is collapsed.
    exempt = rule4_exempt_paths or set()
    single_child = [
        _norm(m.path)
        for m in all_modules
        if len(m.submodules) == 1 and _norm(m.path) not in exempt
    ]
    if single_child:
        listed = ", ".join(repr(p) for p in single_child)
        problems.append(
            f"module(s) with a single child; collapse each "
            f"(a parent needs zero or ≥2 children): {listed}"
        )

    if problems:
        # Capped: the message is fed back verbatim as the repair prompt, and a
        # wholesale-broken tree would otherwise blow that budget listing every
        # offending file. The first _MAX_REPORTED are enough to characterize it.
        shown = problems[:_MAX_REPORTED_PROBLEMS]
        extra = len(problems) - len(shown)
        message = "; ".join(shown)
        if extra:
            message += f"; (and {extra} more violation(s))"
        raise CrossArtifactError(message)

    # Rules 6–8: folds.
    _validate_folds(enriched, repo_path, skeleton, provisional, emitted_set)


def _norm(path: str) -> str:
    return path.strip().strip("/")


def _iter_module_mainfiles(enriched: EnrichedTree):
    for m in enriched.iter_all_modules():
        yield _norm(m.path), m.main_files


def _nearest_emitted_owner(file_path: str, emitted: set[str]) -> str | None:
    """The deepest emitted module directory that contains `file_path`."""
    best: str | None = None
    for path in emitted:
        if file_path == path or file_path.startswith(path + "/"):
            if best is None or len(path) > len(best):
                best = path
    return best


def _check_object_tree_matches_physical(tree: ProjectTree) -> None:
    """Every pair of emitted paths must agree between filesystem ancestry and
    object-tree ancestry — no overlapping top-level siblings or cousins."""
    # Map each path to its object-tree ancestor chain.
    obj_parent: dict[str, str | None] = {}

    def _walk(modules, parent: str | None) -> None:
        for m in modules:
            p = _norm(m.path)
            obj_parent[p] = parent
            _walk(m.submodules, p)

    _walk(tree.modules, None)
    paths = list(obj_parent)
    for a in paths:
        for b in paths:
            if a == b:
                continue
            phys_anc = _is_ancestor(a, b)  # a contains b physically
            # object-tree ancestry: is a on b's parent-chain?
            obj_anc = False
            cur = obj_parent[b]
            while cur is not None:
                if cur == a:
                    obj_anc = True
                    break
                cur = obj_parent[cur]
            if phys_anc and not obj_anc:
                raise CrossArtifactError(
                    f"paths {a!r} and {b!r} are physically nested but not "
                    "object-tree ancestors (overlapping siblings/cousins)"
                )


def _validate_folds(
    enriched: EnrichedTree,
    repo_path: Path,
    skeleton: Skeleton,
    tree: ProjectTree,
    emitted: set[str],
) -> None:
    inventory = skeleton.all_paths()

    # Object-tree ancestor chain for emitted paths.
    obj_parent: dict[str, str | None] = {}

    def _walk(modules, parent: str | None) -> None:
        for m in modules:
            p = _norm(m.path)
            obj_parent[p] = parent
            _walk(m.submodules, p)

    _walk(tree.modules, None)

    def _is_object_ancestor(anc: str, desc: str) -> bool:
        cur = obj_parent.get(desc)
        while cur is not None:
            if cur == anc:
                return True
            cur = obj_parent.get(cur)
        return False

    # Organizational-only passthroughs (skeleton-classified: only direct source
    # is `__init__.py` and fewer than two source-bearing children). Their real
    # content lives in an emitted descendant, so they have no substantive
    # source of their own to prove inspection with — the Rule-8 evidence gate
    # is unsatisfiable for them and is waived below.
    organizational = set(skeleton.organizational_only)

    seen_paths: set[str] = set()
    for fold in enriched.folds:
        fpath = _norm(fold.path)
        into = _norm(fold.into)
        if fpath in seen_paths:
            raise CrossArtifactError(f"duplicate fold path: {fpath!r}")
        seen_paths.add(fpath)
        # Fold path is an inventory path that was not emitted.
        if fpath not in inventory:
            raise CrossArtifactError(
                f"fold path {fpath!r} is not in the skeleton inventory"
            )
        if fpath in emitted:
            raise CrossArtifactError(
                f"fold path {fpath!r} is also emitted as a module"
            )
        # Target must be emitted, physically an ancestor, and object-tree
        # ancestor.
        if into not in emitted:
            raise CrossArtifactError(
                f"fold target {into!r} is not an emitted module"
            )
        if not _is_ancestor(into, fpath) or into == fpath:
            raise CrossArtifactError(
                f"fold path {fpath!r} is not physically nested under target "
                f"{into!r}"
            )
        if not _is_object_ancestor(into, fpath):
            # fpath isn't emitted, so its object-tree ancestor is its nearest
            # emitted ancestor; require that to be `into`.
            nearest = _nearest_emitted_owner(fpath, emitted)
            if nearest != into:
                raise CrossArtifactError(
                    f"fold target {into!r} is not {fpath!r}'s nearest emitted "
                    f"ancestor ({nearest!r})"
                )
        # An organizational-only passthrough (its real source emitted as a child
        # module, itself holding only an `__init__.py`) has no substantive source
        # of its own, so the evidence gate below cannot be met substantively.
        # Accept the fold on the strength of the structural checks already
        # passed; the descendant emission is what actually covers the content.
        if fpath in organizational:
            continue
        # Same reasoning, filesystem-determined: a folded directory holding no
        # direct source-extension file at all (a Java package chain like
        # `src/.../org/apache/kafka`, a pure container of sub-directories) has
        # nothing substantive of its own to cite as evidence. Coverage still
        # holds its required descendants to account individually, so the waiver
        # folds away only the passthrough node itself, never its content.
        if not _dir_has_direct_source_file(repo_path, fpath):
            continue
        # Rule 8: ≥1 real non-symlink source file under the folded path, outside
        # any separately emitted descendant — proof the folded content was
        # inspected. (The old receipt rule additionally required the file in the
        # target's `main_files`; Level 1 decoupled evidence from documentation,
        # so several sibling folds no longer compete for the five slots.)
        valid_evidence = False
        for ev in fold.evidence_files:
            ev = _norm(ev)
            if not (ev == fpath or ev.startswith(fpath + "/")):
                continue
            if not _is_real_source_file(repo_path, ev):
                continue
            owner = _nearest_emitted_owner(ev, emitted)
            if owner is not None and owner != into and _is_ancestor(owner, ev):
                # owned by a separately emitted descendant
                continue
            valid_evidence = True
            break
        if not valid_evidence:
            raise CrossArtifactError(
                f"fold {fpath!r} into {into!r} has no valid evidence file "
                "(a real non-symlink source file under the folded path, "
                "outside separately emitted modules)"
            )


# ── Coverage computation ───────────────────────────────────────────────────


def compute_coverage(enriched: EnrichedTree, skeleton: Skeleton) -> CoverageReport:
    """Compute the coverage ledger. Deliberately small:

        required = {node.path for required skeleton nodes}
        emitted  = {emitted module/submodule paths}
        folded   = {validated FoldRecord paths}
        missing  = required - emitted - folded

    Assumes `validate_enriched_tree` already ran (so folds are valid); this only
    computes set membership. Never adds ancestors of emitted paths to `emitted`.
    """
    required = skeleton.required_paths()
    all_inventory = skeleton.all_paths()
    optional_inventory = all_inventory - required

    emitted = {_norm(m.path) for m in enriched.iter_all_modules()}
    folded = {_norm(f.path) for f in enriched.folds}

    missing = required - emitted - folded

    optional_emitted = optional_inventory & emitted
    optional_folded = optional_inventory & folded
    optional_unaccounted = optional_inventory - emitted - folded

    return CoverageReport(
        required=sorted(required),
        emitted=sorted(emitted),
        folded=sorted(folded),
        missing=sorted(missing),
        invalid_folds=[],
        optional_inventory=sorted(optional_inventory),
        optional_emitted=sorted(optional_emitted),
        optional_folded=sorted(optional_folded),
        optional_unaccounted=sorted(optional_unaccounted),
    )


__all__ = [
    "CoverageReport",
    "CrossArtifactError",
    "compute_coverage",
    "forced_repository_level_files",
    "validate_enriched_tree",
    "validate_source_root_decision",
]
