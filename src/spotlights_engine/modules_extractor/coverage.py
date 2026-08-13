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

from spotlights_engine.modules_extractor.skeleton import is_source_file
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    Skeleton,
    SourceRootDecision,
)
from spotlights_engine.schemas.project import (
    ProjectTree,
    Repository,
    _qualified_name,
)


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
    skip_dependency_resolution: bool = False,
    allowed_internal_qns: set[str] | None = None,
    rule4_exempt_paths: set[str] | None = None,
) -> None:
    """Validate the enriched tree against the filesystem, repo metadata, and
    skeleton inventory. Raises `CrossArtifactError` on any structural failure.

    Coverage (missing required paths) is computed separately by
    `compute_coverage`; this function establishes the invariants that make the
    coverage equation meaningful.

    The three keyword parameters exist only so a *sharded* Stage-3 enrichment
    (see `sharding.py`) can run this same validator against one shard's subtree
    without forking it. Each defaults to today's exact behavior, so the merged
    Stage-5 call is byte-for-byte the unsharded validation:

    - `skip_dependency_resolution` — when True, drop **only** the "internal dep
      resolves to an emitted top-level module in *this* tree" clause. A shard
      legitimately references sibling top-level qualified names it neither sees
      nor emits; the merged tree proves resolvability.
    - `allowed_internal_qns` — when set, every internal (non-external)
      dependency must be a member of this vocabulary. This is the shard-time
      guard that catches an invented qualified name before the merge does.
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
        for f in main_files:
            fpath = f.path
            if not fpath.startswith(module_path + "/") and not (
                PurePosixPath(fpath).parent.as_posix() == module_path
            ):
                # main file must live under the module directory
                if not (fpath == module_path or fpath.startswith(module_path + "/")):
                    raise CrossArtifactError(
                        f"main_file {fpath!r} is not under module {module_path!r}"
                    )
            if allow_any_file:
                if not _is_real_file(repo_path, fpath):
                    raise CrossArtifactError(
                        f"main_file {fpath!r} is not a real non-symlink file"
                    )
            elif not _is_real_source_file(repo_path, fpath):
                raise CrossArtifactError(
                    f"main_file {fpath!r} is not a real non-symlink source file"
                )
            # Not owned by a separately emitted descendant module.
            owner = _nearest_emitted_owner(fpath, emitted_set)
            if owner != module_path:
                raise CrossArtifactError(
                    f"main_file {fpath!r} belongs to emitted descendant "
                    f"{owner!r}, not {module_path!r}"
                )
            if fpath in owned_by:
                raise CrossArtifactError(
                    f"main_file {fpath!r} claimed by both {owned_by[fpath]!r} "
                    f"and {module_path!r}"
                )
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
        raise CrossArtifactError(
            f"module(s) with a single child; collapse each "
            f"(a parent needs zero or ≥2 children): {listed}"
        )

    # Rule 5: dependencies.
    _validate_dependencies(
        enriched,
        provisional,
        repository,
        skip_resolution=skip_dependency_resolution,
        allowed_internal_qns=allowed_internal_qns,
    )

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


def _validate_dependencies(
    enriched: EnrichedTree,
    tree: ProjectTree,
    repository: Repository,
    *,
    skip_resolution: bool = False,
    allowed_internal_qns: set[str] | None = None,
) -> None:
    source_root = repository.source_root
    top_qns = {
        _qualified_name(m.path, source_root): m.path for m in tree.modules
    }
    externals = set(repository.external_dependencies)

    # External names must not collide with an emitted top-level qn.
    for ext in externals:
        if ext in top_qns:
            raise CrossArtifactError(
                f"external dependency {ext!r} collides with emitted top-level "
                "qualified name"
            )

    for m in enriched.modules:
        this_qn = _qualified_name(m.path, source_root)
        seen: set[str] = set()
        for dep in m.depends_on:
            if dep in seen:
                raise CrossArtifactError(
                    f"duplicate dependency {dep!r} on module {this_qn!r}"
                )
            seen.add(dep)
            if dep == this_qn:
                raise CrossArtifactError(
                    f"module {this_qn!r} depends on itself"
                )
            if dep in externals:
                continue
            if allowed_internal_qns is not None and dep not in allowed_internal_qns:
                raise CrossArtifactError(
                    f"internal dependency {dep!r} on {this_qn!r} is not in the "
                    "provided top-level module vocabulary"
                )
            if not skip_resolution and dep not in top_qns:
                raise CrossArtifactError(
                    f"dependency {dep!r} on {this_qn!r} resolves to neither an "
                    "emitted top-level module nor an external dependency"
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

    # main_files per emitted module (normalized).
    main_files_of: dict[str, set[str]] = {}
    for path, files in _iter_module_mainfiles(enriched):
        main_files_of[path] = {f.path for f in files}

    # Organizational-only passthroughs (skeleton-classified: only direct source
    # is `__init__.py` and fewer than two source-bearing children). Their real
    # content lives in an emitted descendant, so no substantive source file of
    # their own can appear in the target's main_files — the Rule 7 in-main_files
    # requirement is unsatisfiable for them and is waived below.
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
        # of its own to cite, so the in-main_files requirement below cannot be
        # met. Accept the fold on the strength of the structural checks already
        # passed; the descendant emission is what actually covers the content.
        if fpath in organizational:
            continue
        # Rule 7: ≥1 unique non-symlink inventoried source file under the folded
        # path, outside any separately emitted descendant, present in the
        # target's main_files.
        target_mains = main_files_of.get(into, set())
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
            if ev in target_mains:
                valid_evidence = True
                break
        if not valid_evidence:
            raise CrossArtifactError(
                f"fold {fpath!r} into {into!r} has no valid evidence file "
                "present in the target's main_files"
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
