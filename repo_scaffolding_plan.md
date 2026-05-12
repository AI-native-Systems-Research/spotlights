# Discovery Engine — Repo Scaffolding Plan

*Handoff document for Claude Code. Creates three repositories with consistent structure, naming, and conventions.*

---

## What this document is

This is the plan CC should follow to create the initial repositories for the Discovery Engine project. The goal is **three repos with parallel structure, shared conventions, and enough scaffolding that bundle owners can clone and start work** without making structural decisions individually. The repos should look like siblings, not cousins.

Reference documents CC should read first:
- `discovery_engine_proposal.md` — the project proposal (vocabulary and architecture)
- `signal_interface_spec.md` — the Signal interface contract

CC should not invent structure or conventions not specified here. When in doubt, ask before deviating.

---

## The three repos

| Repo | Package name | Purpose |
|---|---|---|
| `discovery-engine` | `discovery_engine` | The spine. Schemas, orchestration, integration tests, the demo path. Contains bundles C, D, E, F as subpackages until they earn their own repos. |
| `discovery-observability` | `discovery_observability` | Bundle A. Telemetry collection, signal extraction, trace summarization. |
| `discovery-knowledge` | `discovery_knowledge` | Bundle B. Literature index, archive of past candidates and outcomes, retrieval interface. |

**Naming rule:** repo name uses hyphens (`discovery-engine`), Python package uses underscores (`discovery_engine`). All three follow this pattern.

**Dependency direction:** `discovery-engine` depends on `discovery-observability` and `discovery-knowledge`. The two leaf repos do **not** depend on each other and do **not** depend on the engine. This keeps the dependency graph a tree, not a cycle.

---

## Shared conventions

These apply identically across all three repos.

**Python.** Version 3.11+. Package layout uses `src/` (i.e., `src/discovery_engine/`, not `discovery_engine/` at root). This prevents accidental imports from the working directory during development.

**Tooling.**
- Packaging: `uv` with `pyproject.toml`. No `setup.py`, no `requirements.txt`.
- Linting: `ruff` (replaces black, isort, flake8 in one tool). Default ruleset is fine for now.
- Type checking: `mypy` configured but lenient at start (`strict = false`); ratchet up over time.
- Tests: `pytest`. Test layout mirrors source (`tests/unit/`, `tests/integration/`).
- CI: GitHub Actions. One workflow file per repo, identical structure.

**License.** Apache 2.0. `LICENSE` file at repo root.

**Pre-commit.** A `.pre-commit-config.yaml` running ruff and basic file hygiene (trailing whitespace, end-of-file fixer, yaml/json validation).

---

## Repository structure (shared template)

Every repo has this top-level structure. Bundle-specific contents go inside; the shell is identical.

```
<repo-name>/
├── README.md                  # what this repo is, how it relates to others
├── CHARTER.md                 # the bundle charter for this repo (or "see discovery-engine/charters/" for the spine)
├── LICENSE                    # Apache 2.0
├── pyproject.toml             # uv-managed
├── .pre-commit-config.yaml
├── .gitignore                 # standard Python ignore plus .venv, .ruff_cache, .mypy_cache
├── .github/
│   └── workflows/
│       └── ci.yml             # lint, type-check, test on push and PR
├── src/
│   └── <package_name>/
│       ├── __init__.py        # exposes the public API
│       └── ...                # bundle-specific modules
├── tests/
│   ├── unit/
│   └── integration/
└── docs/
    └── ...                    # bundle-specific design notes, ADRs, etc.
```

`docs/` is intentionally light at start. Bundle owners populate it as the work matures. Do not pre-create empty subdirectories.

---

## Per-repo specifics

### `discovery-engine`

The spine. Owns schemas, orchestration, and the subpackages for bundles that haven't graduated to their own repos yet.

Structure inside `src/discovery_engine/`:

```
src/discovery_engine/
├── __init__.py
├── schemas/                   # the canonical Candidate, Change, and other shared types
│   ├── __init__.py
│   ├── candidate.py
│   ├── change.py
│   └── signal.py              # re-exports types from discovery_observability
├── candidates/                # Bundle C lives here for now
│   └── __init__.py
├── changes/                   # Bundle D lives here for now
│   └── __init__.py
├── validation/                # Bundle E lives here for now
│   └── __init__.py
├── orchestration/             # Bundle F lives here
│   └── __init__.py
└── charters/                  # bundle charters for in-repo bundles
    ├── candidates.md
    ├── changes.md
    ├── validation.md
    └── orchestration.md
```

Schemas are the **only** thing in `discovery_engine` that other repos may import. Everything else is internal. The `schemas` module is the contract surface.

The `signal.py` schema re-exports types from `discovery_observability.signals` so consumers can write `from discovery_engine.schemas import TraceSummary` without depending on observability's internal layout. Re-export, do not re-define.

`pyproject.toml` dependencies: `discovery-observability`, `discovery-knowledge`. Use editable installs locally; pin to specific commits or tags once stable.

### `discovery-observability`

Bundle A's repo. Owns telemetry collection and signal extraction.

Structure inside `src/discovery_observability/`:

```
src/discovery_observability/
├── __init__.py
├── signals/                   # the public API: WorkloadProfile, TraceSummary, Anomaly
│   ├── __init__.py
│   └── types.py
├── collection/                # raw telemetry capture
│   └── __init__.py
├── extraction/                # turning raw traces into TraceSummary objects
│   └── __init__.py
└── detection/                 # anomaly detectors
    └── __init__.py
```

The `signals` submodule is the public API and must match `signal_interface_spec.md` exactly. CC should generate stub dataclasses for `WorkloadProfile`, `TraceSummary`, and `Anomaly` directly from that spec — field names, types, and docstrings drawn from the spec verbatim.

No external dependencies on other Discovery Engine repos. Standalone.

### `discovery-knowledge`

Bundle B's repo. Owns the literature index and the experimental archive.

Structure inside `src/discovery_knowledge/`:

```
src/discovery_knowledge/
├── __init__.py
├── retrieval/                 # the public API: query interface
│   └── __init__.py
├── literature/                # paper and informal-source ingestion and indexing
│   └── __init__.py
├── archive/                   # past candidates, changes, outcomes
│   └── __init__.py
└── corpus/                    # the storage substrate (LLM Wiki experiment)
    └── __init__.py
```

The `retrieval` submodule is the public API: a single function `query(query: str, mode: Literal["signal_driven", "technique_driven"]) -> list[Result]` per Appendix A of the proposal. Other modules are internal.

No external dependencies on other Discovery Engine repos. Standalone.

---

## README structure (shared template)

Every README opens with the same five sections, in this order. Bundle owners customize the content; the section structure stays uniform.

```markdown
# <repo-name>

<one-line description>

## What this is

<2-3 sentences. What this repo is responsible for, mapped to the proposal's architecture.>

## How it fits

<diagram or a few lines showing where this repo sits relative to the other two and to the bundles. Reference discovery_engine_proposal.md for the full picture.>

## Public API

<the names and signatures of the modules/functions that other repos may import. Everything else is internal and may change without notice.>

## Getting started

<clone, install with uv, run tests. 5-line block at most.>

## Status

<one paragraph: what stage, what's working, what's stubbed, what's next. Owners update this as the bundle matures.>
```

After these five sections, owners can add whatever else is useful (architecture notes, glossary, contribution guide, etc.). The first five are non-negotiable.

---

## What CC should do

In order:

1. Create the three repos as local directories with the structure above. Use the shared template strictly.
2. Generate the stub schemas in `discovery-engine` (Candidate, Change) from `discovery_engine_proposal.md` §4.1 and the Signal types in `discovery-observability` from `signal_interface_spec.md`. Dataclasses, type-annotated, with docstrings drawn from the source documents.
3. Generate the README for each repo using the shared template. Fill in "What this is" and "How it fits" from the proposal. Leave "Status" with a placeholder reading "Stage 0 scaffolding. No functional code yet."
4. Generate the CI workflow files (identical across repos): lint with ruff, type-check with mypy, run pytest.
5. Generate `pyproject.toml` for each repo with the medium-opinion tooling: ruff, mypy, pytest, uv-managed. Include the inter-repo dependencies for `discovery-engine`.
6. Generate the `.pre-commit-config.yaml` (identical across repos).
7. Generate the `LICENSE` (Apache 2.0) and `.gitignore` (standard Python).
8. Initialize each repo with `git init` and make the first commit. Commit message: `Initial scaffolding (Stage 0).`
9. Verify each repo's tests run (even if there are no tests yet, `pytest` should report zero passed/zero failed cleanly). Verify ruff and mypy run without errors on the empty scaffold.

## What CC should not do

- Do not write any business logic. The bundles are owned by people; CC's job is to scaffold, not to begin implementation.
- Do not invent additional repos, subpackages, or directories. The structure above is exhaustive for Stage 0.
- Do not add dependencies beyond what is specified here. Owners add their own dependencies as they implement.
- Do not configure GitHub remotes. Local repos only; remote setup is a manual step the reviewer handles.
- Do not write the bundle charters. Those are drafted separately and added afterward.
- Do not generate sample tests, sample modules, or "hello world" examples. The scaffold is intentionally bare; owners add their first real code.
- Do not create more than the three repos specified, even if it seems natural to split further. The fourth repo (`discovery-experiments`) is deliberately deferred.

## Verification checklist

After CC reports completion, the reviewer should be able to:

- `cd` into each repo and run `uv sync && uv run pytest` successfully.
- `uv run ruff check .` cleanly.
- `uv run mypy .` cleanly.
- See identical CI workflow files across all three repos.
- See identical pyproject structure (with different dependencies) across all three repos.
- Import the schemas: `from discovery_engine.schemas import Candidate, Change`, `from discovery_observability.signals import TraceSummary, WorkloadProfile, Anomaly`, `from discovery_knowledge.retrieval import query`.
- See README files for all three repos following the shared template exactly.

If any of these fail, the scaffold is incomplete.

---

## Open decisions for the reviewer

These are left for Udi to resolve before the repos go to remote hosts:

1. **GitHub Enterprise vs github.com.** Affects how owners clone and how CI is configured. Default assumption is GitHub Enterprise (consistent with oh-my-review).
2. **Org name on the remote.** The repos will need a parent org.
3. **Whether to publish the schemas as a separately versioned package** (e.g., `discovery-engine-schemas`) to avoid having `discovery-observability` and `discovery-knowledge` depend on the full engine. For now, schemas live in `discovery-engine` and the leaf repos do not depend back on it; the question only matters if the leaf repos later need to import schemas, which they shouldn't.
