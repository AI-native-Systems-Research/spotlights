# Modules Extractor — API

A function over a project directory that produces a structured map of its modules. Output mirrors [docs/modules.json](modules.json).

## Entrypoint

```python
from pathlib import Path
from spotlights_engine.modules_extractor import extract
from spotlights_engine.schemas.modules import ProjectTree

tree: ProjectTree = extract(root=Path("/path/to/repo"))
tree.to_json(Path("docs/modules.json"))
```

Synchronous, deterministic for a given input, returns an in-memory `ProjectTree`.

## Data model

Defined in [`spotlights_engine.schemas.modules`](../src/spotlights_engine/schemas/modules.py).

```python
class File(BaseModel):
    path: str
    role: str

class Module(BaseModel):
    name: str
    path: str
    description: str = ""
    depends_on: list[str] = []
    main_files: list[File] = []
    submodules: list["Module"] = []

class Repository(BaseModel):
    name: str
    summary: str
    external_dependencies: list[str] = []

class ProjectTree(BaseModel):
    repository: Repository
    modules: list[Module]
```

### `ProjectTree`

| Field | Description |
|---|---|
| `repository` | Repo-level metadata. |
| `modules` | Top-level modules, sorted by `name`. |

Methods:

- `to_json(path: Path) -> None` — Serialize the tree to `path` as indented JSON (UTF-8, trailing newline).
- `from_json(path: Path) -> ProjectTree` — Load a `ProjectTree` from a JSON file written by `to_json`. Round-trip safe; ordering preserved.
- `walk() -> Iterable[tuple[str, Module]]` — Preorder traversal yielding `(qualified_name, module)` for every module in the tree.
- `leaves() -> Iterable[tuple[str, Module]]` — Yield `(qualified_name, module)` for every module with no submodules, in `walk()` order.
- `resolve(qualified_name: str) -> Module | None` — Look up a module by its qualified name. A bare `name` (no `/`) is accepted only when exactly one module in the tree carries it; otherwise returns `None`.

#### Qualified names

A module's *qualified name* is the `/`-joined chain of `name`s from a top-level module down to it — e.g., `v1/engine/attention` for `attention` nested under `engine` nested under top-level `v1`. Top-level modules are referred to by their bare `name`. Qualified names are the canonical identifiers used in `depends_on` and `resolve()`; a bare `name` is accepted by `resolve()` only when exactly one module in the tree carries it.

### `Repository`

| Field | Description |
|---|---|
| `name` | Repository name. |
| `summary` | 2–5 sentence narrative description of the repo. |
| `external_dependencies` | Flat list of external package names (PyPI/etc.) the repo imports. |

### `Module`

| Field | Description |
|---|---|
| `name` | Editorial label, unique among siblings. Matches `^[a-z][a-z0-9_]*$`. Defaults to the basename of `path`. |
| `path` | Repo-relative path, forward slashes, no trailing slash. |
| `description` | Per-module context inlined verbatim into every Stage-1 candidate-discovery iteration. See [Description content requirements](#description-content-requirements). Empty when no describer is configured. |
| `depends_on` | Qualified names of modules this one depends on (e.g., `"v1/engine"`), or entries from `repository.external_dependencies`. |
| `main_files` | Key entry-point files that best represent the module's purpose. |
| `submodules` | Nested modules, same shape, sorted by `name`. |

### `File`

| Field | Description |
|---|---|
| `path` | Repo-relative path under the parent module's `path`. |
| `role` | Short phrase describing why this file matters (e.g., `"FlashAttention backend"`). |

## Description content requirements

`Module.description` is inlined verbatim into every iteration of the Stage-1
candidate-discovery prompt (see
[candidate_discovery/prompts.py](../src/spotlights_engine/candidate_discovery/prompts.py)).
Its job is to give the discovery agent enough per-module context to judge
*where the optimization headroom lives* without re-reading the whole repo.
When a describer is configured, it must produce ~3–6 sentences (or short
labeled lines), covering — in this order — every item below:

1. **What the module owns.** One sentence naming the concrete responsibility,
   distinct from `main_files` and `depends_on`.
2. **Role in the repo's main hot-path flows.** Which named flow(s) (those
   listed in the target repo's `docs/repo_context.md` §9) the module sits on,
   and *where* in the flow. If the module is not on any hot path, say so.
3. **Call frequency.** Per-request, per-token, per-step, per-block, once at
   startup, etc. Stage-1 uses this to weight "real headroom" claims; missing
   it forces the agent to guess.
4. **Headroom signals.** Two or three concrete classes of in-module knob the
   agent should look for — magic constants, dispatch tables, batching/tiling/
   launch parameters, eviction/admission policies, scoring/weighting
   formulas, fusion opportunities, sync points.
5. **Primary entry point(s).** Filename(s) plus the function/method the hot
   path crosses into the module through. The agent uses this to know where
   to start reading.

### Style

- Specificity beats fluency. "Called per-block during prefill, never on
  decode" tells the agent more than "manages blocks."
- Only state facts verified from source. Speculative claims poison every
  iteration that consumes them; Stage-1 cannot tell verified from invented.
- Cross-reference the repo-level flow names from `repo_context.md` §9 by
  name where applicable, so the agent can correlate.

### Anti-patterns (leave out)

- Architectural philosophy ("designed for extensibility", "decoupled by
  design").
- Repeating `main_files` or `depends_on` content — they render separately.
- Per-file detail beyond the entry point — the agent reads the files itself.
- In-flight refactors, deprecations, or aspirations — descriptions are a
  snapshot of stable state.

## Schema export

```python
ProjectTree.model_json_schema()  # JSON Schema, emitted to docs/modules.schema.json by CI
```

