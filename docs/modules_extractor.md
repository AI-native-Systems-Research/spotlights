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

- `to_json(path)` / `from_json(path)` — round-trip safe; ordering preserved.
- `walk() -> Iterable[tuple[str, Module]]` — preorder traversal yielding `(qualified_name, module)`.
- `resolve(qualified_name: str) -> Module | None` — qualified-name lookup. Bare names accepted only when unambiguous.

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
| `description` | 1–3 sentence description of what the module owns. Empty when no describer is configured. |
| `depends_on` | Qualified names of modules this one depends on (e.g., `"v1/engine"`), or entries from `repository.external_dependencies`. |
| `main_files` | Curated 3–7 files a reader should open first. |
| `submodules` | Nested modules, same shape, sorted by `name`. |

### `File`

| Field | Description |
|---|---|
| `path` | Repo-relative path under the parent module's `path`. |
| `role` | Short phrase describing why this file matters (e.g., `"FlashAttention backend"`). |

## Schema export

```python
ProjectTree.model_json_schema()  # JSON Schema, emitted to docs/modules.schema.json by CI
```

