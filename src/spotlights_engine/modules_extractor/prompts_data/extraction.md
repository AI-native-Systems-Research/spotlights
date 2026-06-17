<preamble>
You are running as a non-interactive subprocess. Your current working directory
is the target repository — do NOT modify it; read only.
- Return exactly one valid JSON object as your final assistant message.
- Do not write files. The orchestrator persists the result.
- No markdown fence, no commentary, no "DONE" sentinel.
</preamble>

You are analyzing this repository to produce a structured architectural map. Work through the steps below, then output ONLY the JSON described at the end — no prose before or after.

## Steps

1. **Identify the project**
   - Read the root listing and any manifest files (package.json, pyproject.toml, Cargo.toml, go.mod, pom.xml, *.csproj, Gemfile, etc.) to determine language, framework, and build system.
   - Read the README and any docs/ or ARCHITECTURE.md for stated structure.
   - Locate the source root(s): src/, lib/, app/, internal/, pkg/, or the language-default layout.
   - **Determine `source_root`** — the single repo-relative directory the package(s) live under, used to derive every module's qualified name (qualified name = module path relative to `source_root`). Decide it as follows:
     - src-layout (`src/<pkg>/…`, or a pyproject `package-dir` / `tool.setuptools` declaration): set `source_root` to that directory, e.g. `"src"` or `"lib"`.
     - packages at the repo root (e.g. vLLM-style `vllm/…`): set `source_root` to `""` (empty string = repo root).
   - Emit only modules **under** that source root. When `source_root` is non-empty, every module `path` must start with `source_root + "/"` (and must not equal `source_root` itself). Do not emit a centralized `tests/` directory that sits outside `src/` as a top-level module in this shape; mention tests in descriptions instead.

2. **Discover modules**
   - A "module" is a logical unit with a coherent responsibility — typically a top-level directory under the source root, a package, or a namespace.
   - A "submodule" is a nested logical unit inside a module. Nest recursively as deep as the architecture warrants.
   - Use directory structure as the primary signal, but VERIFY with imports/exports. Directories nothing imports from may be dead code; directories imported across the codebase are likely core.
   - Ignore: node_modules, vendor, .git, dist, build, target, __pycache__, .venv, generated code, test fixtures, lock files.
   - Tests: include a top-level "tests" module if tests are centralized; otherwise note testing per-module in its description.

   **For each candidate directory, pick exactly one of three outcomes — FOLD, LEAF, or SPLIT — by applying the tests below in order.**

   **Step A — FOLD?** Don't emit the directory as a submodule at all; cite its key files in the parent's `main_files`. Fold if ANY of these holds:
   - *Single file* — the directory has one source file.
   - *Organizational-only* — `types/`, `constants/`, `utils/`, `helpers/` and similar that exist only to keep the parent tidy, with no independent responsibility.

   Otherwise the directory will be emitted as a submodule. Continue to Step B to decide its shape.

   **Step B — SPLIT or LEAF?** Default is LEAF (one cohesive submodule). SPLIT into multiple sub-submodules only if a SPLIT test fires AND no LEAF override applies.

   SPLIT tests — fire if ANY is true:
   1. *Description test* — you cannot describe the directory in one sentence without "and" / "," / "also" joining two distinct responsibilities. ("parses queries AND executes them" → parser + executor.)
   2. *main_files overflow* — you cannot pick 1–5 files that meaningfully represent the unit. More than 5 needed → probably 2+ jobs.
   3. *Import-graph test* — files cluster: subset A imports mostly from A, subset B mostly from B, with only a thin bridge. The bridge is the seam.
   4. *External API test* — outside callers import from the directory in two clearly different ways (e.g. some import the parser, others import the AST types). Each import cluster is a sub-submodule.

   LEAF overrides — keep as one submodule even if a SPLIT test fires, if ANY holds:
   - *One concept, multiple files* — a data class + builder + validator for one entity is one unit, even if 6 files. `main_files` picks the 3–5 that matter; the rest are implementation detail.
   - *Strategy/plugin implementations of one interface* — `backends/postgres.py`, `backends/sqlite.py`, etc. are one submodule ("backends"), not one-per-file. `main_files` names the interface + 1–2 representative implementations.

   If you SPLIT, choose the seam from directory boundaries, import clusters, and external API usage, then recurse into each child starting from Step A. If you LEAF, stop.

3. **For each module, determine**
   - Its responsibility, in 1–2 sentences, grounded in actual source — read entry/index files and 2–4 representative files. If you can't justify a description from the code, write "unclear" rather than guessing.
   - Its main files: the 1–5 files most central to the module (entry points, primary classes/services, public API surface, route definitions, schema). Full paths from repo root.

4. **Output**

Return ONLY this JSON, no markdown fences, no commentary. The shape mirrors the
`ProjectTree` schema used downstream (`repository` + `modules`).

{
  "repository": {
    "name": "string — repo name",
    "summary": "2–4 sentences: what the repo does, its project type (e.g. 'Python ML library', 'Rust CLI tool', 'Node.js GraphQL API'), the primary languages/frameworks, and the high-level architectural shape (layered, hexagonal, monorepo with packages, microservices, MVC, etc.)",
    "source_root": "repo-relative directory the package(s) live under, used to derive qualified names (qualified name = module path relative to source_root). 'src' for a src-layout, '' (empty) when packages sit at the repo root, 'lib' if that is the package dir. No leading/trailing slash.",
    "external_dependencies": ["primary external packages, frameworks, and key libraries the repo depends on — flat list of names"]
  },
  "modules": [
    {
      "name": "string — must match ^[a-z][a-z0-9_]*$; defaults to the basename of `path`",
      "path": "path/from/repo/root",
      "description": "1–2 sentences on responsibility",
      "depends_on": ["source-root-relative qualified names of other modules it imports from (e.g. 'vllm/attention' for a root-layout repo, 'spotlights_engine/schemas' for a src-layout one), or names of entries from repository.external_dependencies"],
      "main_files": [
        { "path": "full/path/from/repo/root.ext", "role": "what this file does" }
      ],
      "submodules": [ <LEAF or PARENT>, ... ]
    }
  ]
}

A submodule has exactly two shapes. Pick one per submodule based on the SPLIT tests above.

LEAF — the directory is one cohesive unit, no further nesting warranted:
{
  "name": "string",
  "path": "path/from/repo/root",
  "description": "1 sentence. Must not contain ' and ' joining two responsibilities.",
  "main_files": [ { "path": "...", "role": "..." } ]
}

PARENT — the directory contains 2+ logical units that each deserve their own entry:
{
  "name": "string",
  "path": "path/from/repo/root",
  "description": "1 sentence describing the umbrella responsibility.",
  "main_files": [ { "path": "...", "role": "..." } ],
  "submodules": [ <LEAF or PARENT>, ... ]
}

## Rules
- All paths are relative to the repo root, no leading "./" or "/".
- Module/submodule `name` must match `^[a-z][a-z0-9_]*$` — lowercase, starts with a letter, only letters/digits/underscore — and must equal the normalized basename of `path`. If the directory name violates the pattern (e.g. has dashes, dots, or a leading digit), normalize by lowercasing and replacing offending characters with underscores (a leading digit gets an `m_` prefix); the human-readable basename still goes in `path`. The same per-segment normalization is applied to every segment of the derived qualified name.
- A LEAF submodule has no `submodules` key at all (or an empty list). A PARENT submodule has `submodules` with ≥2 entries. Never emit a parent with only one child — collapse it. `main_files` on a parent lists files that belong to the parent itself, not files that belong to any child.
- `main_files` is required and 1–5 entries, on every module and every submodule. If you cannot name a central file, it isn't a (sub)module.
- Top-level modules use the full schema with `depends_on`. Submodules omit `depends_on` (it is implied by the parent module's dependencies).
- Aim for 5–15 top-level modules. If the repo is genuinely flatter or larger, follow the code.
- `depends_on` lists peer top-level modules by their source-root-relative qualified name (e.g. `vllm/attention`, `spotlights_engine/schemas` — not the bare `attention`) and/or entries from `repository.external_dependencies`. Empty array if none.
- Output valid JSON, parseable by `JSON.parse` — no trailing commas, no comments.
