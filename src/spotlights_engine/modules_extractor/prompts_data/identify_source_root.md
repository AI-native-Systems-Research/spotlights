<preamble>
You are running as a non-interactive subprocess. Your current working directory
is the target repository — do NOT modify it; read only.
- Return exactly one valid JSON object as your final assistant message.
- Do not write files. The orchestrator persists the result.
- No markdown fence, no commentary, no "DONE" sentinel.
</preamble>

You are analyzing this repository to determine its identity and the single
directory its product source lives under. This is stage 1 of a two-phase module
extractor: a later deterministic stage enumerates the directories below the
source root you choose, so choosing it correctly is critical.

## Steps

1. **Identify the project**
   - Read the root listing and any manifest files (package.json, pyproject.toml,
     Cargo.toml, go.mod, pom.xml, *.csproj, Gemfile, etc.) to determine
     language, framework, and build system.
   - Read the README and any docs/ or ARCHITECTURE.md for stated structure.

2. **Determine `source_root`** — the single repo-relative directory the
   package(s) live under. Every module's qualified name is derived relative to
   it. Decide it as follows:
   - src-layout (`src/<pkg>/…`, or a pyproject `package-dir` / `tool.setuptools`
     declaration): set `source_root` to that directory, e.g. `"src"` or `"lib"`.
   - packages at the repo root (e.g. vLLM-style `vllm/…`): set `source_root` to
     `""` (empty string = repo root).
   - For a flat `src/main.py` style project, choose `src`'s parent as the root
     (usually `""`) so `src` itself can be represented as a module.
   - For a monorepo with several unrelated packages, use the lowest common
     schema-compatible root (often `""`) rather than picking one package and
     dropping the others. The schema supports only one source root.
   - The selected root cannot itself be a module (the schema forbids a module
     path equal to `source_root`), so it must contain at least one
     source-bearing descendant directory that can be emitted.

3. **Account for source outside the modeled set** via `excluded_source_paths`.
   Each exclusion names a canonical repo-relative file or directory path, a
   bounded `explanation`, and a `reason` from this enum:
   - `centralized_tests` — a tests tree you are not modeling as product source.
   - `docs` — documentation source.
   - `examples` — example/sample code.
   - `benchmarks` — benchmark harnesses.
   - `tooling` — dev tooling / scripts.
   - `generated` — generated code (only with a concrete, testable reason).
   - `vendored` — third-party vendored code.
   - `not_product_source` — other audited non-product source.
   - `repository_level_file` — a source/entry file sitting directly at the
     selected root that the directory-only module schema cannot represent.

   Rules for exclusions:
   - Every non-ignored source-bearing path **outside** a non-empty `source_root`
     must be covered by an exclusion (listing an ancestor covers its
     descendants). Non-source configuration files do not need exclusions.
   - You may also exclude centralized tests, examples, generated code, vendored
     code, and other audited non-product source **inside** a common root
     (especially when `source_root == ""`).
   - Exclusions form an antichain (none nested under another), cannot equal the
     repository root or the selected source root, and cannot contain the
     selected root. Each exclusion must cover at least one real source file.
   - Every source file sitting directly at the selected root that cannot itself
     be a directory module must be classified with `repository_level_file` (or
     the decision fails).

## Output

Return ONLY this JSON, no markdown fences, no commentary:

{
  "repository": {
    "name": "string — repo name",
    "summary": "2–4 sentences: what the repo does, its project type, primary languages/frameworks, and high-level architectural shape.",
    "source_root": "repo-relative directory the package(s) live under; '' for repo root. No leading/trailing slash.",
    "external_dependencies": ["primary external packages/frameworks the repo depends on — flat list of names"]
  },
  "excluded_source_paths": [
    {
      "path": "repo/relative/path",
      "reason": "one of the enum values above",
      "explanation": "why this path is excluded from product-source modeling"
    }
  ]
}

## Rules
- All paths are repo-relative POSIX, no leading "./" or "/", no "..".
- `name` and `summary` must be non-empty. `external_dependencies` names must be
  unique and non-empty.
- Output valid JSON, parseable by `JSON.parse` — no trailing commas, no comments.
