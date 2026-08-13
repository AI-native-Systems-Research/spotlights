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
   - Open two or three real source files from different top-level directories
     and read their import / package / namespace / include statements. These are
     the ground truth for step 2 — manifests and READMEs describe intent, but
     the import statements show where qualified names actually resolve from.

2. **Determine `source_root`** — the single repo-relative directory that the
   language's qualified names are resolved *relative to*. Every module's
   qualified name is derived relative to it, so `source_root` is a **container
   of namespaces, never a namespace itself**.

   **Decision procedure (applies to every language):**

   a. Find the **namespace-root directories**: the top-level directories from
      which the language/build system resolves a source file's fully-qualified
      name (import path, package declaration, namespace, crate module path,
      include path). These are *product code*, so they must remain modelable as
      modules.
   b. Set `source_root` to their **common parent directory**. If that parent is
      the repository root, set `source_root` to `""` (empty string).
   c. Verify with the **qualified-name test**: take a representative source file
      deep in the tree and write out its fully-qualified name as the language
      states it. The directory segments of that name, joined, must equal the
      file's path **relative to `source_root`**. If your candidate root swallows
      leading segments of the qualified name, it is one (or more) levels too
      deep — move up. Re-run the test on a second file in a different top-level
      directory; both must agree.

   **Disqualifying signals — the candidate is a module, not the root.** Move up
   one level if the candidate directory:
   - is named after the distribution/package/crate/module declared in a manifest
     (`[project] name`, `name` in `Cargo.toml`/`package.json`, `artifactId`,
     `module` in `go.mod`, gemspec name);
   - is the first segment of the project's own import statements
     (`import colpali_engine.models…`, `use mycrate::…`,
     `import com.example.foo…`, `#include "mylib/foo.h"`);
   - carries a marker showing it *is* a package rather than a container of
     packages — `__init__.py` directly inside it, or a manifest
     (`package.json`, `Cargo.toml`, `go.mod`, `*.gemspec`, `pom.xml`) that
     declares the directory itself as the package/crate/module.

   A manifest at the **repository root** is not a disqualifying signal — it
   declares the repo, and the repo root is already the shallowest candidate.
   Nor is a crate/entry file such as `src/lib.rs`, `src/main.rs`, or
   `lib/index.ts` sitting directly in an otherwise container-shaped directory:
   such files are handled by `repository_level_file`, not by moving the root.

   **Language cheatsheet** (illustrative; apply the procedure, not the table):
   - **Python** — root layout `pkg/…` at the repo root → `""` (the package dir
     `pkg` is a *module*, not the root). src layout `src/pkg/…`, or a
     `package-dir` / `tool.setuptools` / `tool.hatch.build` declaration →
     `"src"` (or whatever that directory is).
   - **Java / Kotlin / Scala** — the package declaration is materialized as
     directories, so the root is the directory the package path starts under:
     Maven/Gradle `src/main/java` (or `src/main/scala`, `src/main/kotlin`);
     `com/…` is then a module. A multi-project Gradle/sbt build with
     `moduleA/src/main/java/…` has no single such root → use the common parent
     (often `""`).
   - **Go** — `go.mod` declares the module path and packages resolve relative to
     the directory holding it → `""` for a repo-root `go.mod`; the directory of
     a nested `go.mod` otherwise. `cmd/`, `internal/`, `pkg/` are modules.
   - **Rust** — a single crate resolves `crate::…` from `src/` → `"src"`. A
     Cargo workspace (`crates/<name>/src/…`, `members = [...]`) has one root per
     crate → use the common parent (often `""` or the workspace dir).
   - **C / C++** — no language-level namespace directories; use the include
     convention. If headers are included as `#include "proj/foo.h"` resolved
     against `include/`, the root is `"include"` only when all product source
     lives there; when code is split across `src/` and `include/` (or `lib/`),
     their common parent — usually `""` — is the root.
   - **C# / .NET** — `src/<Project>/…` next to a `.sln` → `"src"`; a single
     project at the repo root → `""`.
   - **JavaScript / TypeScript** — single package `src/…` → `"src"` (respecting
     `rootDir`/`baseUrl` in `tsconfig.json`); a `packages/*` workspace →
     the common parent (`""` or `"packages"`).
   - **Ruby** — `lib/<gem>/…` → `"lib"`. **PHP** — the PSR-4 autoload prefix
     directory from `composer.json`.

   **Additional constraints:**
   - For a flat `src/main.py`-style project (source files sitting directly in
     `src` with no package subdirectories), choose `src`'s parent (usually `""`)
     so `src` itself can be represented as a module.
   - For a monorepo with several unrelated packages, use the lowest common
     schema-compatible root (often `""`) rather than picking one package and
     dropping the others. The schema supports only one source root.
   - The selected root cannot itself be a module (the schema forbids a module
     path equal to `source_root`), so it must contain at least one
     source-bearing descendant directory that can be emitted.
   - **Prefer the shallowest defensible root.** A non-empty `source_root` makes
     every source-bearing sibling tree (tests, examples, scripts, tooling)
     unmodelable and forces an exclusion for each, and it strips the leading
     namespace segment from every qualified name. When two candidates both pass
     the qualified-name test, choose the shallower one; when genuinely
     undecidable, choose `""`.

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
   - Source files sitting directly at the selected root cannot themselves be
     directory modules. You should classify the ones you notice with
     `repository_level_file`, but the orchestrator also injects this
     classification deterministically for any it detects, so you do not need to
     enumerate every one exhaustively.

## Output

Return ONLY this JSON, no markdown fences, no commentary:

{
  "repository": {
    "name": "string — repo name",
    "summary": "2–4 sentences: what the repo does, its project type, primary languages/frameworks, and high-level architectural shape.",
    "source_root": "repo-relative container directory that qualified names resolve relative to — the parent of the top-level namespace/package directories, never a package/crate/namespace directory itself; '' for repo root. No leading/trailing slash."
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
- `name` and `summary` must be non-empty.
- Output valid JSON, parseable by `JSON.parse` — no trailing commas, no comments.
