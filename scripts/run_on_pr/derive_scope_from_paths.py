#!/usr/bin/env python3
"""derive_scope_from_paths.py — derive the `--include` scope straight from the
PR's changed source-file *paths*, with no modules extractor in the loop.

A "module", for scoping purposes, is simply **the folder the changed source
lives in**, expressed as a `source_root`-relative, per-segment-normalized path.
The engine treats such a value as a *virtual prefix* (see
`spotlights_manager/filters.py:apply_filter`): it expands to every real module
nested beneath that folder. So passing `vllm/v1/kv_offload` scopes the run to
that directory's modules without ever asking the LLM extractor what the modules
are.

Why this is safe / correct:
- We reuse the engine's own `_normalize_source_root` and
  `_normalize_module_segment`, so the qn we emit is byte-for-byte what the
  engine derives from a module whose `path` is that folder.
- `source_root` is inferred deterministically to match the engine: a common
  wrapper folder (`"src"`, or `"python"` for sglang's `python/sglang/…` layout)
  when *every* changed source file lives under that same top-level folder, else
  `""` (repo-root layout, e.g. vLLM's `vllm/…`).
- A file that sits directly at the source root (e.g. `src/foo.py`, whose folder
  *is* the source root) yields no sub-folder qn; it is reported in
  `root_level_files`. The agent turns any such case into the all-modules
  fallback rather than emitting an empty/attic scope.

Emits:
    {
      "source_root": "src" | "",
      "include": ["vllm/v1/kv_offload", ...],   # sorted, deduped folder qns
      "root_level_files": [...],                  # files whose folder == source_root
      "file_module": { "<file>": "<qn>", ... }    # folder qn per mapped file
    }

The fallback *policy* (when to run all modules) lives in the agent; this script
only reports the path-derived facts.

Usage:
    derive_scope_from_paths.py --files a/b.py c/d.py
    derive_scope_from_paths.py --files-json ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

# Reuse the engine's normalization so path-derived qns match the engine exactly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spotlights_engine.schemas.project import (  # noqa: E402
    _normalize_module_segment,
    _normalize_source_root,
)


def _norm(path: str) -> str:
    return path.strip().strip("/")


# Top-level folders the LLM extractor commonly promotes to `source_root` so the
# emitted qns drop the wrapper (e.g. `src/` in most repos, `python/` in sglang's
# `python/sglang/…` layout). We mirror that here — the deterministic script has
# no LLM, so it must recognize the same wrappers or the derived `--include` qns
# won't match the engine's real modules. Ordered by specificity is irrelevant;
# only one can apply since the check requires *every* file under the same root.
_SOURCE_ROOT_CANDIDATES = ("src", "python")


def _infer_source_root(changed_files: list[str]) -> str:
    """Mirror the engine's `source_root` handling: strip a common wrapper folder
    (`src/` or `python/`) iff *every* changed file lives under that same
    top-level folder, else `""`.

    `ProjectTree._infer_source_root_when_omitted` only *infers* `"src"`, but the
    LLM extractor sets `source_root` explicitly for other layouts (e.g. sglang's
    `python/`). We recognize those wrappers here so the path-derived qns match
    the modules the engine actually emits."""
    norm = [_norm(f) for f in changed_files if _norm(f)]
    if not norm:
        return ""
    for root in _SOURCE_ROOT_CANDIDATES:
        if all(f.startswith(root + "/") for f in norm):
            return root
    return ""


def _folder_qn(file_path: str, source_root: str) -> str | None:
    """qn of the *folder* containing `file_path`, `source_root`-relative and
    per-segment normalized. Returns None when that folder is the source root
    itself (no meaningful sub-folder to scope to)."""
    folder = str(PurePosixPath(_norm(file_path)).parent)
    if folder in ("", "."):
        rel = ""
    elif not source_root:
        rel = folder
    elif folder == source_root:
        rel = ""
    elif folder.startswith(source_root + "/"):
        rel = folder[len(source_root) + 1 :]
    else:
        # File is not under the inferred source_root (mixed layout); treat its
        # whole folder path as the qn rather than dropping it.
        rel = folder
    parts = [seg for seg in rel.split("/") if seg]
    if not parts:
        return None
    return "/".join(_normalize_module_segment(seg) for seg in parts)


def derive(changed_files: list[str]) -> dict:
    source_root = _normalize_source_root(_infer_source_root(changed_files))
    include: set[str] = set()
    root_level: list[str] = []
    file_module: dict[str, str] = {}

    for raw in changed_files:
        if not _norm(raw):
            continue
        qn = _folder_qn(raw, source_root)
        if qn is None:
            root_level.append(raw)
            continue
        include.add(qn)
        file_module[raw] = qn

    return {
        "source_root": source_root,
        "include": sorted(include),
        "root_level_files": sorted(root_level),
        "file_module": file_module,
    }


def _load_changed_files(args: argparse.Namespace) -> list[str]:
    if args.files_json:
        data = json.loads(Path(args.files_json).read_text(encoding="utf-8"))
        files = data.get("changed_source_files", [])
        if not isinstance(files, list):
            raise ValueError("--files-json must contain a list at changed_source_files")
        return list(files)
    return list(args.files or [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive --include scope from changed-file paths.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs="*", help="Changed source file paths")
    src.add_argument(
        "--files-json",
        help="Path to ground_truth.json; reads changed_source_files from it",
    )
    ap.add_argument("-o", "--output", default=None, help="Write JSON here (default: stdout)")
    args = ap.parse_args()

    out = derive(_load_changed_files(args))
    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
