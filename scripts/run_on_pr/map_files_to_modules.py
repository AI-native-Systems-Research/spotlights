#!/usr/bin/env python3
"""map_files_to_modules.py — map changed files to engine module qualified names.

Given the modules extractor's `ProjectTree` JSON and the PR's changed *source*
files (base-side paths from `ground_truth.json`), return the set of slash-form
module qualified names to pass to `spotlights-engine --include`.

The one real transform here (design "qn derivation, not skew"): a changed file
is matched against each module's **repo-relative `path`**, but the emitted value
is the qn returned by `ProjectTree.walk()` for that module — never a hand-built
string — so `source_root` stripping and per-segment normalization stay in
lockstep with `project.py:_qualified_name`.

Matching is segment-aware: a file belongs to a module when
`file == module.path` or `file.startswith(module.path + "/")` (so `pkg/cache`
does not spuriously match `pkg/cache_v2/x.py`). Each file is assigned to its
**most-specific (deepest) containing module** — the one with the longest
matching `path`. A file under no module is `unmapped`; a file whose deepest
match is tied between two modules of equal path length is `ambiguous` (should
not happen given unique paths, but reported defensively).

Emits:
    {
      "include": ["vllm/v1/kv_offload", ...],   # sorted, deduped qns to scope to
      "unmapped_files": [...],                    # changed files under no module
      "ambiguous_files": [...],                   # files with a tied deepest match
      "file_module": { "<file>": "<qn>", ... }    # winning qn per mapped file
    }

The skill's fallback policy (run all modules when include is empty / any file is
unmapped or ambiguous) lives in the agent, not here — this script only reports
the mapping facts.

Usage:
    map_files_to_modules.py --tree project_tree.json --files a.py b.py
    map_files_to_modules.py --tree project_tree.json --files-json ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the engine's ProjectTree so qn derivation matches the engine exactly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spotlights_engine.schemas.project import ProjectTree  # noqa: E402


def _norm(path: str) -> str:
    return path.strip().strip("/")


def map_files(tree: ProjectTree, changed_files: list[str]) -> dict:
    # (qn, normalized_path) for every module, from the engine's own walk().
    modules = [(qn, _norm(m.path)) for qn, m in tree.walk()]

    include: set[str] = set()
    unmapped: list[str] = []
    ambiguous: list[str] = []
    file_module: dict[str, str] = {}

    for raw in changed_files:
        f = _norm(raw)
        # Candidate modules that contain this file (segment-aware).
        matches = [
            (qn, mpath)
            for qn, mpath in modules
            if f == mpath or f.startswith(mpath + "/")
        ]
        if not matches:
            unmapped.append(raw)
            continue
        # Deepest = longest module path. Detect a tie at the max depth.
        max_len = max(len(mpath) for _, mpath in matches)
        deepest = [qn for qn, mpath in matches if len(mpath) == max_len]
        if len(deepest) > 1:
            ambiguous.append(raw)
            continue
        qn = deepest[0]
        include.add(qn)
        file_module[raw] = qn

    return {
        "include": sorted(include),
        "unmapped_files": sorted(unmapped),
        "ambiguous_files": sorted(ambiguous),
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
    ap = argparse.ArgumentParser(description="Map changed files to module qns.")
    ap.add_argument("--tree", required=True, help="ProjectTree JSON (modules extractor output)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs="*", help="Changed source file paths")
    src.add_argument(
        "--files-json",
        help="Path to ground_truth.json; reads changed_source_files from it",
    )
    ap.add_argument("-o", "--output", default=None, help="Write JSON here (default: stdout)")
    args = ap.parse_args()

    tree = ProjectTree.from_json(Path(args.tree))
    changed_files = _load_changed_files(args)
    out = map_files(tree, changed_files)

    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
