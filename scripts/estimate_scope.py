"""Measure a repository's structure well enough to feed the scoping advisor.

The advisor's only input is the in-scope module count, which normally comes from
`spotlights modules extract` -- an LLM step that costs money. For a free
up-front estimate, approximate it structurally: a module is a source directory
substantial enough that the extractor would name it.

Calibrated against the one repo where both numbers are known: RocksDB's
extractor produced 29 modules in the tree, and the thresholds below reproduce
that order of magnitude. The output is a RANGE of plausible module counts, not a
single figure, because the threshold is a judgement call and the advisor's
sensitivity to it should be visible rather than hidden.

Usage:
    python scripts/estimate_scope.py <repo-dir>
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

CODE_EXT = {
    ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".cu",
    ".cuh", ".rs", ".go", ".java", ".kt", ".scala", ".ts", ".tsx", ".js",
    ".jsx", ".mlir", ".td", ".proto", ".swift", ".m", ".mm",
}
SKIP_DIRS = {
    ".git", ".github", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist", ".tox",
    "third_party", "vendor", "external",
}
# Directories whose contents the engine is not asked to find findings in.
NON_PRODUCT = {"tests", "test", "testing", "docs", "doc", "examples", "example",
               "benchmarks", "tools", "scripts", "ci"}


def walk(root: Path) -> dict[str, tuple[int, int]]:
    """{dir_relpath: (code_files, code_loc)} counting each directory's OWN files."""
    out: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(".egg-info") and not d.startswith(".")
        ]
        files = loc = 0
        here = Path(dirpath)
        for name in filenames:
            if Path(name).suffix.lower() not in CODE_EXT:
                continue
            try:
                with (here / name).open("rb") as fh:
                    n = sum(1 for _ in fh)
            except OSError:
                continue
            files += 1
            loc += n
        rel = here.relative_to(root).as_posix() or "."
        if files:
            out[rel] = (files, loc)
    return out


def is_product(rel: str) -> bool:
    return not any(part in NON_PRODUCT for part in rel.split("/"))


def modules_at(dirs: dict[str, tuple[int, int]], min_loc: int) -> list[str]:
    """Directories big enough to be named as their own module."""
    return sorted(d for d, (_f, loc) in dirs.items() if loc >= min_loc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("--min-loc", type=int, default=1500,
                    help="LOC threshold for a directory to count as a module.")
    args = ap.parse_args()

    dirs = walk(args.repo)
    prod = {d: v for d, v in dirs.items() if is_product(d)}

    tot_f = sum(f for f, _ in dirs.values())
    tot_l = sum(l for _, l in dirs.values())
    prod_f = sum(f for f, _ in prod.values())
    prod_l = sum(l for _, l in prod.values())

    print(f"repo            {args.repo}")
    print(f"code files      {tot_f:,}   ({prod_f:,} outside tests/docs/tools)")
    print(f"code loc        {tot_l:,}   ({prod_l:,} outside tests/docs/tools)")
    print(f"source dirs     {len(dirs):,}  ({len(prod):,} product)")

    print("\nmodule count is threshold-dependent -- the advisor's only input:")
    print(f"  {'min loc/dir':>12}{'modules (product code)':>26}")
    for thr in (500, 1000, 1500, 3000, 5000):
        mods = modules_at(prod, thr)
        mark = "  <- default" if thr == args.min_loc else ""
        print(f"  {thr:>12}{len(mods):>26}{mark}")

    chosen = modules_at(prod, args.min_loc)
    print(f"\nat --min-loc {args.min_loc}, the {len(chosen)} candidate modules:")
    for d in sorted(chosen, key=lambda x: -prod[x][1]):
        f, l = prod[d]
        print(f"  {d:<44}{f:>5} files{l:>9,} loc")

    # Roll-up by top-level package, for orientation.
    top: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for d, (f, l) in dirs.items():
        t = d.split("/")[0]
        top[t][0] += f
        top[t][1] += l
    print(f"\n{'top level':<24}{'files':>7}{'loc':>10}")
    for t, (f, l) in sorted(top.items(), key=lambda kv: -kv[1][1]):
        tag = "" if is_product(t) else "   (not product code)"
        print(f"  {t:<22}{f:>7}{l:>10,}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
