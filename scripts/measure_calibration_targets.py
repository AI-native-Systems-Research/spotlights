"""Measure the structural size of every historical run's in-scope modules.

The naive `tokens = modules x constant` model failed its backtest (median 65%
error) because tokens/module spans 23x across targets -- a big dense C++ module
costs far more than a small Python one. This script produces the missing
target-side signal: for each completed run, how large were the modules it
actually analyzed, measured against the exact tree the run saw.

Each run's `result.json` gives `report.project_tree` (module qualified name ->
repo-relative directory) and `module_runs` (which modules ran, with status).
Sizes are read from a clone pinned to that run's `target.commit_sha`.

Two sizes are recorded per module, because nested modules overlap -- rocksdb's
`db` contains `db/compaction`, and counting both inflates the total:

  recursive  -- everything under the module directory
  own        -- recursive minus any subdirectory that is itself a module

Emits a JSON dataset for the model fit; measures nothing about cost or tokens.

Usage:
    python scripts/measure_calibration_targets.py \
        --paper-runs /c/projects/vs_code/spotlights-paper/experiments/full-runs \
        --clones /c/temp/spotlights-calib \
        --out artifacts/calibration_features.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BILLED_STATUSES = ("SUCCEEDED", "DEGRADED", "FAILED")

# Extensions counted as source. Deliberately narrow: build junk, lock files and
# generated blobs would swamp LOC on repos like vllm without predicting tokens.
CODE_EXT = {
    ".py", ".pyx", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
    ".cu", ".cuh", ".go", ".java", ".kt", ".scala", ".rs", ".rb", ".js", ".jsx",
    ".ts", ".tsx", ".swift", ".m", ".mm", ".sh", ".bash", ".proto", ".sql",
    ".pl", ".php", ".cs", ".ipynb",
}
DOC_EXT = {".md", ".rst", ".txt", ".adoc"}
CFG_EXT = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
             ".pytest_cache", "dist", "build", ".tox", ".idea", ".eggs"}

# Repo-name -> clone directory name, where they differ.
CLONE_ALIASES = {
    "iocr": "IOCR",
}


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def git(dir_: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(dir_), *args],
        capture_output=True, text=True, check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


@dataclass
class Size:
    files: int = 0
    loc: int = 0
    bytes: int = 0
    code_files: int = 0
    code_loc: int = 0
    doc_loc: int = 0
    cfg_loc: int = 0

    def add(self, path: Path) -> None:
        ext = path.suffix.lower()
        try:
            raw = path.read_bytes()
        except OSError:
            return
        if b"\0" in raw[:8000]:  # binary
            self.files += 1
            self.bytes += len(raw)
            return
        n = raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
        self.files += 1
        self.bytes += len(raw)
        self.loc += n
        if ext in CODE_EXT:
            self.code_files += 1
            self.code_loc += n
        elif ext in DOC_EXT:
            self.doc_loc += n
        elif ext in CFG_EXT:
            self.cfg_loc += n


def measure(root: Path, exclude: set[Path]) -> Size:
    """Count everything under `root`, skipping the `exclude` subtrees."""
    s = Size()
    if not root.is_dir():
        return s
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_symlink():
                continue
            if e.is_dir():
                if e.name in SKIP_DIRS or e in exclude:
                    continue
                stack.append(e)
            elif e.is_file():
                s.add(e)
    return s


def flatten_tree(modules: list[dict], prefix: str = "") -> dict[str, str]:
    """project_tree -> lookup accepting either key `module_runs` might use.

    Manifest generations disagree: rocksdb keys `module_runs` by the slash-form
    qualified name (`db/compaction`), while vllm and colpali key it by the
    repo-relative path (`vllm/attention`). Both resolve to the same directory,
    so index under both and let the caller look up whichever it has.
    """
    out: dict[str, str] = {}
    for m in modules or []:
        qn = f"{prefix}{m.get('name') or ''}"
        path = m.get("path")
        if path:
            out[qn] = path
            out.setdefault(path, path)
        out.update(flatten_tree(m.get("submodules") or [], qn + "/"))
    return out


@dataclass
class RunFeatures:
    run: str
    repo: str
    commit: str
    clone_ok: bool
    total_tokens: int = 0
    wall_s: float | None = None
    api_s: float | None = None
    engine_sha: str = ""
    n_modules_billed: int = 0
    n_modules_in_tree: int = 0
    unmatched_modules: list[str] = field(default_factory=list)
    # Summed over billed modules only.
    scope: dict[str, int] = field(default_factory=dict)
    repo_total: dict[str, int] = field(default_factory=dict)
    modules: list[dict] = field(default_factory=list)


def repo_key(url: str, fallback: str) -> str:
    if not url:
        return fallback
    return url.rstrip("/").removesuffix(".git").split("/")[-1].split(":")[-1].lower()


def collect_runs(root: Path) -> list[dict]:
    runs = []
    for res in sorted(root.glob("**/result.json")):
        d = _load(res)
        man_p = res.parent / "run_manifest.json"
        man = _load(man_p) if man_p.exists() else {}
        target = man.get("target") or {}
        url = target.get("repo_url") or ""
        sha = target.get("commit_sha") or ""
        if not url or not sha:
            rep = (d.get("report") or {}).get("repository") or {}
            url = url or rep.get("url") or ""
            sha = sha or rep.get("commit") or ""
        runs.append(
            {
                "name": str(res.parent.relative_to(root)).replace("\\", "/"),
                "url": url,
                "sha": sha,
                "result": d,
                "manifest": man,
            }
        )
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-runs", type=Path, required=True)
    ap.add_argument("--clones", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--no-checkout",
        action="store_true",
        help="never move a clone's HEAD; measure whatever is checked out",
    )
    args = ap.parse_args()

    runs = collect_runs(args.paper_runs)
    print(f"{len(runs)} runs with result.json under {args.paper_runs}")

    # Group by (clone, sha) so each checkout happens once.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        key = repo_key(r["url"], r["name"].split("/")[0].lower())
        r["repo"] = key
        groups[(key, r["sha"])].append(r)

    print("\n(repo, commit) pairs needed:")
    for (repo, sha), rs in sorted(groups.items()):
        print(f"  {repo:<36} {sha[:10] or '(none)':<11} {len(rs)} run(s)")

    out: list[RunFeatures] = []
    for (repo, sha), rs in sorted(groups.items()):
        clone = args.clones / CLONE_ALIASES.get(repo, repo)
        ok = clone.is_dir() and (clone / ".git").exists()
        if ok and sha and not args.no_checkout:
            have = git(clone, "cat-file", "-t", sha) == "commit"
            if not have:
                print(f"\n!! {repo}: commit {sha[:10]} not in {clone}")
                ok = False
            else:
                cur = git(clone, "rev-parse", "HEAD")
                if not cur.startswith(sha.rstrip()[:10].rstrip()):
                    print(f"\n== {repo}: checkout {sha[:10]} (was {cur[:10]})")
                    if not git(clone, "checkout", "-q", sha) == "":
                        pass
                    now = git(clone, "rev-parse", "HEAD")
                    if not now.startswith(sha[:10]):
                        print(f"   FAILED to check out {sha[:10]} (HEAD={now[:10]})")
                        ok = False
        elif not ok:
            print(f"\n!! {repo}: no clone at {clone}")

        for r in rs:
            fe = build_features(r, clone if ok else None, repo, sha)
            out.append(fe)
            print(
                f"  {fe.run:<44} mods={fe.n_modules_billed:<3} "
                f"code_loc(own)={fe.scope.get('own_code_loc', 0):>9,} "
                f"files={fe.scope.get('own_code_files', 0):>6,} "
                f"tok={fe.total_tokens / 1e6:>7.1f}M"
                + (f"  UNMATCHED={len(fe.unmatched_modules)}" if fe.unmatched_modules else "")
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump([asdict(f) for f in out], fh, indent=2)
    print(f"\nwrote {args.out}  ({len(out)} runs)")
    return 0


def build_features(r: dict, clone: Path | None, repo: str, sha: str) -> RunFeatures:
    d = r["result"]
    man = r["manifest"]
    pt = ((d.get("report") or {}).get("project_tree") or {})
    qn_to_path = flatten_tree(pt.get("modules") or [])
    module_runs = d.get("module_runs") or {}

    timing = man.get("timing") or {}
    fe = RunFeatures(
        run=r["name"],
        repo=repo,
        commit=sha[:10],
        clone_ok=clone is not None,
        total_tokens=man.get("total_tokens") or 0,
        wall_s=timing.get("wall_clock_s"),
        api_s=timing.get("api_time_s"),
        engine_sha=((man.get("spotlights") or {}).get("commit_sha") or "")[:8],
        n_modules_in_tree=len(set(qn_to_path.values())),
    )

    billed = []
    for qn, mr in module_runs.items():
        status = (mr or {}).get("status") if isinstance(mr, dict) else None
        if status is None or str(status).upper() in BILLED_STATUSES:
            billed.append(qn)
    fe.n_modules_billed = len(billed)

    if clone is None:
        return fe

    # Only a *billed* nested module was analyzed in its own right, so only those
    # subtrees are subtracted when computing a module's own size.
    billed_dirs = {clone / qn_to_path[q] for q in billed if q in qn_to_path}
    scope: dict[str, int] = defaultdict(int)

    for qn in sorted(billed):
        rel = qn_to_path.get(qn)
        if rel is None:
            fe.unmatched_modules.append(qn)
            continue
        d_ = clone / rel
        if not d_.is_dir():
            fe.unmatched_modules.append(f"{qn} -> {rel} (missing)")
            continue
        rec = measure(d_, exclude=set())
        nested = {x for x in billed_dirs if x != d_ and _is_under(x, d_)}
        own = measure(d_, exclude=nested) if nested else rec
        fe.modules.append(
            {
                "qn": qn,
                "path": rel,
                "recursive": asdict(rec),
                "own": asdict(own),
                "nested_modules": len(nested),
            }
        )
        for k, v in asdict(own).items():
            scope[f"own_{k}"] += v
        for k, v in asdict(rec).items():
            scope[f"rec_{k}"] += v

    fe.scope = dict(scope)
    fe.repo_total = asdict(measure(clone, exclude=set()))
    return fe


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
