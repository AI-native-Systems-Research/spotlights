"""Run the modules_extractor (step 1) against a configurable set of repos.

Batch testing/eval tool: loops the single-repo runner's body over a YAML
repo-set config, writing each repo's results into its own sub-folder of one
output root. One repo failing does not kill the batch; re-running skips repos
that already succeeded.

Examples:
    uv run --no-sync python scripts/run_modules_extractor_batch.py
    uv run --no-sync python scripts/run_modules_extractor_batch.py --dry-run
    uv run --no-sync python scripts/run_modules_extractor_batch.py \\
        --only skydiscover --force

Note on --max-parallel-repos: the extractor already fans out up to
`max_parallel_enrich_shards` (default 5) Claude subprocesses per repo, so
batch-level parallelism multiplies subprocess count and API load fast.
Sequential (the default) is the recommended mode.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.modules_extractor import (
    ExtractorConfig,
    ModulesExtractorError,
    extract_with_telemetry,
)
from spotlights_engine.schemas.pipeline import ModulesExtractorInput

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "scripts" / "extractor_batch_repos.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "modules_extractor_batch"

_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_SHA_RE = re.compile(r"[0-9a-fA-F]{7,40}")

# ── Config models ──────────────────────────────────────────────────────────

_EXTRACTOR_FIELDS = frozenset(ExtractorConfig.model_fields)
# Owned by the batch script itself; never configurable per repo/batch.
_RESERVED_FIELDS = frozenset({"artifacts_dir"})


def _validate_extractor_overrides(where: str, values: dict[str, Any]) -> None:
    """Fail fast on unknown/reserved ExtractorConfig keys or bad values."""
    unknown = sorted(set(values) - _EXTRACTOR_FIELDS)
    if unknown:
        raise ValueError(f"{where}: unknown ExtractorConfig field(s): {', '.join(unknown)}")
    reserved = sorted(set(values) & _RESERVED_FIELDS)
    if reserved:
        raise ValueError(
            f"{where}: field(s) owned by the batch script, not configurable: {', '.join(reserved)}"
        )
    # Value validation (types/ranges) via the model itself.
    ExtractorConfig(**values)


class RepoSpec(BaseModel):
    """One target repo: a unique name plus exactly one of `path` / `url`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: Path | None = None
    url: str | None = None
    ref: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> RepoSpec:
        if not _NAME_RE.fullmatch(self.name) or self.name in {".", ".."}:
            raise ValueError(
                f"repo name {self.name!r} is not filesystem-safe (allowed: [A-Za-z0-9._-]+)"
            )
        if self.name == "_clones":
            raise ValueError("repo name '_clones' is reserved for the clone area")
        if (self.path is None) == (self.url is None):
            raise ValueError(f"repo {self.name!r}: exactly one of `path` / `url` is required")
        if self.ref is not None and self.url is None:
            raise ValueError(f"repo {self.name!r}: `ref` is only valid with `url`")
        _validate_extractor_overrides(f"repo {self.name!r} overrides", self.overrides)
        return self


class BatchConfig(BaseModel):
    """Whole config file: batch-wide `defaults` plus the `repos` list."""

    model_config = ConfigDict(extra="forbid")

    defaults: dict[str, Any] = Field(default_factory=dict)
    repos: list[RepoSpec]

    @model_validator(mode="after")
    def _check(self) -> BatchConfig:
        _validate_extractor_overrides("defaults", self.defaults)
        if not self.repos:
            raise ValueError("`repos` must list at least one repo")
        names = [r.name for r in self.repos]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate repo names: {', '.join(dupes)}")
        return self


def load_batch_config(path: Path) -> BatchConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    return BatchConfig.model_validate(data)


def build_effective_config(
    spec: RepoSpec,
    batch: BatchConfig,
    *,
    artifacts_dir: Path,
    claude_bin: str | None,
) -> ExtractorConfig:
    """defaults -> per-repo overrides -> CLI --claude-bin, on ExtractorConfig."""
    merged: dict[str, Any] = {**batch.defaults, **spec.overrides}
    if claude_bin is not None:
        merged["claude_bin"] = claude_bin
    return ExtractorConfig(artifacts_dir=artifacts_dir, **merged)


# ── Repo resolution / clone-on-demand ──────────────────────────────────────


def _run_git(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git command failed ({' '.join(cmd)}): {proc.stderr.strip()}")


def clone_dir_for(spec: RepoSpec, output_root: Path) -> Path:
    return output_root / "_clones" / spec.name


def resolve_repo_path(spec: RepoSpec, *, output_root: Path, force: bool) -> Path:
    """Local path (resolved against the project root) or clone-on-demand."""
    if spec.path is not None:
        p = spec.path if spec.path.is_absolute() else PROJECT_ROOT / spec.path
        return p.resolve()

    assert spec.url is not None
    dest = clone_dir_for(spec, output_root)
    if dest.exists():
        if not force:
            return dest  # reuse existing clone as-is (no fetch)
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if spec.ref is not None and _SHA_RE.fullmatch(spec.ref):
        # A SHA cannot be `--branch`-ed on a shallow clone: full clone + checkout.
        _run_git(["git", "clone", spec.url, str(dest)])
        _run_git(["git", "-C", str(dest), "checkout", "--detach", spec.ref])
    else:
        cmd = ["git", "clone", "--depth", "1"]
        if spec.ref is not None:
            cmd += ["--branch", spec.ref]
        cmd += [spec.url, str(dest)]
        _run_git(cmd)
    return dest


# ── Assignment metrics ────────────────────────────────────────────────────


def _distribution(values: list[int]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def collect_tree_metrics(run_dir: Path) -> dict[str, Any] | None:
    """Assignment metrics from a finished run's artifacts.

    Best-effort: any failure returns an ``error`` marker rather than failing
    the batch row.
    """
    try:
        from spotlights_engine.modules_extractor.derive import (
            module_children,
            territory_source_file_counts,
        )
        from spotlights_engine.modules_extractor.stage_schemas import (
            ResolvedAssignmentTree,
            Skeleton,
        )

        skeleton_file = run_dir / "02_skeleton" / "skeleton.json"
        if not skeleton_file.is_file():
            skeleton_file = run_dir / "skeleton.json"
        if not skeleton_file.is_file():
            return None
        skeleton = Skeleton.model_validate(
            json.loads(skeleton_file.read_text(encoding="utf-8"))
        )

        resolved_file = run_dir / "resolved_assignments.json"
        if resolved_file.is_file():
            resolved = ResolvedAssignmentTree.model_validate(
                json.loads(resolved_file.read_text(encoding="utf-8"))
            )
            counts = territory_source_file_counts(resolved, skeleton)
            children = module_children(resolved)
            leaves = [m for m, cs in children.items() if not cs]
            lints_file = run_dir / "assignment_lints.json"
            lints = (
                json.loads(lints_file.read_text(encoding="utf-8"))
                if lints_file.is_file()
                else []
            )
            lint_counts: dict[str, int] = {}
            for lint in lints:
                code = lint.get("code", "unknown")
                lint_counts[code] = lint_counts.get(code, 0) + 1
            return {
                "merge_threshold": resolved.merge_threshold,
                "modules": len(resolved.module_paths()),
                "leaves": len(leaves),
                "leaf_territory": _distribution([counts[m] for m in leaves]),
                "lint_counts": lint_counts,
                "unowned_optional": 0,
            }

        return None
    except Exception as exc:  # noqa: BLE001 — metrics must never fail the row
        return {"error": f"{type(exc).__name__}: {exc}"}


# ── Per-repo runner ────────────────────────────────────────────────────────


def read_run_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def preserve_stale_run_dir(repo_dir: Path) -> None:
    """Rename a leftover run dir to `.failed.<ts>`, keeping only the last one.

    The library rejects a pre-existing `modules_extractor/` run dir (resume is
    not supported), so any leftover must be moved aside before a rerun.
    """
    run_dir = repo_dir / "modules_extractor"
    if not run_dir.exists():
        return
    for old in sorted(repo_dir.glob("modules_extractor.failed.*")):
        shutil.rmtree(old, ignore_errors=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.rename(repo_dir / f"modules_extractor.failed.{stamp}")


def run_repo(
    spec: RepoSpec,
    batch: BatchConfig,
    *,
    output_root: Path,
    claude_bin: str | None,
    force: bool,
    sequential: bool,
) -> dict[str, Any]:
    """Run one repo end to end; never raises except KeyboardInterrupt."""
    repo_dir = output_root / spec.name
    summary_path = repo_dir / "run_summary.json"

    prior = read_run_summary(summary_path)
    if prior is not None and prior.get("status") == "ok" and not force:
        return {
            "name": spec.name,
            "status": "skipped",
            "repo_path": prior.get("repo_path"),
            "cost_usd": None,
            "duration_s": None,
            "modules_total": prior.get("modules_total"),
            "modules_leaves": prior.get("modules_leaves"),
        }

    started = datetime.now(UTC)
    t0 = time.monotonic()
    summary: dict[str, Any] = {
        "name": spec.name,
        "repo_path": None,
        "status": "failed",
        "error": None,
        "duration_s": None,
        "cost_usd": None,
        "input_tokens": None,
        "output_tokens": None,
        "modules_total": None,
        "modules_leaves": None,
        "started_at": started.isoformat(),
        "finished_at": None,
        "config": None,
    }
    try:
        repo_dir.mkdir(parents=True, exist_ok=True)
        repo_path = resolve_repo_path(spec, output_root=output_root, force=force)
        summary["repo_path"] = str(repo_path)
        cfg = build_effective_config(spec, batch, artifacts_dir=repo_dir, claude_bin=claude_bin)
        summary["config"] = cfg.model_dump(mode="json")
        preserve_stale_run_dir(repo_dir)

        with (repo_dir / "log.txt").open("w", encoding="utf-8") as log:

            def on_event(line: str) -> None:
                log.write(line.rstrip("\n") + "\n")
                log.flush()
                if sequential:
                    print(f"[{spec.name}] {line}", flush=True)

            result = extract_with_telemetry(
                ModulesExtractorInput(repo_path=repo_path),
                config=cfg,
                on_event=on_event,
            )

        tree = result.project_tree
        inv = result.invocation
        summary["status"] = "ok"
        summary["cost_usd"] = inv.cost_usd
        summary["input_tokens"] = inv.input_tokens
        summary["output_tokens"] = inv.output_tokens
        summary["modules_total"] = len(list(tree.walk()))
        summary["modules_leaves"] = len(list(tree.leaves()))
        summary["tree_metrics"] = collect_tree_metrics(repo_dir / "modules_extractor")
    except KeyboardInterrupt:
        raise
    except (ModulesExtractorError, Exception) as exc:  # noqa: B014 — per design
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        # A failed run may still have best-effort artifacts worth comparing.
        summary["tree_metrics"] = collect_tree_metrics(repo_dir / "modules_extractor")
    summary["duration_s"] = round(time.monotonic() - t0, 1)
    summary["finished_at"] = datetime.now(UTC).isoformat()
    try:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[{spec.name}] failed to write run_summary.json: {exc}", file=sys.stderr)
    return summary


# ── Batch summaries ────────────────────────────────────────────────────────


def _totals(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    cost = 0.0
    for s in summaries:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
        if s["status"] == "ok" and s.get("cost_usd") is not None:
            cost += s["cost_usd"]
    return {**counts, "cost_usd": round(cost, 2)}


def write_batch_summary_json(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    started_at: str,
    finished_at: str | None,
    config_path: Path,
    output_root: Path,
) -> None:
    data = {
        "started_at": started_at,
        "finished_at": finished_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "totals": _totals(summaries),
        "repos": summaries,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def format_table(summaries: list[dict[str, Any]]) -> str:
    rows: list[tuple[str, str, str, str, str]] = []
    for s in summaries:
        status = s["status"]
        duration = f"{s['duration_s']:.1f}s" if s.get("duration_s") is not None else "—"
        if status == "ok":
            cost = f"${s['cost_usd']:.2f}" if s.get("cost_usd") is not None else "—"
            tail = f"{s['modules_total']} ({s['modules_leaves']})"
        elif status == "skipped":
            cost = "—"
            if s.get("modules_total") is not None:
                tail = f"{s['modules_total']} ({s['modules_leaves']})   # from previous run"
            else:
                tail = "—"
        else:  # failed
            cost = "—"
            err = s.get("error") or {}
            msg = (err.get("message") or "").replace("\n", " ")
            if len(msg) > 120:
                msg = msg[:117] + "..."
            tail = f"{err.get('type', 'Error')}: {msg}"
        rows.append((s["name"], status, duration, cost, tail))

    totals = _totals(summaries)
    total_tail = (
        f"{totals['ok']} ok / {totals['failed']} failed / "
        f"{totals['skipped']} skipped   ${totals['cost_usd']:.2f}"
    )

    name_w = max([len("repo"), len("total"), *(len(r[0]) for r in rows)])
    lines = [
        f"{'repo'.ljust(name_w)}  {'status'.ljust(7)}  {'duration'.rjust(9)}  "
        f"{'cost'.rjust(7)}  modules (leaves)"
    ]
    for name, status, duration, cost, tail in rows:
        lines.append(
            f"{name.ljust(name_w)}  {status.ljust(7)}  {duration.rjust(9)}  {cost.rjust(7)}  {tail}"
        )
    lines.append(f"{'total'.ljust(name_w)}  {total_tail}")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Repo-set YAML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"One sub-folder per repo is created here (default: {DEFAULT_OUTPUT_ROOT})",
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help="Run only this configured repo (repeatable).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun repos even if their status is ok; re-clone url repos.",
    )
    p.add_argument(
        "--max-parallel-repos",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Repos to run concurrently (default: 1, recommended). Each repo "
            "already runs up to max_parallel_enrich_shards (default 5) Claude "
            "subprocesses, so N multiplies subprocess count and API load."
        ),
    )
    p.add_argument(
        "--claude-bin",
        default=None,
        help="Claude Code binary, forwarded into every repo's ExtractorConfig.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan (repos + effective configs); run nothing.",
    )
    return p


def _select_repos(batch: BatchConfig, only: list[str] | None) -> list[RepoSpec]:
    if only is None:
        return list(batch.repos)
    by_name = {r.name: r for r in batch.repos}
    unknown = sorted(set(only) - set(by_name))
    if unknown:
        raise SystemExit(
            f"error: --only name(s) not in config: {', '.join(unknown)} "
            f"(configured: {', '.join(by_name)})"
        )
    # Keep config order; de-duplicate repeated --only flags.
    wanted = set(only)
    return [r for r in batch.repos if r.name in wanted]


def _print_dry_run(
    selected: list[RepoSpec],
    batch: BatchConfig,
    *,
    output_root: Path,
    claude_bin: str | None,
    force: bool,
) -> None:
    print(f"dry run: {len(selected)} repo(s), output root {output_root}")
    for spec in selected:
        repo_dir = output_root / spec.name
        print(f"\n{spec.name}:")
        if spec.path is not None:
            resolved = (
                spec.path if spec.path.is_absolute() else PROJECT_ROOT / spec.path
            ).resolve()
            state = "exists" if resolved.is_dir() else "MISSING"
            print(f"  source: path {spec.path} -> {resolved} ({state})")
        else:
            dest = clone_dir_for(spec, output_root)
            ref = f" @ {spec.ref}" if spec.ref else ""
            reuse = dest.exists() and not force
            action = "reuse existing clone" if reuse else "clone"
            print(f"  source: url {spec.url}{ref} -> {dest} ({action})")
        prior = read_run_summary(repo_dir / "run_summary.json")
        if prior is not None and prior.get("status") == "ok" and not force:
            print("  action: skip (previous run ok)")
        else:
            print(f"  action: run -> {repo_dir}")
        cfg = build_effective_config(spec, batch, artifacts_dir=repo_dir, claude_bin=claude_bin)
        dump = cfg.model_dump(mode="json")
        print("  effective config:")
        for key in sorted(dump):
            print(f"    {key}: {dump[key]}")


def main() -> int:
    args = _build_argparser().parse_args()
    config_path: Path = args.config.resolve()
    output_root: Path = args.output_root.resolve()

    try:
        batch = load_batch_config(config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: invalid config {config_path}: {exc}", file=sys.stderr)
        return 1

    selected = _select_repos(batch, args.only)

    if args.dry_run:
        _print_dry_run(
            selected,
            batch,
            output_root=output_root,
            claude_bin=args.claude_bin,
            force=args.force,
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    batch_started = datetime.now(UTC).isoformat()
    summary_json = output_root / "batch_summary.json"
    summary_md = output_root / "batch_summary.md"

    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def record(summary: dict[str, Any], *, finished: bool = False) -> None:
        with lock:
            results[summary["name"]] = summary
            ordered = [results[r.name] for r in selected if r.name in results]
            write_batch_summary_json(
                summary_json,
                ordered,
                started_at=batch_started,
                finished_at=datetime.now(UTC).isoformat() if finished else None,
                config_path=config_path,
                output_root=output_root,
            )

    sequential = args.max_parallel_repos <= 1
    interrupted = False
    try:
        if sequential:
            for spec in selected:
                print(f"=== {spec.name} ===", flush=True)
                record(
                    run_repo(
                        spec,
                        batch,
                        output_root=output_root,
                        claude_bin=args.claude_bin,
                        force=args.force,
                        sequential=True,
                    )
                )
        else:
            executor = ThreadPoolExecutor(max_workers=args.max_parallel_repos)
            try:
                pending = {
                    executor.submit(
                        run_repo,
                        spec,
                        batch,
                        output_root=output_root,
                        claude_bin=args.claude_bin,
                        force=args.force,
                        sequential=False,
                    ): spec
                    for spec in selected
                }
                while pending:
                    done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                    for fut in done:
                        spec = pending.pop(fut)
                        summary = fut.result()
                        print(f"[{spec.name}] {summary['status']}", flush=True)
                        record(summary)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted — finalizing batch summary", file=sys.stderr)

    ordered = [results[r.name] for r in selected if r.name in results]
    write_batch_summary_json(
        summary_json,
        ordered,
        started_at=batch_started,
        finished_at=datetime.now(UTC).isoformat(),
        config_path=config_path,
        output_root=output_root,
    )
    table = format_table(ordered)
    summary_md.write_text(f"# modules_extractor batch\n\n```\n{table}\n```\n", encoding="utf-8")
    print()
    print(table)
    print(f"\nbatch summary: {summary_json}")

    if interrupted:
        return 130
    return 0 if all(s["status"] in ("ok", "skipped") for s in ordered) else 1


if __name__ == "__main__":
    sys.exit(main())
