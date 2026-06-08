#!/usr/bin/env python3
"""Run each pytest test ID in a fresh pytest process and merge JSON reports.

Workaround for vLLM EngineCore tests under GPU Exclusive Process compute mode:
running multiple parametrized cases in one pytest session fails with
``cudaErrorDevicesUnavailable`` because the prior case's EngineCore subprocess
hasn't released its CUDA context before the next case's MemorySnapshot init
runs. Exec'ing a fresh pytest per case guarantees the prior process tree (and
its CUDA context) is gone before the next starts.

Usage (as drop-in for the harness ``invoke``)::

    serial_pytest.py <test_path> [-k EXPR] [extra pytest args...]

The harness appends ``--json-report --json-report-file=<path>``; this wrapper
intercepts that path, runs each collected node-id in its own pytest with a
private per-case report file, then merges all per-case reports into the
harness path. Exit code is non-zero if any case failed.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def _split_args(argv: list[str]) -> tuple[list[str], str | None, list[str]]:
    """Return (positional_and_other_args, json_report_file, k_expr).

    Strips ``--json-report`` / ``--json-report-file=`` so we can manage them
    per-case, and extracts ``-k EXPR`` to pass to ``--collect-only``.
    """
    out: list[str] = []
    json_path: str | None = None
    k_expr: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--json-report":
            i += 1
            continue
        if a.startswith("--json-report-file="):
            json_path = a.split("=", 1)[1]
            i += 1
            continue
        if a == "--json-report-file":
            json_path = argv[i + 1]
            i += 2
            continue
        if a == "-k":
            k_expr = argv[i + 1]
            i += 2
            continue
        if a.startswith("-k="):
            k_expr = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return out, json_path, k_expr


def _collect(pytest_argv: list[str], k_expr: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", *pytest_argv]
    if k_expr:
        cmd += ["-k", k_expr]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode or 2)
    nodeids: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("="):
            nodeids.append(line)
    return nodeids


def _run_one(nodeid: str, extra: list[str], report_file: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        nodeid,
        "--json-report",
        f"--json-report-file={report_file}",
        *extra,
    ]
    print(f"\n>>> serial_pytest: {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def _merge(reports: list[Path], target: Path) -> tuple[int, int, int]:
    merged: dict = {
        "summary": {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "total": 0},
        "tests": [],
        "collectors": [],
        "exitcode": 0,
    }
    for r in reports:
        if not r.exists():
            continue
        try:
            data = json.loads(r.read_text())
        except json.JSONDecodeError:
            continue
        s = data.get("summary", {})
        for k in ("passed", "failed", "skipped", "error", "total"):
            merged["summary"][k] = merged["summary"].get(k, 0) + s.get(k, 0)
        merged["tests"].extend(data.get("tests", []))
        merged["collectors"].extend(data.get("collectors", []))
        ec = data.get("exitcode", 0)
        if ec and not merged["exitcode"]:
            merged["exitcode"] = ec
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged))
    s = merged["summary"]
    return s.get("passed", 0), s.get("failed", 0), s.get("skipped", 0)


def main(argv: list[str]) -> int:
    pytest_argv, json_path, k_expr = _split_args(argv)
    if not pytest_argv:
        print("serial_pytest: missing test path", file=sys.stderr)
        return 2

    nodeids = _collect(pytest_argv, k_expr)
    if not nodeids:
        print("serial_pytest: no tests collected", file=sys.stderr)
        # Still emit an empty report so the harness parser is happy.
        if json_path:
            Path(json_path).write_text(
                json.dumps({"summary": {"passed": 0, "failed": 0, "skipped": 0}, "tests": []})
            )
        return 5  # pytest's "no tests collected" exit code

    print(f"serial_pytest: running {len(nodeids)} cases one at a time:", flush=True)
    for n in nodeids:
        print(f"  - {n}", flush=True)

    # Drop the original test path positional from per-case extras; we pass the
    # nodeid instead. Anything else (markers, -p, -x, etc.) is forwarded.
    test_path = pytest_argv[0]
    extra = [a for a in pytest_argv[1:] if a != test_path]

    reports: list[Path] = []
    overall_rc = 0
    with tempfile.TemporaryDirectory() as td:
        for i, nodeid in enumerate(nodeids):
            r = Path(td) / f"case_{i:03d}.json"
            rc = _run_one(nodeid, extra, r)
            reports.append(r)
            if rc != 0 and overall_rc == 0:
                overall_rc = rc

        if json_path:
            passed, failed, skipped = _merge(reports, Path(json_path))
            print(
                f"\nserial_pytest: merged report — passed={passed} failed={failed} skipped={skipped}",
                flush=True,
            )

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
