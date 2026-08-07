"""`spotlights-engine doctor` — preflight environment checks.

Verifies the two agent CLIs are resolvable on PATH, the target repo exists,
and a cost rate table resolves, BEFORE a run spends money on parallel agent
fan-out. Exits 0 iff every check passes, else 1.

Note: this checks that `claude` / `codex` are installed and on PATH. It does
NOT verify they are authenticated — auth lives in each CLI's own login state
and cannot be probed without spending a request.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.costing.rates import (
    RATES_ENV_VAR,
    _BUNDLED_RATES_PATH,
)
from spotlights_engine.defaults import DEFAULT_REPO


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_cli(name: str) -> CheckResult:
    resolved = shutil.which(name)
    if resolved is None:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"`{name}` not found on PATH — install it (see README) and re-open your shell",
        )
    return CheckResult(name=name, ok=True, detail=resolved)


def check_repo(repo: Path) -> CheckResult:
    ok = repo.is_dir()
    detail = str(repo) if ok else f"target repo not found at {repo}"
    return CheckResult(name="repo", ok=ok, detail=detail)


def check_rates() -> CheckResult:
    env_path = os.environ.get(RATES_ENV_VAR)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return CheckResult(name="rates", ok=True, detail=f"{RATES_ENV_VAR}={p}")
        return CheckResult(
            name="rates",
            ok=False,
            detail=f"{RATES_ENV_VAR}={env_path} does not point to a readable file",
        )
    if _BUNDLED_RATES_PATH.is_file():
        return CheckResult(name="rates", ok=True, detail=f"bundled {_BUNDLED_RATES_PATH.name}")
    return CheckResult(name="rates", ok=False, detail="no rate table found")


def run_checks(repo: Path) -> list[CheckResult]:
    return [
        check_cli("claude"),
        check_cli("codex"),
        check_repo(repo),
        check_rates(),
    ]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine doctor",
        description=(
            "Preflight checks: agent CLIs on PATH, target repo present, rate "
            "table resolvable. Run this before your first spotlights-engine run."
        ),
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO,
        help=f"Path to the target repo to verify (default: {DEFAULT_REPO}).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    results = run_checks(args.repo)
    for r in results:
        mark = "ok  " if r.ok else "FAIL"
        print(f"[{mark}] {r.name}: {r.detail}")
    all_ok = all(r.ok for r in results)
    if not all_ok:
        print("\ndoctor: one or more checks failed — see messages above.")
    return 0 if all_ok else 1
