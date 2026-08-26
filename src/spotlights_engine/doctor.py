"""`spotlights-engine doctor` — live preflight for the agent CLIs.

Unlike a PATH-only check, this actually *runs* each agent CLI with a trivial
prompt (like `brew doctor` or `flutter doctor` probe their toolchains). One
call proves three things at once:

- **install** — the CLI resolves on PATH and executes;
- **auth** — it exits 0 and emits parseable output (an unauthenticated CLI
  errors out, so this catches "not logged in");
- **pricing readiness** — the model the CLI reports has a matching row in the
  contracted rate table, so a real run will not silently drop cost.

The model id is read from the CLI's own output using the exact same parsers
the costing code uses, and looked up with the exact same `provider:model`
key (context-window tag stripped, CLI-family fallback for a CLI that reports
no model). Exits 0 iff every check passes, else 1.

Running the CLIs spends a tiny amount on each probe (a one-word prompt). This
is deliberate: it is the only way to verify auth and the reported model.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.costing.rates import (
    _BUNDLED_RATES_PATH,
    RATES_ENV_VAR,
    ModelRate,
    _canonical_model_id,
    load_rates,
)
from spotlights_engine.costing.records import PROVIDER_FOR_CLI

# Minimal prompt: cheapest thing that still forces a real model turn and a
# usage payload we can read the model id from.
_PROBE_PROMPT = "Reply with the single word: ok"
# Probes should be fast; a cold CLI that hangs past this is a failure to report.
_PROBE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ProbeOutcome:
    """Result of actually running a CLI once.

    `ok` is True only when the process exited 0 with parseable output; `model`
    is the model id the CLI reported (None when it reports none — e.g. codex).
    """

    ok: bool
    model: str | None
    error: str


def _probe_model(name: str, *, cwd: Path | None = None) -> ProbeOutcome:
    """Run one agent CLI with a trivial prompt and read the reported model.

    Reuses the same exec clients and usage parsers the pipeline uses, so the
    model id and rate key match what a real run would produce. Any failure to
    launch, a non-zero exit, or unparseable output becomes `ok=False`.
    """
    from spotlights_engine.module_deep_research.agent_exec import ModuleResearchRunner

    run_cwd = cwd or Path.cwd()
    client: ModuleResearchRunner
    try:
        if name == "claude":
            from spotlights_engine.module_deep_research.claude_exec import (
                ClaudeExecClient,
                ClaudeExecOptions,
            )

            client = ClaudeExecClient(
                ClaudeExecOptions(
                    cwd=run_cwd,
                    max_turns=1,
                    allowed_tools=(),
                    timeout_seconds=_PROBE_TIMEOUT_SECONDS,
                )
            )
        elif name == "codex":
            from spotlights_engine.module_deep_research.codex_exec import (
                CodexExecClient,
                CodexExecOptions,
            )

            client = CodexExecClient(
                CodexExecOptions(
                    cwd=run_cwd,
                    search=False,
                    json_events=True,
                    timeout_seconds=_PROBE_TIMEOUT_SECONDS,
                )
            )
        else:  # pragma: no cover - defensive
            return ProbeOutcome(ok=False, model=None, error=f"unknown CLI {name!r}")

        result = client.run(_PROBE_PROMPT, check=False)
    except FileNotFoundError:
        return ProbeOutcome(ok=False, model=None, error="failed to launch")
    except Exception as exc:  # timeout, decode error, etc.
        return ProbeOutcome(ok=False, model=None, error=f"{type(exc).__name__}: {exc}")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        hint = stderr[-1] if stderr else "no error output"
        return ProbeOutcome(
            ok=False,
            model=None,
            error=f"exit {result.returncode} — likely not logged in ({hint})",
        )
    if result.usage is None:
        return ProbeOutcome(
            ok=False,
            model=None,
            error="ran but emitted no parseable usage — cannot verify model",
        )
    return ProbeOutcome(ok=True, model=result.usage.model, error="")


def _rate_key_for(name: str, model: str | None) -> str:
    """Mirror `costing.rates._rate_key`: model → `provider:model` (context tag
    and LiteLLM route prefix stripped), else the `provider:cli` family fallback.
    """
    provider = PROVIDER_FOR_CLI[name]
    if model:
        return f"{provider}:{_canonical_model_id(model)}"
    return f"{provider}:{name}"


def probe_cli(name: str, *, rates: dict[str, ModelRate], cwd: Path | None = None) -> CheckResult:
    """Live-check one agent CLI: install, auth, and priced model."""
    resolved = shutil.which(name)
    if resolved is None:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"`{name}` not found on PATH — install it (see README) and re-open your shell",
        )

    outcome = _probe_model(name, cwd=cwd)
    if not outcome.ok:
        return CheckResult(name=name, ok=False, detail=outcome.error)

    key = _rate_key_for(name, outcome.model)
    if key not in rates:
        reported = outcome.model or "(no model id reported)"
        return CheckResult(
            name=name,
            ok=False,
            detail=(
                f"reported model {reported!r} has no rate row ({key}). "
                f"Add a row keyed {key!r} to the rate table or set {RATES_ENV_VAR} "
                "to a table that prices it — an unpriced model silently drops from "
                'the cost total. See docs/cost-and-manifest.md → "Adding a model to the rate table".'
            ),
        )
    return CheckResult(
        name=name,
        ok=True,
        detail=f"{resolved} → priced as {key}",
    )


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


def run_checks() -> list[CheckResult]:
    rates_check = check_rates()
    # If the rate table itself is unreadable, load_rates would raise; probing
    # then can't validate pricing, so fall back to an empty table (every model
    # reads as unpriced, which the rates check already flagged).
    try:
        rates = load_rates()
    except (OSError, ValueError):
        rates = {}
    return [
        probe_cli("claude", rates=rates),
        probe_cli("codex", rates=rates),
        rates_check,
    ]


def _build_argparser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="spotlights-engine doctor",
        description=(
            "Live environment preflight: run each agent CLI with a trivial "
            "prompt to verify it is installed, authenticated, and reports a "
            "model the rate table prices. Spends a tiny amount per CLI probe. "
            "Run before your first spotlights-engine run."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    _build_argparser().parse_args(argv)
    results = run_checks()
    for r in results:
        mark = "ok  " if r.ok else "FAIL"
        print(f"[{mark}] {r.name}: {r.detail}")
    all_ok = all(r.ok for r in results)
    if not all_ok:
        print("\ndoctor: one or more checks failed — see messages above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
