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
from spotlights_engine.model_config import (
    MODELS_ENV_VAR,
    ModelConfig,
    load_model_config,
    models_path,
)

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


def _probe_model(
    name: str, *, cwd: Path | None = None, model: str | None = None
) -> ProbeOutcome:
    """Run one agent CLI with a trivial prompt and read the reported model.

    Reuses the same exec clients and usage parsers the pipeline uses, so the
    model id and rate key match what a real run would produce. `model` is the
    configured global id, passed so the probe exercises the model a real run
    would use rather than the CLI's own default — without it, doctor can report
    an unpriced model that no run would ever ask for. Any failure to launch, a
    non-zero exit, or unparseable output becomes `ok=False`.
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
                    model=model,
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
                    model=model,
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


def probe_cli(
    name: str,
    *,
    rates: dict[str, ModelRate],
    cwd: Path | None = None,
    model: str | None = None,
) -> CheckResult:
    """Live-check one agent CLI: install, auth, and priced model.

    `model` is the configured global id for this CLI, so the probe checks the
    model a real run would use.
    """
    resolved = shutil.which(name)
    if resolved is None:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"`{name}` not found on PATH — install it (see README) and re-open your shell",
        )

    outcome = _probe_model(name, cwd=cwd, model=model)
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


def _effective_codex_model(configured: str | None) -> str | None:
    """The Codex model a run will really use when the config says nothing.

    Candidate discovery (step 2) keeps a built-in `gpt-5.5` default — it is part
    of the resume fingerprint, so it cannot be dropped. Reporting or probing
    "inherit" while step 2 pins a model would make a green doctor followed by a
    403 on every module. Steps 3 and 5 do inherit in that case, which is a
    pre-existing asymmetry this only reports, never hides.
    """
    if configured:
        return configured
    from spotlights_engine.candidate_discovery.api import DiscoveryConfig

    return DiscoveryConfig().codex_model


def check_models() -> CheckResult:
    """Report the configured model per CLI and where the value came from.

    This is a config check, not a probe: it says what the engine *will ask for*.
    `probe_cli` above reports what a CLI actually answers with. A blank value
    here means the engine passes no `--model` and the CLI picks for itself.
    """
    resolved = models_path()
    from_env = bool(os.environ.get(MODELS_ENV_VAR))
    source = (
        f"{MODELS_ENV_VAR}={resolved}" if from_env else f"bundled {resolved.name}"
    )
    if not resolved.is_file():
        # An absent *bundled* file is fine — the file is optional and both CLIs
        # fall back to their own default. An absent file that someone explicitly
        # pointed the env var at is a typo, and silently using the CLI default
        # while the operator believes their pin is active is the worst outcome.
        if from_env:
            return CheckResult(
                name="models",
                ok=False,
                detail=f"{source} does not point to a readable file",
            )
        return CheckResult(
            name="models",
            ok=True,
            detail=f"no {resolved.name} — both CLIs use their own default",
        )
    try:
        cfg = load_model_config()
    except (OSError, ValueError) as exc:
        return CheckResult(name="models", ok=False, detail=f"{source}: {exc}")
    codex_effective = _effective_codex_model(cfg.codex)
    codex_label = codex_effective or "CLI default"
    if not cfg.codex and codex_effective:
        codex_label = f"{codex_effective} (step 2 default; steps 3+5 inherit)"
    parts = [
        f"claude={cfg.claude or 'CLI default'}",
        f"codex={codex_label}",
    ]
    return CheckResult(
        name="models", ok=True, detail=f"{source} → {', '.join(parts)}"
    )


def run_checks() -> list[CheckResult]:
    rates_check = check_rates()
    # If the rate table itself is unreadable, load_rates would raise; probing
    # then can't validate pricing, so fall back to an empty table (every model
    # reads as unpriced, which the rates check already flagged).
    try:
        rates = load_rates()
    except (OSError, ValueError):
        rates = {}
    # Probe the models a run would actually ask for, not each CLI's own default.
    try:
        models = load_model_config()
    except (OSError, ValueError):
        models = ModelConfig()
    return [
        probe_cli("claude", rates=rates, model=models.claude),
        *_codex_probes(rates=rates, configured=models.codex),
        rates_check,
        check_models(),
    ]


def _codex_probes(
    *, rates: dict[str, ModelRate], configured: str | None
) -> list[CheckResult]:
    """One codex probe, or two when the pipeline is not using a single model.

    With nothing configured, step 2 pins its built-in default while steps 3 and 5
    inherit `~/.codex/config.toml` — two different models. Probing only one of
    them means a green doctor can still be followed by unpriced usage or a 403 in
    whichever step went unchecked, which is exactly what this check exists to
    prevent. So probe both and label them.
    """
    if configured:
        return [probe_cli("codex", rates=rates, model=configured)]

    step2 = _effective_codex_model(None)
    inherited = probe_cli("codex", rates=rates, model=None)
    if not step2:
        return [inherited]
    pinned = probe_cli("codex", rates=rates, model=step2)
    return [
        CheckResult(
            name="codex (steps 3+5, inherited)",
            ok=inherited.ok,
            detail=inherited.detail,
        ),
        CheckResult(
            name=f"codex (step 2, {step2})", ok=pinned.ok, detail=pinned.detail
        ),
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
