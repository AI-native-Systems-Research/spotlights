"""`spotlights-engine rates refresh` — build a rate table from a LiteLLM gateway.

A LiteLLM proxy publishes the per-token prices it bills at on `/model/info`.
This command fetches them, keys each row by the `provider:model` id the CLIs
report (the gateway's `model_name` alias), and merges them over a base table
(the bundled `rates.json` by default). The result is a file to point
`SPOTLIGHTS_RATES_FILE` at.

Rows are merged by *canonical* key — the same key `load_rates` builds — so a
gateway row `openai:azure/gpt-5.5` replaces the bundled `openai:gpt-5.5`
instead of colliding with it at load time. A gateway model with no published
input/output price is skipped and reported, never filled from list price.

The `openai:codex` CLI-family fallback row is re-pointed at the gateway rate
of the model Codex actually runs, when the gateway prices that model.

Endpoint discovery (first hit wins): `--endpoint`, `SPOTLIGHTS_LITELLM_URL`,
`LITELLM_BASE_URL`, `ANTHROPIC_BASE_URL` (env, then `~/.claude/settings.json`),
`OPENAI_BASE_URL`, then the active provider's `base_url` in
`~/.codex/config.toml`. API key: `--api-key-env VAR`, `LITELLM_API_KEY`,
`ANTHROPIC_AUTH_TOKEN` (env, then Claude settings), the Codex provider's
`env_key`. A missing key is fine for gateways that need none.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spotlights_engine.costing.rates import (
    _BUNDLED_RATES_PATH,
    RATES_ENV_VAR,
    ModelRate,
    _canonical_rate_key,
    canonical_model_id,
    load_rates,
)

DEFAULT_OUT = Path("~/.config/spotlights/rates.json")
_FETCH_TIMEOUT_SECONDS = 30
# Tried in order against the base URL. LiteLLM serves both; some front-door
# routers only forward the `/v1` form.
_MODEL_INFO_PATHS = ("/model/info", "/v1/model/info")
_CODEX_FALLBACK_KEY = "openai:codex"

#: Fetches `url` with `headers`, returns the parsed JSON body. Injected in tests.
Fetcher = Callable[[str, Mapping[str, str]], Any]


class RefreshError(RuntimeError):
    """The refresh could not produce a table (no endpoint, fetch failed, ...)."""


@dataclass(frozen=True)
class Endpoint:
    url: str
    source: str  # human-facing, e.g. "ANTHROPIC_BASE_URL (~/.claude/settings.json)"


@dataclass
class RefreshReport:
    endpoint: str = ""
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    codex_fallback: str = ""


# ---------------------------------------------------------------- discovery


def _claude_settings() -> dict[str, Any]:
    root = os.environ.get("CLAUDE_CONFIG_DIR")
    path = (Path(root) if root is not None else Path.home() / ".claude") / "settings.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    env = payload.get("env") if isinstance(payload, dict) else None
    return env if isinstance(env, dict) else {}


def _codex_config() -> dict[str, Any]:
    home = os.environ.get("CODEX_HOME")
    path = (Path(home) if home else Path.home() / ".codex") / "config.toml"
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _codex_provider(cfg: Mapping[str, Any]) -> dict[str, Any]:
    name = cfg.get("model_provider")
    providers = cfg.get("model_providers")
    if not isinstance(name, str) or not isinstance(providers, dict):
        return {}
    provider = providers.get(name)
    return provider if isinstance(provider, dict) else {}


def discover_endpoint(explicit: str | None = None) -> Endpoint | None:
    if explicit:
        return Endpoint(explicit, "--endpoint")
    for var in ("SPOTLIGHTS_LITELLM_URL", "LITELLM_BASE_URL", "ANTHROPIC_BASE_URL"):
        if os.environ.get(var):
            return Endpoint(os.environ[var], var)
    claude_url = _claude_settings().get("ANTHROPIC_BASE_URL")
    if isinstance(claude_url, str) and claude_url:
        return Endpoint(claude_url, "ANTHROPIC_BASE_URL (Claude settings.json)")
    if os.environ.get("OPENAI_BASE_URL"):
        return Endpoint(os.environ["OPENAI_BASE_URL"], "OPENAI_BASE_URL")
    codex_url = _codex_provider(_codex_config()).get("base_url")
    if isinstance(codex_url, str) and codex_url:
        return Endpoint(codex_url, "base_url (Codex config.toml)")
    return None


def discover_api_key(env_var: str | None = None) -> str | None:
    if env_var:
        return os.environ.get(env_var) or None
    for var in ("LITELLM_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    token = _claude_settings().get("ANTHROPIC_AUTH_TOKEN")
    if isinstance(token, str) and token:
        return token
    codex_key_var = _codex_provider(_codex_config()).get("env_key")
    if isinstance(codex_key_var, str) and os.environ.get(codex_key_var):
        return os.environ[codex_key_var]
    return None


def discover_codex_model(explicit: str | None = None) -> str | None:
    """The model behind Codex invocations that report no model id.

    `models.yaml` is what the engine passes as `--model`; without it Codex
    falls back to its own `config.toml`, and step 2 to its built-in default.
    """
    if explicit:
        return explicit
    from spotlights_engine.model_config import load_model_config

    try:
        configured = load_model_config().codex
    except (OSError, ValueError):
        configured = None
    if configured:
        return configured
    toml_model = _codex_config().get("model")
    if isinstance(toml_model, str) and toml_model:
        return toml_model
    from spotlights_engine.candidate_discovery.api import DiscoveryConfig

    return DiscoveryConfig().codex_model


# ---------------------------------------------------------------- fetch


def _http_fetch(url: str, headers: Mapping[str, str]) -> Any:
    req = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _base_url(url: str) -> str:
    # Codex/OpenAI base URLs usually end in `/v1`; `/model/info` sits at the root.
    return re.sub(r"/v1/?$", "", url.rstrip("/"))


def fetch_model_info(
    endpoint: str, headers: Mapping[str, str], fetch: Fetcher | None = None
) -> list[dict[str, Any]]:
    fetch = fetch or _http_fetch
    base = _base_url(endpoint)
    errors: list[str] = []
    for path in _MODEL_INFO_PATHS:
        url = base + path
        try:
            payload = fetch(url, headers)
        except urllib.error.HTTPError as exc:
            errors.append(f"{url}: HTTP {exc.code}")
            continue
        except (urllib.error.URLError, OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        errors.append(f"{url}: response has no 'data' list")
    raise RefreshError("could not read model info — " + "; ".join(errors))


# ---------------------------------------------------------------- build


def provider_for_model(model: str) -> str | None:
    """Guess the CLI provider from the model id; None when it is neither."""
    name = canonical_model_id(model).lower()
    if "claude" in name:
        return "anthropic"
    if re.search(r"(^|/)(gpt|o\d|codex)", name):
        return "openai"
    return None


def _cost(info: Mapping[str, Any], field_name: str) -> float | None:
    value = info.get(field_name)
    return float(value) if isinstance(value, (int, float)) else None


def rows_from_model_info(
    items: Iterable[tuple[str | None, dict[str, Any]]],
    *,
    today: str,
    report: RefreshReport,
) -> dict[str, ModelRate]:
    """Gateway rows keyed by canonical `provider:model`.

    Each item is `(provider, model_info row)`. A provider pins the row to it
    (per-CLI fetch mode); None guesses it from the model name. Duplicate
    aliases with identical prices (LiteLLM lists one row per deployment)
    collapse; differing prices for the same canonical key are ambiguous, so
    that key is dropped and reported.
    """
    rows: dict[str, ModelRate] = {}
    dropped: set[str] = set()
    for provider, item in items:
        name = item.get("model_name")
        info = item.get("model_info")
        if not isinstance(name, str) or not isinstance(info, dict):
            continue
        prov = provider or provider_for_model(name)
        if prov is None:
            report.unmapped.append(name)
            continue
        key = _canonical_rate_key(f"{prov}:{name}")
        inp, out = _cost(info, "input_cost_per_token"), _cost(info, "output_cost_per_token")
        if inp is None or out is None:
            report.unpriced.append(key)
            continue
        cache_read = _cost(info, "cache_read_input_token_cost")
        cache_create = _cost(info, "cache_creation_input_token_cost")
        missing = [
            label
            for label, value in (("cache_read", cache_read), ("cache_create", cache_create))
            if value is None
        ]
        note = f"From LiteLLM /model/info ({name}) on {today}"
        if missing:
            note += f"; gateway publishes no {'/'.join(missing)} price, set to 0"
        rate = ModelRate(
            input=inp,
            output=out,
            cache_read=cache_read or 0.0,
            cache_create=cache_create or 0.0,
            note=note,
        )
        if key in dropped:
            continue
        prior = rows.get(key)
        if prior is not None and prior.model_dump(exclude={"note"}) != rate.model_dump(
            exclude={"note"}
        ):
            del rows[key]
            dropped.add(key)
            report.conflicts.append(key)
            continue
        rows.setdefault(key, rate)
    return rows


def merge_rates(
    base: dict[str, ModelRate],
    gateway: dict[str, ModelRate],
    *,
    codex_model: str | None,
    report: RefreshReport,
) -> dict[str, ModelRate]:
    merged = dict(base)
    for key, rate in sorted(gateway.items()):
        (report.updated if key in merged else report.added).append(key)
        merged[key] = rate
    if codex_model:
        codex_key = _canonical_rate_key(f"openai:{codex_model}")
        codex_rate = gateway.get(codex_key)
        if codex_rate is not None:
            merged[_CODEX_FALLBACK_KEY] = codex_rate.model_copy(
                update={
                    "note": f"CLI-family fallback = {codex_key} gateway rate ({codex_rate.note})"
                }
            )
            report.codex_fallback = f"{_CODEX_FALLBACK_KEY} → {codex_key}"
        else:
            report.codex_fallback = (
                f"{_CODEX_FALLBACK_KEY} kept from base: gateway does not price {codex_key}"
            )
    return merged


def refresh_rates(
    *,
    endpoint: str,
    headers: Mapping[str, str],
    per_cli_headers: Mapping[str, Mapping[str, str]] | None = None,
    base_path: Path = _BUNDLED_RATES_PATH,
    codex_model: str | None = None,
    today: str | None = None,
    fetch: Fetcher | None = None,
) -> tuple[dict[str, Any], RefreshReport]:
    """Fetch, build, and merge. Returns the JSON payload to write + a report.

    `per_cli_headers` maps provider (`anthropic`/`openai`) to extra headers;
    when given, one fetch runs per provider and its rows are pinned to it —
    for routers that scope the model list by a per-harness header.
    """
    today = today or _dt.datetime.now(_dt.UTC).date().isoformat()
    report = RefreshReport(endpoint=endpoint)
    fetches: list[tuple[str | None, Mapping[str, str]]] = (
        [(prov, {**headers, **extra}) for prov, extra in per_cli_headers.items()]
        if per_cli_headers
        else [(None, headers)]
    )
    items = [
        (provider, row)
        for provider, hdrs in fetches
        for row in fetch_model_info(endpoint, hdrs, fetch)
    ]
    gateway = rows_from_model_info(items, today=today, report=report)
    if not gateway:
        raise RefreshError(
            f"gateway at {endpoint} returned no priced model "
            f"({len(report.unpriced)} without price, {len(report.unmapped)} not Claude/OpenAI)"
        )
    base = load_rates(base_path)
    merged = merge_rates(base, gateway, codex_model=codex_model, report=report)
    payload: dict[str, Any] = {
        "_note": (
            f"Base table {base_path.name} plus LiteLLM gateway prices from {endpoint}, "
            f"fetched {today} by `spotlights-engine rates refresh`. Keys are canonical "
            "provider:model (route prefix and context tag stripped)."
        )
    }
    payload.update({key: rate.model_dump() for key, rate in sorted(merged.items())})
    return payload, report


# ---------------------------------------------------------------- CLI


def _parse_header(raw: str) -> tuple[str, str]:
    name, sep, value = raw.partition(":")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"header must be 'Name: value', got {raw!r}")
    return name.strip(), value.strip()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine rates",
        description="Manage the contracted rate table the engine prices runs with.",
    )
    sub = p.add_subparsers(dest="rates_command", required=True)
    r = sub.add_parser(
        "refresh",
        help="Build a rate table from a LiteLLM gateway's /model/info prices.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    r.add_argument("--endpoint", help="LiteLLM base URL (default: discovered, see above).")
    r.add_argument(
        "--api-key-env",
        metavar="VAR",
        help="Env var holding the gateway key (default: discovered; none is fine).",
    )
    r.add_argument(
        "--header",
        action="append",
        type=_parse_header,
        default=[],
        metavar="'NAME: VALUE'",
        help="Extra request header for every fetch. Repeatable.",
    )
    r.add_argument(
        "--claude-header",
        type=_parse_header,
        metavar="'NAME: VALUE'",
        help="Per-CLI mode: header for the Claude fetch; its rows are keyed anthropic:*.",
    )
    r.add_argument(
        "--codex-header",
        type=_parse_header,
        metavar="'NAME: VALUE'",
        help="Per-CLI mode: header for the Codex fetch; its rows are keyed openai:*.",
    )
    r.add_argument(
        "--codex-model",
        help="Model behind the openai:codex fallback row (default: discovered).",
    )
    r.add_argument(
        "--base",
        type=Path,
        default=_BUNDLED_RATES_PATH,
        help="Table whose rows are kept unless the gateway prices them (default: bundled).",
    )
    r.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where to write the table (default: {DEFAULT_OUT}).",
    )
    r.add_argument("--dry-run", action="store_true", help="Print the table; write nothing.")
    return p


def _print_report(report: RefreshReport, out: Path | None) -> None:
    def line(label: str, items: list[str]) -> None:
        if items:
            print(f"  {label} ({len(items)}): {', '.join(sorted(set(items)))}", file=sys.stderr)

    print(f"rates refresh: endpoint {report.endpoint}", file=sys.stderr)
    line("added", report.added)
    line("updated", report.updated)
    line("skipped, no gateway price", report.unpriced)
    line("skipped, not a Claude/OpenAI model", report.unmapped)
    line("skipped, conflicting prices for one key", report.conflicts)
    if report.codex_fallback:
        print(f"  {report.codex_fallback}", file=sys.stderr)
    if out is not None:
        print(f"wrote {out}", file=sys.stderr)
        if os.environ.get(RATES_ENV_VAR) != str(out):
            print(f'next: export {RATES_ENV_VAR}="{out}"', file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    endpoint = discover_endpoint(args.endpoint)
    if endpoint is None:
        print(
            "rates refresh: no LiteLLM endpoint found — pass --endpoint or set "
            "SPOTLIGHTS_LITELLM_URL / ANTHROPIC_BASE_URL",
            file=sys.stderr,
        )
        return 1
    headers = dict(args.header)
    key = discover_api_key(args.api_key_env)
    if key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {key}"
    per_cli = {
        prov: dict([hdr])
        for prov, hdr in (("anthropic", args.claude_header), ("openai", args.codex_header))
        if hdr is not None
    }
    try:
        payload, report = refresh_rates(
            endpoint=endpoint.url,
            headers=headers,
            per_cli_headers=per_cli or None,
            base_path=args.base,
            codex_model=discover_codex_model(args.codex_model),
        )
    except (RefreshError, OSError, ValueError) as exc:
        print(f"rates refresh: {exc}", file=sys.stderr)
        return 1
    report.endpoint = f"{endpoint.url} (from {endpoint.source})"
    text = json.dumps(payload, indent=2) + "\n"
    if args.dry_run:
        print(text, end="")
        _print_report(report, None)
        return 0
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        load_rates(tmp)  # the engine must be able to read what we hand it
    except (OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"rates refresh: built table does not load: {exc}", file=sys.stderr)
        return 1
    tmp.replace(out)
    _print_report(report, out)
    return 0


__all__ = [
    "Endpoint",
    "RefreshError",
    "RefreshReport",
    "discover_api_key",
    "discover_codex_model",
    "discover_endpoint",
    "fetch_model_info",
    "main",
    "merge_rates",
    "provider_for_model",
    "refresh_rates",
    "rows_from_model_info",
]
