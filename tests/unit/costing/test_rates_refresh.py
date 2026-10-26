from __future__ import annotations

import json
import urllib.error
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from spotlights_engine.costing import refresh
from spotlights_engine.costing.rates import load_rates


def _row(name: str, inp: float | None = 1e-6, out: float | None = 2e-6, **extra: Any):
    info: dict[str, Any] = {"input_cost_per_token": inp, "output_cost_per_token": out}
    info.update(extra)
    return {"model_name": name, "model_info": info}


GATEWAY = {
    "data": [
        _row(
            "aws/claude-opus-5",
            3e-6,
            15e-6,
            cache_read_input_token_cost=3e-7,
            cache_creation_input_token_cost=3.75e-6,
        ),
        # Same canonical key, same price — a second deployment of one alias.
        _row(
            "aws/claude-opus-5",
            3e-6,
            15e-6,
            cache_read_input_token_cost=3e-7,
            cache_creation_input_token_cost=3.75e-6,
        ),
        _row("azure/gpt-5.5", 2e-6, 12e-6, cache_read_input_token_cost=2e-7),
        _row("text-embedding-3-large", 1e-7, 0.0),
        _row("claude-haiku-5-5", None, None),
    ]
}


def _fetcher(payloads: Mapping[str, Any], calls: list[tuple[str, dict[str, str]]] | None = None):
    def fetch(url: str, headers: Mapping[str, str]) -> Any:
        if calls is not None:
            calls.append((url, dict(headers)))
        if url not in payloads:
            raise urllib.error.HTTPError(url, 404, "not found", None, None)  # type: ignore[arg-type]
        payload = payloads[url]
        return payload(headers) if callable(payload) else payload

    return fetch


@pytest.fixture(autouse=True)
def _no_user_config(monkeypatch, tmp_path):
    # Discovery must not read the developer's real ~/.claude or ~/.codex.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    for var in (
        "SPOTLIGHTS_LITELLM_URL",
        "LITELLM_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "LITELLM_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "SPOTLIGHTS_RATES_FILE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_refresh_merges_by_canonical_key_and_replaces_bundled_row():
    payload, report = refresh.refresh_rates(
        endpoint="http://gw:4000",
        headers={},
        codex_model="gpt-5.5",
        today="2026-10-25",
        fetch=_fetcher({"http://gw:4000/model/info": GATEWAY}),
    )
    # `azure/gpt-5.5` lands on the bundled `openai:gpt-5.5` key instead of
    # sitting next to it (which would make load_rates raise on collision).
    assert "openai:azure/gpt-5.5" not in payload
    assert payload["openai:gpt-5.5"]["input"] == 2e-6
    assert payload["anthropic:claude-opus-5"]["cache_create"] == 3.75e-6
    # Bundled rows the gateway does not price are kept.
    assert payload["anthropic:claude-opus-4-8"]["input"] == 0.0000038
    assert sorted(report.updated) == ["anthropic:claude-opus-5", "openai:gpt-5.5"]
    assert report.unpriced == ["anthropic:claude-haiku-5-5"]
    assert report.unmapped == ["text-embedding-3-large"]
    assert report.conflicts == []


def test_codex_fallback_follows_codex_model():
    payload, report = refresh.refresh_rates(
        endpoint="http://gw:4000",
        headers={},
        codex_model="azure/gpt-5.5",
        today="2026-10-25",
        fetch=_fetcher({"http://gw:4000/model/info": GATEWAY}),
    )
    assert payload["openai:codex"]["input"] == 2e-6
    assert payload["openai:codex"]["output"] == 12e-6
    assert "openai:codex → openai:gpt-5.5" in report.codex_fallback


def test_codex_fallback_kept_when_gateway_lacks_codex_model():
    payload, report = refresh.refresh_rates(
        endpoint="http://gw:4000",
        headers={},
        codex_model="gpt-9",
        today="2026-10-25",
        fetch=_fetcher({"http://gw:4000/model/info": GATEWAY}),
    )
    assert payload["openai:codex"]["input"] == 0.0000025  # bundled value
    assert "kept from base" in report.codex_fallback


def test_missing_cache_prices_are_zero_and_noted():
    payload, _ = refresh.refresh_rates(
        endpoint="http://gw:4000",
        headers={},
        today="2026-10-25",
        fetch=_fetcher({"http://gw:4000/model/info": GATEWAY}),
    )
    row = payload["openai:gpt-5.5"]
    assert row["cache_create"] == 0.0
    assert "no cache_create price" in row["note"]


def test_conflicting_prices_for_one_key_are_dropped():
    data = {"data": [_row("aws/claude-opus-5", 3e-6), _row("bedrock/claude-opus-5", 4e-6)]}
    payload, report = refresh.refresh_rates(
        endpoint="http://gw:4000",
        headers={},
        today="2026-10-25",
        fetch=_fetcher({"http://gw:4000/model/info": {"data": [*data["data"], _row("gpt-5.5")]}}),
    )
    assert report.conflicts == ["anthropic:claude-opus-5"]
    assert payload["anthropic:claude-opus-5"]["input"] == 0.0000038  # bundled kept


def test_falls_back_to_v1_path_and_strips_v1_from_base():
    calls: list[tuple[str, dict[str, str]]] = []
    refresh.refresh_rates(
        endpoint="http://gw:4000/v1/",
        headers={"Authorization": "Bearer k"},
        today="2026-10-25",
        fetch=_fetcher({"http://gw:4000/v1/model/info": GATEWAY}, calls),
    )
    assert [url for url, _ in calls] == [
        "http://gw:4000/model/info",
        "http://gw:4000/v1/model/info",
    ]
    assert calls[0][1]["Authorization"] == "Bearer k"


def test_per_cli_headers_pin_provider():
    # A router that scopes the model list by a harness header and uses model
    # names that say nothing about the provider (the DAM sandbox shape).
    def by_harness(headers: Mapping[str, str]) -> Any:
        if headers.get("X-Harness") == "claude-code":
            return {"data": [_row("ibm/claude-opus-5", 3e-6)]}
        return {"data": [_row("ibm/azure/gpt-5.5", 2e-6)]}

    payload, _ = refresh.refresh_rates(
        endpoint="http://router",
        headers={},
        per_cli_headers={
            "anthropic": {"X-Harness": "claude-code"},
            "openai": {"X-Harness": "codex"},
        },
        codex_model="ibm/azure/gpt-5.5",
        today="2026-10-25",
        fetch=_fetcher({"http://router/model/info": by_harness}),
    )
    assert payload["anthropic:ibm/claude-opus-5"]["input"] == 3e-6
    assert payload["openai:ibm/azure/gpt-5.5"]["input"] == 2e-6
    assert payload["openai:codex"]["input"] == 2e-6


def test_no_priced_model_is_an_error():
    with pytest.raises(refresh.RefreshError, match="no priced model"):
        refresh.refresh_rates(
            endpoint="http://gw",
            headers={},
            fetch=_fetcher({"http://gw/model/info": {"data": [_row("gpt-5.5", None)]}}),
        )


def test_unreachable_gateway_is_an_error():
    with pytest.raises(refresh.RefreshError, match="HTTP 404"):
        refresh.refresh_rates(endpoint="http://gw", headers={}, fetch=_fetcher({}))


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("aws/claude-opus-5[1m]", "anthropic"),
        ("claude-haiku-5-5", "anthropic"),
        ("azure/gpt-5.5", "openai"),
        ("o4-mini", "openai"),
        ("gpt-5-codex", "openai"),
        ("text-embedding-3-large", None),
        ("llama-3-70b", None),
    ],
)
def test_provider_for_model(model, provider):
    assert refresh.provider_for_model(model) == provider


def test_discover_endpoint_and_key_from_claude_settings(monkeypatch, tmp_path):
    settings = tmp_path / "claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://gw", "ANTHROPIC_AUTH_TOKEN": "tok"}})
    )
    ep = refresh.discover_endpoint()
    assert ep is not None and ep.url == "http://gw" and "settings.json" in ep.source
    assert refresh.discover_api_key() == "tok"


def test_discover_endpoint_and_key_from_codex_config(monkeypatch, tmp_path):
    cfg = tmp_path / "codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text(
        'model = "gpt-5.5"\nmodel_provider = "gw"\n'
        '[model_providers.gw]\nbase_url = "http://gw/v1"\nenv_key = "GW_KEY"\n'
    )
    monkeypatch.setenv("GW_KEY", "codex-key")
    ep = refresh.discover_endpoint()
    assert ep is not None and ep.url == "http://gw/v1"
    assert refresh.discover_api_key() == "codex-key"


def test_explicit_endpoint_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://from-env")
    ep = refresh.discover_endpoint("http://explicit")
    assert ep is not None and ep.url == "http://explicit"


def test_main_writes_loadable_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(refresh, "_http_fetch", _fetcher({"http://gw/model/info": GATEWAY}))
    out = tmp_path / "rates.json"
    code = refresh.main(
        ["refresh", "--endpoint", "http://gw", "--codex-model", "gpt-5.5", "--out", str(out)]
    )
    assert code == 0
    rates = load_rates(out)
    assert rates["openai:codex"].input == 2e-6
    err = capsys.readouterr().err
    assert "wrote" in err and "export SPOTLIGHTS_RATES_FILE" in err


def test_main_dry_run_writes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(refresh, "_http_fetch", _fetcher({"http://gw/model/info": GATEWAY}))
    out = tmp_path / "rates.json"
    code = refresh.main(["refresh", "--endpoint", "http://gw", "--dry-run", "--out", str(out)])
    assert code == 0
    assert not out.exists()
    assert "openai:gpt-5.5" in json.loads(capsys.readouterr().out)


def test_main_without_endpoint_fails(capsys):
    assert refresh.main(["refresh"]) == 1
    assert "no LiteLLM endpoint" in capsys.readouterr().err


def test_main_sends_bearer_from_api_key_env(monkeypatch, tmp_path):
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(refresh, "_http_fetch", _fetcher({"http://gw/model/info": GATEWAY}, calls))
    monkeypatch.setenv("MY_KEY", "secret")
    refresh.main(["refresh", "--endpoint", "http://gw", "--api-key-env", "MY_KEY", "--dry-run"])
    assert calls[0][1]["Authorization"] == "Bearer secret"


def test_cli_dispatches_rates_subcommand(monkeypatch):
    from spotlights_engine import cli

    seen: list[list[str]] = []
    monkeypatch.setattr(refresh, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["rates", "refresh", "--dry-run"]) == 0
    assert seen == [["refresh", "--dry-run"]]


def test_written_table_has_no_duplicate_canonical_keys(monkeypatch, tmp_path):
    # Guard for the bug the DAM script only avoided by luck: a gateway alias
    # with a stripped route prefix must not sit beside the bundled key.
    monkeypatch.setattr(refresh, "_http_fetch", _fetcher({"http://gw/model/info": GATEWAY}))
    out = tmp_path / "rates.json"
    refresh.main(["refresh", "--endpoint", "http://gw", "--out", str(out)])
    keys = [k for k in json.loads(Path(out).read_text()) if not k.startswith("_")]
    assert len(keys) == len(set(keys))
    load_rates(out)
