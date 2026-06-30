"""Unit tests for `signoz_tool` — the read-only SigNoz SQL primitive.

No live dependency: the only HTTP seam (`SignozClient._post`) is monkeypatched
to return captured-shape response fixtures, per the verification plan in
`docs/signal-based/signoz_signal_extraction.md` §7.
"""

from __future__ import annotations

import pytest

from spotlights_engine.signal_pipeline.signoz_tool import (
    Credentials,
    SignozClient,
    SignozError,
)

# Captured response shape (producer doc § "Response shape"): one series per
# row, string columns under `labels`, numeric columns under `values[].value`
# (as strings). `SELECT serviceName, count()` → this.
_COUNT_RESPONSE = {
    "status": "success",
    "data": {
        "result": [
            {
                "queryName": "A",
                "series": [
                    {"labels": {"serviceName": "vllm"}, "values": [{"value": "43491"}]},
                ],
            }
        ]
    },
}

# `--list-runs` shape: a single string column, no numeric → empty `values`.
# Includes a duplicate to exercise dedup and out-of-order ids to exercise sort.
_LIST_RUNS_RESPONSE = {
    "status": "success",
    "data": {
        "result": [
            {
                "series": [
                    {"labels": {"run_id": "20260615T073440Z"}, "values": []},
                    {"labels": {"run_id": "20260614T120000Z"}, "values": []},
                    {"labels": {"run_id": "20260615T073440Z"}, "values": []},
                ]
            }
        ]
    },
}


def _client_with(monkeypatch, response):
    """A client whose HTTP seam returns `response` and records the SQL sent."""
    client = SignozClient(Credentials("http://signoz.test:8080", "viewer-key"))
    sent: dict = {}

    def fake_post(self, sql):
        sent["sql"] = sql
        return response

    monkeypatch.setattr(SignozClient, "_post", fake_post)
    return client, sent


# ── Credentials / .env ──────────────────────────────────────────────────


@pytest.fixture
def clean_signoz_env(monkeypatch):
    """`from_env` loads via `load_dotenv`, which mutates `os.environ`.
    `delenv` here gives each test a clean slate *and* tags the keys for
    monkeypatch teardown, so `load_dotenv`'s writes don't leak across tests."""
    monkeypatch.delenv("SIGNOZ_URL", raising=False)
    monkeypatch.delenv("SIGNOZ_API_KEY", raising=False)


def test_from_env_reads_url_and_key(tmp_path, clean_signoz_env):
    env = tmp_path / ".env"
    env.write_text("SIGNOZ_URL=http://h:8080\nSIGNOZ_API_KEY=k123=\n")
    creds = Credentials.from_env(env)
    assert creds.url == "http://h:8080"
    assert creds.api_key == "k123="  # base64 '=' preserved by python-dotenv


def test_from_env_missing_key_raises(tmp_path, clean_signoz_env):
    env = tmp_path / ".env"
    env.write_text("SIGNOZ_URL=http://h:8080\n")  # no API key
    with pytest.raises(SignozError, match="SIGNOZ_API_KEY"):
        Credentials.from_env(env)


# ── Scoped-SQL construction ─────────────────────────────────────────────


def test_build_scoped_sql_substitutes_run_id_into_every_cte():
    run_id = "20260615T073440Z"
    sql = SignozClient.build_scoped_sql(run_id, "SELECT count() FROM spans")
    # run_id appears once per alias-scoping clause (spans, logs, run_fp) = 3×.
    assert sql.count(run_id) == 3
    # Canonical plumbing is present; the agent's SQL is appended verbatim.
    assert "signoz_index_v3" in sql
    assert "INNER JOIN run_fp" in sql
    assert sql.rstrip().endswith("SELECT count() FROM spans")
    # No unsubstituted placeholder leaks through.
    assert "<id>" not in sql


@pytest.mark.parametrize(
    "bad",
    [
        "2026-06-15",  # wrong format
        "20260615T073440Z'; DROP TABLE x; --",  # injection attempt
        "latest",
        "",
    ],
)
def test_build_scoped_sql_rejects_non_canonical_run_id(bad):
    with pytest.raises(ValueError, match="canonical"):
        SignozClient.build_scoped_sql(bad, "SELECT 1")


# ── Request envelope ────────────────────────────────────────────────────


def test_build_request_body_wraps_clickhouse_sql_table_panel():
    body = SignozClient.build_request_body("SELECT 1")
    cq = body["compositeQuery"]
    assert cq["queryType"] == "clickhouse_sql"
    assert cq["panelType"] == "table"
    assert cq["chQueries"]["A"]["query"] == "SELECT 1"
    assert cq["chQueries"]["A"]["disabled"] is False
    # start/end are a wide outer bound, not a selector.
    assert body["start"] < body["end"]


# ── Response merge ──────────────────────────────────────────────────────


def test_merge_single_numeric_becomes_value():
    rows = SignozClient.merge_series_rows(_COUNT_RESPONSE)
    assert rows == [{"serviceName": "vllm", "value": 43491}]


def test_merge_string_only_row_is_just_labels():
    rows = SignozClient.merge_series_rows(_LIST_RUNS_RESPONSE)
    assert rows[0] == {"run_id": "20260615T073440Z"}
    assert all("value" not in r and "values" not in r for r in rows)


def test_merge_multiple_numerics_become_values_list():
    response = {
        "data": {
            "result": [
                {
                    "series": [
                        {
                            "labels": {"span_name": "llm_request"},
                            "values": [{"value": "0.012"}, {"value": "0.230"}],
                        }
                    ]
                }
            ]
        }
    }
    rows = SignozClient.merge_series_rows(response)
    assert rows == [{"span_name": "llm_request", "values": [0.012, 0.230]}]


def test_merge_empty_result_is_empty_list():
    assert SignozClient.merge_series_rows({"data": {"result": []}}) == []
    assert SignozClient.merge_series_rows({}) == []


def test_coerce_number_handles_int_float_and_passthrough():
    coerce = SignozClient._coerce_number
    assert coerce("43491") == 43491 and isinstance(coerce("43491"), int)
    assert coerce("0.5") == 0.5 and isinstance(coerce("0.5"), float)
    assert coerce("not-a-number") == "not-a-number"
    assert coerce(None) is None


# ── Public queries (HTTP mocked) ────────────────────────────────────────


def test_list_runs_returns_sorted_distinct(monkeypatch):
    client, sent = _client_with(monkeypatch, _LIST_RUNS_RESPONSE)
    runs = client.list_runs()
    assert runs == ["20260614T120000Z", "20260615T073440Z"]  # sorted, deduped
    assert "run_id" in sent["sql"]  # the canonical discovery query ran


def test_run_query_scopes_then_merges(monkeypatch):
    client, sent = _client_with(monkeypatch, _COUNT_RESPONSE)
    rows = client.run_query("20260615T073440Z", "SELECT serviceName, count() FROM spans")
    # The SQL that hit the wire was the agent's, wrapped in canonical CTEs.
    assert "20260615T073440Z" in sent["sql"]
    assert "metric_samples AS" in sent["sql"]
    assert sent["sql"].rstrip().endswith("SELECT serviceName, count() FROM spans")
    # And the split response came back merged.
    assert rows == [{"serviceName": "vllm", "value": 43491}]


def test_run_query_rejects_bad_run_id_before_any_http(monkeypatch):
    client = SignozClient(Credentials("http://signoz.test:8080", "k"))

    def boom(self, sql):  # pragma: no cover - must not be reached
        raise AssertionError("HTTP must not fire for an invalid run_id")

    monkeypatch.setattr(SignozClient, "_post", boom)
    with pytest.raises(ValueError, match="canonical"):
        client.run_query("nope", "SELECT 1")


# ── HTTP error surfacing ────────────────────────────────────────────────


def test_post_wraps_http_error_as_signoz_error(monkeypatch):
    import urllib.error

    client = SignozClient(Credentials("http://signoz.test:8080", "k"))

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", hdrs=None, fp=_BytesIO(b"read-only key")
        )

    from spotlights_engine.signal_pipeline import signoz_tool

    monkeypatch.setattr(signoz_tool.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SignozError, match="HTTP 403"):
        client.run_query("20260615T073440Z", "DROP TABLE x")


class _BytesIO:
    """Minimal file-like for HTTPError.read() in the test above."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data
