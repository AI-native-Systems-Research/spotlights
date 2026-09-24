"""`signoz-sql` — the signal pipeline's one read-only SQL tool against SigNoz.

The SQL analogue of the `jq` the file-based path relies on: run a ClickHouse
query against captured OTel data in SigNoz and hand back clean row dicts. Two
modes, both invoked by the stage-01 agent via `Bash`:

    python -m spotlights_engine.signal_pipeline.signoz_tool --list-runs
    python -m spotlights_engine.signal_pipeline.signoz_tool --run <run_id> "<SELECT ...>"

`--run` prepends a canonical, run-scoped CTE block (the `<run_id>` substituted
exactly once) so the agent only ever authors analytics over the stable aliases
`spans` / `logs` / `metric_samples` — never the raw tables, the metrics
fingerprint-join, or a `run_id` filter. The split SigNoz response (`labels`
strings + `values[].value` numerics) is merged here so the agent always sees
flat dicts. See `docs/signal-based/signoz_signal_extraction.md` (design) and
`discovery_observability/docs/signoz-data-model.md` (producer contract).

Read-only by construction: the configured `SIGNOZ_API_KEY` is Viewer-scoped
(server-enforced — the real guarantee). The key is read from `.env` here so it
never lands on the agent's command line or in the pipeline's `_logs/`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


class SignozError(RuntimeError):
    """Any failure resolving credentials or querying SigNoz."""


class Credentials:
    """SigNoz endpoint + API key, sourced from `.env`.

    Reading the key here (rather than letting the agent hand-build a `curl`)
    is what keeps it off the agent's command line and out of `_logs/`."""

    def __init__(self, url: str, api_key: str) -> None:
        self.url = url
        self.api_key = api_key

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Credentials:
        # `env_path=None` lets python-dotenv find the project-root `.env`
        # (it walks up from this file to the root); tests pass an explicit path.
        load_dotenv(env_path)
        url = os.getenv("SIGNOZ_URL")
        api_key = os.getenv("SIGNOZ_API_KEY")
        if not url or not api_key:
            raise SignozError("SIGNOZ_URL and/or SIGNOZ_API_KEY missing from .env")
        return cls(url, api_key)


class SignozClient:
    """Read-only ClickHouse-over-SigNoz client.

    Owns the canonical, tool-only SQL (run discovery + run-scoping CTEs), the
    request envelope, and the split-response merge — everything the agent must
    not re-derive per run. Only the analytical SQL passed to `run_query` is the
    agent's; `run_id` plumbing lives here and varies only by the validated id."""

    _API_PATH = "/api/v4/query_range"

    # The query API requires start/end (epoch ms) but treats them as a wide
    # outer bound only — `run_id` does the selecting, not time (producer doc §
    # "Selecting a run"). A fixed 2020..2100 window covers any retention with
    # no clock read, keeping the request deterministic.
    _WINDOW_START_MS = 1_577_836_800_000  # 2020-01-01T00:00:00Z
    _WINDOW_END_MS = 4_102_444_800_000  # 2100-01-01T00:00:00Z

    # `run_id` is canonical (UTC start-timestamp) and must never carry agent-
    # or user-authored SQL. Validating the shape before substituting it into
    # the CTE template is the guard against quote-breakout injection.
    _RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")

    # Canonical run discovery — distinct non-empty run_id across traces
    # (producer doc § "Discover what runs exist"). Tool-owned.
    _LIST_RUNS_SQL = (
        "SELECT DISTINCT resources_string['run_id'] AS run_id "
        "FROM signoz_traces.signoz_index_v3 "
        "WHERE resources_string['run_id'] != ''"
    )

    # Canonical run-scoped CTEs. `<id>` is substituted exactly once (validated);
    # the agent's SQL is appended and references only the three aliases. Absorbs
    # per-signal run_id placement, the metrics fingerprint join, and the
    # self-telemetry (run_id='') exclusion — see design §4.1.
    #
    # NB: the `SELECT *` in the spans/logs CTEs is intentional and safe — do not
    # "fix" it to an explicit scalar column list. It must stay `*` so the
    # Map columns (`attributes_number`, `attributes_string`, `resources_string`)
    # are available for by-key access in the agent's SQL
    # (e.g. `attributes_number['gen_ai.latency.e2e']`) — the core of the
    # analysis. It does NOT trigger the "HTTP 500: JSON Scan value must be
    # clickhouse.JSON…" error: ClickHouse prunes columns the outer query never
    # references, so a Map column is only serialized to the SigNoz API when the
    # agent's *outer* query projects it (a whole-map / `SELECT *` in the final
    # result). That outer-projection case is what the prompt forbids; the inner
    # CTE `SELECT *` is pruned away for scalar queries. (Verified live: scalar
    # queries over `spans` return rows through this template.)
    _SCOPED_CTE_TEMPLATE = (
        "WITH\n"
        "  spans AS (SELECT * FROM signoz_traces.signoz_index_v3\n"
        "            WHERE resources_string['run_id'] = '<id>'),\n"
        "  logs  AS (SELECT * FROM signoz_logs.logs_v2\n"
        "            WHERE resources_string['run_id'] = '<id>'),\n"
        "  run_fp AS (SELECT fingerprint FROM signoz_metrics.time_series_v4\n"
        "             WHERE JSONExtractString(labels,'run_id') = '<id>'),\n"
        "  metric_samples AS (SELECT s.metric_name, s.unix_milli, s.value\n"
        "                     FROM signoz_metrics.samples_v4 s\n"
        "                     INNER JOIN run_fp t ON s.fingerprint = t.fingerprint)\n"
    )

    def __init__(self, credentials: Credentials, *, timeout: int = 60) -> None:
        self._credentials = credentials
        self._timeout = timeout

    @classmethod
    def from_env(cls, env_path: Path | None = None, *, timeout: int = 60) -> SignozClient:
        return cls(Credentials.from_env(env_path), timeout=timeout)

    # ── query construction (tool-owned) ─────────────────────────────────

    @classmethod
    def build_scoped_sql(cls, run_id: str, agent_sql: str) -> str:
        """Prepend the canonical run-scoped CTEs to the agent's analytical SQL.

        `run_id` is validated to the canonical `YYYYMMDDTHHMMSSZ` form and
        substituted once per alias — the agent never authors the filter."""
        if not cls._RUN_ID_RE.match(run_id):
            raise ValueError(
                f"run_id {run_id!r} is not the canonical YYYYMMDDTHHMMSSZ form"
            )
        return cls._SCOPED_CTE_TEMPLATE.replace("<id>", run_id) + agent_sql

    @classmethod
    def build_request_body(cls, sql: str) -> dict:
        """Wrap SQL in the `clickhouse_sql` table-panel envelope SigNoz expects."""
        return {
            "start": cls._WINDOW_START_MS,
            "end": cls._WINDOW_END_MS,
            "step": 60,
            "compositeQuery": {
                "queryType": "clickhouse_sql",
                "panelType": "table",
                "chQueries": {"A": {"query": sql, "disabled": False}},
            },
        }

    # ── response merge ──────────────────────────────────────────────────

    @staticmethod
    def _coerce_number(value):
        """SigNoz returns numerics as strings. Coerce to int/float; leave
        anything non-numeric (or None) untouched."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    @classmethod
    def merge_series_rows(cls, response: dict) -> list[dict]:
        """Merge SigNoz's split response into flat row dicts.

        Results arrive as `data.result[0].series`, one entry per row, with
        string columns under `labels` and numeric/aggregate columns under
        `values[].value` (as strings). A single numeric merges as `value`;
        multiple as `values`; a row with no numeric (e.g. `--list-runs`) is
        just its labels."""
        result = (response.get("data") or {}).get("result") or []
        if not result:
            return []
        series = result[0].get("series") or []
        rows: list[dict] = []
        for entry in series:
            row = dict(entry.get("labels") or {})
            nums = [
                cls._coerce_number(v.get("value"))
                for v in (entry.get("values") or [])
                if "value" in v
            ]
            if len(nums) == 1:
                row["value"] = nums[0]
            elif len(nums) > 1:
                row["values"] = nums
            rows.append(row)
        return rows

    # ── HTTP + public queries ───────────────────────────────────────────

    def _post(self, sql: str) -> dict:
        """POST one ClickHouse query to SigNoz; return the parsed JSON."""
        endpoint = self._credentials.url.rstrip("/") + self._API_PATH
        body = json.dumps(self.build_request_body(sql)).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "SIGNOZ-API-KEY": self._credentials.api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise SignozError(f"SigNoz query failed (HTTP {e.code}): {detail}") from e
        except urllib.error.URLError as e:
            raise SignozError(f"could not reach SigNoz at {endpoint}: {e}") from e

    def list_runs(self) -> list[str]:
        """Distinct non-empty `run_id`s, sorted (the YYYYMMDDTHHMMSSZ id sorts
        chronologically, so the max is the latest run)."""
        rows = self.merge_series_rows(self._post(self._LIST_RUNS_SQL))
        return sorted({r["run_id"] for r in rows if r.get("run_id")})

    def run_query(self, run_id: str, agent_sql: str) -> list[dict]:
        """Run the agent's SQL on top of the canonical run-scoped CTEs."""
        return self.merge_series_rows(self._post(self.build_scoped_sql(run_id, agent_sql)))


class SignozCli:
    """`signoz-sql` entry point — argument parsing and dispatch."""

    @classmethod
    def main(cls, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(
            prog="signoz-sql",
            description="Read-only ClickHouse SQL against captured OTel data in SigNoz.",
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--list-runs",
            action="store_true",
            help="list distinct run_ids present in SigNoz",
        )
        group.add_argument(
            "--run",
            metavar="RUN_ID",
            help="run SQL over the run-scoped aliases spans/logs/metric_samples",
        )
        parser.add_argument(
            "sql",
            nargs="?",
            help="SQL over spans/logs/metric_samples (required with --run)",
        )
        args = parser.parse_args(argv)

        try:
            client = SignozClient.from_env()
            if args.list_runs:
                if args.sql is not None:
                    parser.error("--list-runs takes no SQL argument")
                payload = client.list_runs()
            else:
                if not args.sql:
                    parser.error("--run requires a SQL argument")
                payload = client.run_query(args.run, args.sql)
        except (SignozError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 1

        print(json.dumps(payload, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(SignozCli.main())
