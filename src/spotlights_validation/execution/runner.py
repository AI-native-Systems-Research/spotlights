"""Phase 2 execution: run a ValidationPlan against a source tree."""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from spotlights_validation.schemas import (
    BenchmarkOutputTemplate,
    BenchmarkResult,
    MetricResult,
    TestError,
    TestResult,
    ValidationPlan,
    ValidationPlanEntry,
    ValidationResult,
)

if TYPE_CHECKING:
    pass

# Detect whether pytest-json-report is installed once at import time.
try:
    import pytest_jsonreport  # noqa: F401
    _HAS_JSON_REPORT = True
except ImportError:
    _HAS_JSON_REPORT = False


def run_validation_plan(
    plan: ValidationPlan,
    source_tree: Path,
    *,
    dry_run: bool = False,
    timeout_multiplier: float = 2.0,
    logs_dir: Path | None = None,
) -> ValidationResult:
    """Execute every entry in *plan* against *source_tree* in priority order.

    Returns a ValidationResult with pass/fail/conditional verdict.
    When *dry_run* is True, commands are logged but not executed.
    When *logs_dir* is set, per-entry combined stdout/stderr and pytest-json
    reports are persisted there so failure details survive the run.
    """
    entries = sorted(plan.entries, key=lambda e: e.priority)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing per-entry logs to: {logs_dir}", flush=True)

    test_results: list[TestResult] = []
    benchmark_results: list[BenchmarkResult] = []
    halt_triggered = False
    halted_at: str | None = None
    skipped_entries: list[str] = []
    plan_skipped: list[tuple[str, str]] = []

    total = len(entries)
    for fallback_index, entry in enumerate(entries, start=1):
        h = entry.harness_entry

        if halt_triggered:
            skipped_entries.append(h.name)
            continue

        index = entry.index if entry.index is not None else fallback_index

        if entry.skipped:
            reason = entry.skip_reason or "marked skipped in plan"
            print(
                f"[{index}/{total}] [P{entry.priority}] "
                f"{h.kind.upper():12s}  {h.name}  -- SKIPPED ({reason})",
                flush=True,
            )
            plan_skipped.append((h.name, reason))
            continue

        print(
            f"[{index}/{total}] [P{entry.priority}] "
            f"{h.kind.upper():12s}  {h.name}",
            flush=True,
        )

        if dry_run:
            _record_dry_run(entry, test_results, benchmark_results)
            continue

        result, failed = _execute_entry(
            entry, source_tree, timeout_multiplier, logs_dir
        )

        if h.kind == "benchmark":
            benchmark_results.append(result)  # type: ignore[arg-type]
            print(f"    -> captured ({result.workload_id})", flush=True)
        else:
            test_results.append(result)  # type: ignore[arg-type]
            status = "FAIL" if failed else "pass"
            print(
                f"    -> {status} (passed={result.passed} failed={result.failed} "
                f"skipped={result.skipped} in {result.duration_seconds:.1f}s)",
                flush=True,
            )
            for err in result.errors:
                location = (
                    f" {err.file}:{err.lineno}"
                    if err.file and err.lineno is not None
                    else ""
                )
                first_line = err.message.splitlines()[0] if err.message else ""
                print(f"       ✗ {err.nodeid}{location}", flush=True)
                if first_line:
                    print(f"         {first_line}", flush=True)
            if result.log_path:
                print(f"       log: {result.log_path}", flush=True)

        if failed and entry.halt_on_failure:
            halt_triggered = True
            halted_at = h.name
            print(f"  HALT — {h.name} failed and halt_on_failure=true", flush=True)

    verdict, reasoning, conditions = _compute_verdict(
        test_results, benchmark_results, halt_triggered, halted_at, skipped_entries
    )

    notes_parts: list[str] = []
    if skipped_entries:
        notes_parts.append(f"Skipped due to halt: {', '.join(skipped_entries)}")
    if plan_skipped:
        rendered = ", ".join(
            f"{name} ({reason})" if reason else name for name, reason in plan_skipped
        )
        notes_parts.append(f"Plan-level skips: {rendered}")
    if dry_run:
        notes_parts.append("dry-run: commands were not executed")

    return ValidationResult(
        change_ref=plan.change_ref,
        verdict=verdict,
        verdict_reasoning=reasoning,
        conditions=conditions,
        test_results=test_results,
        benchmark_results=benchmark_results,
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _stream_subprocess(
    cmd: list[str],
    cwd: Path,
    timeout: float | None,
    prefix: str = "    │ ",
) -> tuple[int, str]:
    """Run *cmd* with stdout+stderr streamed to the terminal as it arrives.

    Returns ``(returncode, captured_combined_output)``. Combined output is the
    interleaved stdout/stderr stream, the same text shown live.
    """
    print(f"    cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    print(f"    cwd: {cwd}", flush=True)
    print(
        f"    timeout: {f'{timeout:.0f}s' if timeout else 'none'}",
        flush=True,
    )

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    captured: list[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            captured.append(line)
            sys.stdout.write(prefix + line)
            sys.stdout.flush()

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        t.join(timeout=2.0)
        raise

    t.join()
    return proc.returncode, "".join(captured)


def _execute_entry(
    entry: ValidationPlanEntry,
    source_tree: Path,
    timeout_multiplier: float,
    logs_dir: Path | None,
) -> tuple[TestResult | BenchmarkResult, bool]:
    """Run the entry's invoke command and return (result, failed)."""
    h = entry.harness_entry
    timeout = (
        h.estimated_duration * timeout_multiplier if h.estimated_duration else None
    )

    if h.kind == "benchmark":
        return _run_benchmark(entry, source_tree, timeout, logs_dir)

    return _run_test(entry, source_tree, timeout, logs_dir)


def _run_test(
    entry: ValidationPlanEntry,
    source_tree: Path,
    timeout: float | None,
    logs_dir: Path | None,
) -> tuple[TestResult, bool]:
    h = entry.harness_entry
    cmd = shlex.split(h.invoke)

    use_json_report = h.output_format == "pytest-json" and _HAS_JSON_REPORT

    json_report_target: Path | None = None
    log_target: Path | None = None
    if logs_dir is not None:
        json_report_target = (logs_dir / f"{h.id}.report.json").resolve()
        log_target = (logs_dir / f"{h.id}.log").resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        json_report_path = (
            json_report_target
            if json_report_target is not None
            else Path(tmpdir) / "report.json"
        )

        if use_json_report:
            cmd += ["--json-report", f"--json-report-file={json_report_path}"]

        t0 = time.monotonic()
        returncode, output = _stream_subprocess(cmd, source_tree, timeout)
        duration = time.monotonic() - t0

        if log_target is not None:
            log_target.write_text(output)

        passed, failed, skipped, errors = _parse_test_output(
            returncode,
            output,
            json_report_path if use_json_report else None,
            h.output_format,
        )

    result = TestResult(
        harness_id=h.id,
        script=h.name,
        kind=h.kind,
        invoke=h.invoke,
        index=entry.index,
        priority=entry.priority,
        halt_on_failure=entry.halt_on_failure,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        duration_seconds=round(duration, 2),
        log_path=str(log_target) if log_target else None,
        json_report_path=(
            str(json_report_target)
            if use_json_report and json_report_target and json_report_target.exists()
            else None
        ),
    )
    return result, failed > 0 or returncode != 0


def _run_benchmark(
    entry: ValidationPlanEntry,
    source_tree: Path,
    timeout: float | None,
    logs_dir: Path | None,
) -> tuple[BenchmarkResult, bool]:
    h = entry.harness_entry
    cmd = shlex.split(h.invoke)

    # Run once per workload; if no workloads just run once with a placeholder.
    workloads = entry.workloads or [None]  # type: ignore[list-item]
    results: list[BenchmarkResult] = []

    for wl in workloads:
        wl_cmd = cmd
        wl_id = wl.workload_id if wl else "default"
        if wl and wl.config_path:
            # Benchmarks that accept a --workload or --config flag: pass it if
            # the invoke doesn't already reference a config path.
            if wl.config_path not in h.invoke:
                wl_cmd = cmd + [f"--workload={wl.config_path}"]

        _returncode, output = _stream_subprocess(wl_cmd, source_tree, timeout)

        if logs_dir is not None:
            (logs_dir / f"{h.id}.{wl_id}.log").write_text(output)

        metrics = _extract_benchmark_metrics(output, source_tree, h.output_template)
        br = BenchmarkResult(
            workload_id=wl.workload_id if wl else "default",
            workload_class=wl.workload_class if wl else "batch-inference",
            script=h.name,
            invoke=h.invoke,
            metrics=metrics,
            raw_output=output[-8000:] if output else "",  # cap at 8 KB
        )
        results.append(br)

    # Return the first (or only) result; caller receives it.
    # For multi-workload benchmarks all results are packed as separate BenchmarkResult objects
    # but this function returns a single record; _execute_entry is called once per entry.
    # Multi-workload fan-out is handled in the loop above by returning the last result.
    combined_raw = "\n---\n".join(r.raw_output for r in results)
    combined_metrics: list[MetricResult] = []
    for r in results:
        combined_metrics.extend(r.metrics)
    combined = BenchmarkResult(
        workload_id=results[-1].workload_id,
        workload_class=results[-1].workload_class,
        script=h.name,
        invoke=h.invoke,
        index=entry.index,
        priority=entry.priority,
        halt_on_failure=entry.halt_on_failure,
        metrics=combined_metrics,
        raw_output=combined_raw,
    )
    return combined, False  # benchmarks never trigger halt


# ANSI color escape sequences leak into stdout from rich/colorama tools.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# `key = value` summary lines (e.g. `requests_per_sec = 1.095`).
_KV_EQUALS_RE = re.compile(
    r"^\s*([A-Za-z][\w\s/().%-]*?)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)

# `Label: value` summary lines (e.g. `Mean TTFT (ms):   123.45`).
# Restrict the label side to avoid matching log timestamps and prose.
_KV_COLON_RE = re.compile(
    r"^\s*([A-Za-z][\w()/%.\- ]{2,60}?):\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)

# Rejected label substrings — these come from prose lines, not metric tables.
_LABEL_BLOCKLIST = ("INFO", "DEBUG", "WARNING", "ERROR", "Using", "Estimated")


def _extract_benchmark_metrics(
    output: str,
    source_tree: Path,
    template: BenchmarkOutputTemplate | None,
) -> list[MetricResult]:
    """Lift summary metrics out of a benchmark's captured output.

    When *template* is set the runner follows it explicitly — different
    benchmarks emit different shapes (JSON file on disk, JSON on stdout,
    key=value lines, regex-extractable strings) and the template tells us
    which to use. Without a template we fall back to a generic line scraper.
    """
    fmt = template.format if template else "key-value-text"

    if fmt == "json-file":
        assert template is not None
        return _metrics_from_json_file(template, source_tree)
    if fmt == "json-stdout":
        assert template is not None
        return _metrics_from_json_text(output, template)
    if fmt == "regex-text":
        assert template is not None
        return _metrics_from_regex(output, template)
    return _metrics_from_key_value_text(output, template)


def _metrics_from_key_value_text(
    output: str, template: BenchmarkOutputTemplate | None
) -> list[MetricResult]:
    """Generic ``key = value`` / ``Label: value`` scraper. Last write wins."""
    if not output:
        return []

    seen: dict[str, float] = {}
    for raw_line in output.splitlines():
        line = _ANSI_RE.sub("", raw_line).rstrip()
        if not line or any(token in line for token in _LABEL_BLOCKLIST):
            continue
        m = _KV_EQUALS_RE.match(line) or _KV_COLON_RE.match(line)
        if not m:
            continue
        name = " ".join(m.group(1).split()).strip()
        try:
            seen[name] = float(m.group(2))
        except ValueError:
            continue

    if template and template.metric_paths:
        # Allowlist mode: only return metrics whose names match the template.
        wanted = set(template.metric_paths.keys())
        seen = {k: v for k, v in seen.items() if k in wanted}

    return [MetricResult(name=n, measured_value=v) for n, v in seen.items()]


def _metrics_from_json_file(
    template: BenchmarkOutputTemplate, source_tree: Path
) -> list[MetricResult]:
    pattern = template.source or ""
    if not pattern:
        return []

    expanded = os.path.expandvars(pattern)
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = source_tree / expanded

    matches = _resolve_glob(candidate)
    if not matches:
        return [
            MetricResult(
                name=f"<missing:{pattern}>",
                measured_value=None,
            )
        ]

    # Pick the most recently modified file when multiple match.
    chosen = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(chosen.read_text())
    except (OSError, json.JSONDecodeError):
        return [
            MetricResult(name=f"<unreadable:{chosen}>", measured_value=None)
        ]
    return _metrics_from_dict(data, template.metric_paths)


def _metrics_from_json_text(
    output: str, template: BenchmarkOutputTemplate
) -> list[MetricResult]:
    payload = _find_json_blob(output)
    if payload is None:
        return []
    return _metrics_from_dict(payload, template.metric_paths)


def _metrics_from_regex(
    output: str, template: BenchmarkOutputTemplate
) -> list[MetricResult]:
    metrics: list[MetricResult] = []
    for name, pattern in template.metric_paths.items():
        try:
            regex = re.compile(pattern, re.MULTILINE)
        except re.error:
            metrics.append(MetricResult(name=name, measured_value=None))
            continue
        match = regex.search(output)
        if not match or not match.groups():
            metrics.append(MetricResult(name=name, measured_value=None))
            continue
        try:
            metrics.append(
                MetricResult(name=name, measured_value=float(match.group(1)))
            )
        except ValueError:
            metrics.append(MetricResult(name=name, measured_value=None))
    return metrics


def _metrics_from_dict(
    payload: object, paths: dict[str, str]
) -> list[MetricResult]:
    metrics: list[MetricResult] = []
    if not paths:
        # No paths specified — flatten any numeric leaves under "metrics".
        flat = _flatten_numeric(payload)
        return [MetricResult(name=k, measured_value=v) for k, v in flat.items()]

    for name, dotted in paths.items():
        value = _get_dotted(payload, dotted)
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)):
            metrics.append(MetricResult(name=name, measured_value=float(value)))
        else:
            metrics.append(MetricResult(name=name, measured_value=None))
    return metrics


def _get_dotted(obj: object, dotted: str) -> object:
    """Walk a JSON-like structure by dotted key/index path. Returns None on miss."""
    cur: object = obj
    for part in dotted.split("."):
        if part == "":
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


def _flatten_numeric(obj: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, key))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _find_json_blob(text: str) -> object | None:
    """Return the first parseable JSON object/array in *text*, or None."""
    if not text:
        return None
    text = _ANSI_RE.sub("", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    return None


def _resolve_glob(candidate: Path) -> list[Path]:
    if "*" in str(candidate) or "?" in str(candidate):
        # Split into the longest non-glob root + the glob remainder.
        parts = candidate.parts
        for i, part in enumerate(parts):
            if "*" in part or "?" in part:
                root = Path(*parts[:i]) if i > 0 else Path(".")
                pattern = str(Path(*parts[i:]))
                return sorted(root.glob(pattern))
    return [candidate] if candidate.exists() else []


_LONGREPR_MAX_CHARS = 4000

# Pytest exit codes that mean "the process didn't run the tests it was asked to
# run" — distinct from "tests ran and some failed" (1). Without this, a
# collection error or all-deselected run produces summary={0,0,0} and would be
# silently recorded as a clean pass.
_PYTEST_EXIT_REASONS = {
    2: "test execution interrupted (collection or runtime error)",
    3: "pytest internal error",
    4: "pytest usage error",
    5: "no tests collected",
}


def _parse_test_output(
    returncode: int,
    output: str,
    json_path: Path | None,
    output_format: str,
) -> tuple[int, int, int, list[TestError]]:
    """Return (passed, failed, skipped, errors) from process output."""

    if output_format == "pytest-json" and json_path and json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            summary = data.get("summary", {})
            passed = summary.get("passed", 0)
            failed = summary.get("failed", 0)
            skipped = summary.get("skipped", 0)
            errors = [
                _test_error_from_json(t)
                for t in data.get("tests", [])
                if t.get("outcome") in ("failed", "error")
            ]
            collector_errors = [
                _test_error_from_collector(c)
                for c in data.get("collectors", [])
                if c.get("outcome") in ("failed", "error")
            ]
            errors.extend(collector_errors)
            # Collector errors aren't counted in summary.failed, so reflect
            # them in the failed count when the per-test summary is empty.
            if collector_errors and failed == 0:
                failed = len(collector_errors)

            report_exitcode = data.get("exitcode", returncode)
            no_tests_ran = (passed + failed + skipped) == 0
            framework_error = report_exitcode in _PYTEST_EXIT_REASONS
            if not errors and (no_tests_ran or framework_error):
                reason = _PYTEST_EXIT_REASONS.get(
                    report_exitcode, f"pytest exit code {report_exitcode}"
                )
                collected = summary.get("collected", 0)
                deselected = summary.get("deselected", 0)
                msg = (
                    f"{reason} (collected={collected} deselected={deselected} "
                    f"passed={passed} failed={failed} skipped={skipped})"
                )
                return 0, 1, 0, [
                    TestError(
                        nodeid=f"<pytest-exit-{report_exitcode}>",
                        message=msg,
                        longrepr=msg,
                    )
                ]
            return passed, failed, skipped, errors
        except Exception as exc:  # fall through to exit-code-only
            errors = [
                TestError(
                    nodeid="<json-report-parse-error>",
                    message=f"Could not parse pytest-json-report: {exc}",
                )
            ]
            if returncode == 0:
                return 1, 0, 0, errors
            return 0, 1, 0, errors

    if returncode == 0:
        return 1, 0, 0, []

    # Exit-code-only fallback: scrape FAILED/ERROR lines from combined output.
    error_lines = [
        line for line in output.splitlines()
        if "FAILED" in line or "ERROR" in line or "error" in line.lower()
    ][:10]
    err = TestError(
        nodeid=f"<exit-code-{returncode}>",
        message=(error_lines[0] if error_lines else f"exit code {returncode}"),
        longrepr="\n".join(error_lines)[:_LONGREPR_MAX_CHARS],
    )
    return 0, 1, 0, [err]


def _test_error_from_json(test_obj: dict) -> TestError:
    """Build a TestError from a pytest-json-report test entry."""
    nodeid = test_obj.get("nodeid", "?")
    # Pick the first failing phase: setup -> call -> teardown
    phase_name = "call"
    phase_obj: dict = {}
    for candidate in ("setup", "call", "teardown"):
        cand_obj = test_obj.get(candidate) or {}
        if cand_obj.get("outcome") in ("failed", "error"):
            phase_name = candidate
            phase_obj = cand_obj
            break
    if not phase_obj:
        phase_obj = test_obj.get("call") or {}

    crash = phase_obj.get("crash") or {}
    longrepr_raw = phase_obj.get("longrepr") or ""
    if isinstance(longrepr_raw, dict):  # pytest-json-report sometimes nests
        longrepr_raw = json.dumps(longrepr_raw)
    longrepr = str(longrepr_raw)[:_LONGREPR_MAX_CHARS]

    return TestError(
        nodeid=nodeid,
        phase=phase_name,
        message=str(crash.get("message", ""))[:1000],
        file=crash.get("path"),
        lineno=crash.get("lineno"),
        longrepr=longrepr,
    )


def _test_error_from_collector(coll_obj: dict) -> TestError:
    """Build a TestError from a pytest-json-report collector failure."""
    longrepr_raw = coll_obj.get("longrepr") or ""
    if isinstance(longrepr_raw, dict):
        longrepr_raw = json.dumps(longrepr_raw)
    longrepr = str(longrepr_raw)[:_LONGREPR_MAX_CHARS]
    first_line = longrepr.splitlines()[0] if longrepr else "collection failed"
    return TestError(
        nodeid=coll_obj.get("nodeid", "<collector>"),
        phase="collect",
        message=first_line[:1000],
        longrepr=longrepr,
    )


def _record_dry_run(
    entry: ValidationPlanEntry,
    test_results: list[TestResult],
    benchmark_results: list[BenchmarkResult],
) -> None:
    h = entry.harness_entry
    print(f"  would run: {h.invoke}")
    if h.kind == "benchmark":
        for wl in entry.workloads or [None]:  # type: ignore[list-item]
            benchmark_results.append(
                BenchmarkResult(
                    workload_id=wl.workload_id if wl else "default",
                    workload_class=wl.workload_class if wl else "batch-inference",
                    script=h.name,
                    invoke=h.invoke,
                    index=entry.index,
                    priority=entry.priority,
                    halt_on_failure=entry.halt_on_failure,
                    raw_output="[dry-run]",
                )
            )
    else:
        test_results.append(
            TestResult(
                harness_id=h.id,
                script=h.name,
                kind=h.kind,
                invoke=h.invoke,
                index=entry.index,
                priority=entry.priority,
                halt_on_failure=entry.halt_on_failure,
                passed=1,
                duration_seconds=0.0,
            )
        )


def _compute_verdict(
    test_results: list[TestResult],
    benchmark_results: list[BenchmarkResult],
    halt_triggered: bool,
    halted_at: str | None,
    skipped_entries: list[str],
) -> tuple[str, str, list[str]]:
    if halt_triggered:
        return (
            "fail",
            f"Validation halted after '{halted_at}' failed (halt_on_failure=true). "
            f"{len(skipped_entries)} entries were not executed.",
            [],
        )

    correctness_failures = [
        r for r in test_results if r.kind == "correctness" and r.failed > 0
    ]
    if correctness_failures:
        names = ", ".join(r.script for r in correctness_failures)
        return "fail", f"Correctness checks failed: {names}", []

    any_failure = any(r.failed > 0 for r in test_results)
    if any_failure:
        failed_names = [r.script for r in test_results if r.failed > 0]
        conditions = [f"recheck: {n}" for n in failed_names]
        return (
            "conditional",
            f"All halt-on-failure checks passed but {len(failed_names)} non-critical "
            f"entries reported failures.",
            conditions,
        )

    total_passed = sum(r.passed for r in test_results)
    return "pass", f"All {total_passed} test checks passed.", []
