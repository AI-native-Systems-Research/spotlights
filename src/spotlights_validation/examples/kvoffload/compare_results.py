#!/usr/bin/env python3
"""Compare validation results: baseline vs. change.

Produces two comparison tables in CSV and Markdown:
  (a) Unit/Integration/Correctness/Stress tests
  (b) Benchmark metrics
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

BASELINE_RESULT = OUTPUT_DIR / "baseline" / "validation_result_baseline.json"
BASELINE_PLAN = OUTPUT_DIR / "baseline" / "validation_plan_baseline.json"
CHANGE_RESULT = OUTPUT_DIR / "change" / "validation_result_change.json"
CHANGE_PLAN = OUTPUT_DIR / "change" / "validation_plan_change.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def determine_test_status(plan_entry, result_entry):
    if plan_entry and plan_entry.get("skipped"):
        return "skipped"
    if result_entry is None:
        return "not_executed"
    if result_entry.get("failed", 0) > 0:
        return "fail"
    if result_entry.get("passed", 0) > 0:
        return "pass"
    return "no_tests"


def determine_benchmark_status(plan_entry, result_entry):
    if plan_entry and plan_entry.get("skipped"):
        return "skipped"
    if result_entry is None:
        return "not_executed"
    if result_entry.get("metrics"):
        return "pass"
    return "no_data"


def build_plan_index(plan):
    index = {}
    for entry in plan.get("entries", []):
        hid = entry["harness_entry"]["id"]
        index[hid] = entry
    return index


def build_test_result_index(result):
    index = {}
    for tr in result.get("test_results", []):
        index[tr["harness_id"]] = tr
    return index


def build_benchmark_result_index(result):
    index = {}
    for br in result.get("benchmark_results", []):
        script = br.get("script", "")
        if "LRU" in script:
            index["lru"] = br
        elif "ARC" in script:
            index["arc"] = br
        elif "evolved" in script:
            index["evolved"] = br
    return index


def write_csv(rows, headers, path):
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def write_md(rows, headers, path):
    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")


def test_comparison_table():
    baseline_plan = build_plan_index(load_json(BASELINE_PLAN))
    change_plan = build_plan_index(load_json(CHANGE_PLAN))
    baseline_result = build_test_result_index(load_json(BASELINE_RESULT))
    change_result = build_test_result_index(load_json(CHANGE_RESULT))

    test_kinds = {"unit", "integration", "correctness", "stress"}
    all_ids = []
    for entry in load_json(BASELINE_PLAN).get("entries", []):
        hid = entry["harness_entry"]["id"]
        if entry["harness_entry"]["kind"] in test_kinds:
            all_ids.append(hid)

    headers = ["row", "index", "id", "status", "priority", "halt_on_failure",
               "invoke", "pass", "failed", "skipped"]
    rows = []

    for hid in all_ids:
        plan_entry = baseline_plan.get(hid)
        result_entry = baseline_result.get(hid)
        status = determine_test_status(plan_entry, result_entry)
        idx = plan_entry["index"] if plan_entry else ""
        priority = plan_entry["priority"] if plan_entry else ""
        halt = plan_entry["halt_on_failure"] if plan_entry else ""
        invoke = plan_entry["harness_entry"]["invoke"] if plan_entry else ""
        passed = result_entry.get("passed", "") if result_entry else ""
        failed = result_entry.get("failed", "") if result_entry else ""
        skipped_count = result_entry.get("skipped", "") if result_entry else ""
        rows.append(["baseline", idx, hid, status, priority, halt,
                     invoke, passed, failed, skipped_count])

    for hid in all_ids:
        plan_entry = change_plan.get(hid)
        result_entry = change_result.get(hid)
        status = determine_test_status(plan_entry, result_entry)
        idx = plan_entry["index"] if plan_entry else ""
        priority = plan_entry["priority"] if plan_entry else ""
        halt = plan_entry["halt_on_failure"] if plan_entry else ""
        invoke = plan_entry["harness_entry"]["invoke"] if plan_entry else ""
        passed = result_entry.get("passed", "") if result_entry else ""
        failed = result_entry.get("failed", "") if result_entry else ""
        skipped_count = result_entry.get("skipped", "") if result_entry else ""
        rows.append(["evolved", idx, hid, status, priority, halt,
                     invoke, passed, failed, skipped_count])

    return headers, rows


def benchmark_comparison_table():
    baseline_plan = build_plan_index(load_json(BASELINE_PLAN))
    change_plan = build_plan_index(load_json(CHANGE_PLAN))
    baseline_result = load_json(BASELINE_RESULT)
    change_result = load_json(CHANGE_RESULT)

    baseline_benchmarks = build_benchmark_result_index(baseline_result)
    change_benchmarks = build_benchmark_result_index(change_result)

    all_metric_names = []
    for br in (list(baseline_benchmarks.values()) + list(change_benchmarks.values())):
        for m in br.get("metrics", []):
            if m["name"] not in all_metric_names:
                all_metric_names.append(m["name"])

    headers = ["row", "index", "id", "status", "priority", "halt_on_failure",
               "invoke"] + all_metric_names
    rows = []

    configs = [
        ("baseline-lru", "benchmark-multi-turn-kv-offload-lab-lru",
         baseline_plan, baseline_benchmarks.get("lru")),
        ("baseline-arc", "benchmark-multi-turn-kv-offload-lab-arc",
         baseline_plan, baseline_benchmarks.get("arc")),
        ("evolved", "benchmark-multi-turn-kv-offload-lab",
         change_plan, change_benchmarks.get("evolved")),
    ]

    for row_label, hid, plan, result_entry in configs:
        plan_entry = plan.get(hid)
        status = determine_benchmark_status(plan_entry, result_entry)
        idx = plan_entry["index"] if plan_entry else ""
        priority = plan_entry["priority"] if plan_entry else ""
        halt = plan_entry["halt_on_failure"] if plan_entry else ""
        invoke = plan_entry["harness_entry"]["invoke"] if plan_entry else ""

        metric_values = {}
        if result_entry:
            for m in result_entry.get("metrics", []):
                metric_values[m["name"]] = m["measured_value"]

        row = [row_label, idx, hid, status, priority, halt, invoke]
        for mn in all_metric_names:
            row.append(metric_values.get(mn, ""))
        rows.append(row)

    return headers, rows


def get_failure_reasons(result_entry):
    if not result_entry:
        return []
    return [err.get("message", "unknown") for err in result_entry.get("errors", [])]


def regressions_and_fixes_section():
    baseline_plan = build_plan_index(load_json(BASELINE_PLAN))
    change_plan = build_plan_index(load_json(CHANGE_PLAN))
    baseline_result = build_test_result_index(load_json(BASELINE_RESULT))
    change_result = build_test_result_index(load_json(CHANGE_RESULT))

    test_kinds = {"unit", "integration", "correctness", "stress"}
    all_ids = []
    for entry in load_json(BASELINE_PLAN).get("entries", []):
        hid = entry["harness_entry"]["id"]
        if entry["harness_entry"]["kind"] in test_kinds:
            all_ids.append(hid)

    regressions = []
    fixes = []

    for hid in all_ids:
        baseline_status = determine_test_status(baseline_plan.get(hid), baseline_result.get(hid))
        change_status = determine_test_status(change_plan.get(hid), change_result.get(hid))

        if baseline_status == "pass" and change_status == "fail":
            reasons = get_failure_reasons(change_result.get(hid))
            regressions.append((hid, reasons))
        elif baseline_status == "fail" and change_status == "pass":
            reasons = get_failure_reasons(baseline_result.get(hid))
            fixes.append((hid, reasons))

    lines = []
    lines.append("## Regressions (passed in baseline, failed in evolved)\n")
    if regressions:
        for hid, reasons in regressions:
            lines.append(f"- **{hid}**")
            for reason in reasons:
                lines.append(f"  - {reason}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("## Fixes (failed in baseline, passed in evolved)\n")
    if fixes:
        for hid, reasons in fixes:
            lines.append(f"- **{hid}**")
            lines.append("  - Baseline failure reason:")
            for reason in reasons:
                lines.append(f"    - {reason}")
    else:
        lines.append("None")
    lines.append("")

    return "\n".join(lines)


def write_combined_md(test_headers, test_rows, bench_headers, bench_rows, path):
    with open(path, "w") as f:
        f.write("# Validation Results Comparison\n\n")

        f.write(regressions_and_fixes_section())
        f.write("\n")

        f.write("## Tests Comparison\n\n")
        f.write("| " + " | ".join(test_headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in test_headers) + " |\n")
        for row in test_rows:
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")
        f.write("\n")

        f.write("## Benchmark Comparison\n\n")
        f.write("| " + " | ".join(bench_headers) + " |\n")
        f.write("| " + " | ".join("---" for _ in bench_headers) + " |\n")
        for row in bench_rows:
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")


def main():
    output_path = OUTPUT_DIR / "comparison"
    output_path.mkdir(parents=True, exist_ok=True)

    test_headers, test_rows = test_comparison_table()
    write_csv(test_rows, test_headers, output_path / "tests_comparison.csv")

    bench_headers, bench_rows = benchmark_comparison_table()
    write_csv(bench_rows, bench_headers, output_path / "benchmark_comparison.csv")

    md_path = output_path / "summary.md"
    write_combined_md(test_headers, test_rows, bench_headers, bench_rows, md_path)

    print(f"Output written to {output_path}/")
    print()
    with open(md_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
