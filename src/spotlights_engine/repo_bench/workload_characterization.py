"""Characterize workload types from filtered PR bodies.

Classifies each PR's benchmark workload as synthetic, recorded_trace,
combination, or unknown. Extracts generator parameters (input/output
lengths, concurrency, request rate) and trace sources (sharegpt, HF
datasets, timed traces).

Two-tier extraction:
  Tier 1 — Regex: parses `vllm bench`, `benchmark_serving.py`, and
    `python benchmarks/*.py` commands for flags.
  Tier 2 — LLM fallback (opt-in): handles prose-only PRs and ambiguous
    custom datasets.

Standalone analysis tool — runs after the filter step.

Output:
  workload_characterization.json   # per-PR entries + aggregate
  workload_characterization.md     # summary tables
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.repo_bench.storage import (
    atomic_write_text,
    data_root,
    raw_dir,
    read_jsonl_lenient,
    write_json,
)

log = logging.getLogger(__name__)

# ── Schemas ──────────────────────────────────────────────────────────────


WorkloadType = Literal["synthetic", "recorded_trace", "combination", "unknown"]
BenchmarkScope = Literal["serving", "kernel", "accuracy"]


class BenchmarkEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    workload_type: WorkloadType
    trace_source: str = ""
    trace_dataset_path: str = ""
    generator_params: dict[str, Any] = Field(default_factory=dict)
    source_command: str = ""
    benchmark_scope: BenchmarkScope = "serving"


class PRCharacterization(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pr_number: int
    title: str = ""
    entries: list[BenchmarkEntry] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"
    extraction_method: Literal["regex", "llm"] = "regex"


# ── Command detection patterns ───────────────────────────────────────────

# Multi-line command capture: tolerates `\<newline>` continuations and
# indented flag lines.
_VLLM_BENCH_CMD_RE = re.compile(
    r"""(?ix)
    (vllm \s+ bench \s+ (?:serve|throughput|latency) \s+
     (?: \\\n | [^\n`] )+
     (?: \n \s* (?: --|-[a-z]) [^\n`]* )*
    )
    """,
    re.VERBOSE,
)

_VLLM_BENCH_SWEEP_RE = re.compile(
    r"""(?ix)
    (vllm \s+ bench \s+ sweep \s+ serve \s+
     (?: \\\n | [^\n`] )+
     (?: \n \s* (?: --|-[a-z]) [^\n`]* )*
    )
    """,
    re.VERBOSE,
)

_BENCHMARK_SERVING_RE = re.compile(
    r"""(?ix)
    ((?:python[3]?\s+)?benchmark_serving\.py \s+
     (?: \\\n | [^\n`] )+
     (?: \n \s* (?: --|-[a-z]) [^\n`]* )*
    )
    """,
    re.VERBOSE,
)

_KERNEL_BENCH_RE = re.compile(
    r"(python[3]?\s+benchmarks/(?:kernels|fused_kernels|attention_benchmarks)/[^\n`]+)",
    re.IGNORECASE,
)

_GENERIC_BENCH_RE = re.compile(
    r"(python[3]?\s+benchmarks/[^\n`]+)",
    re.IGNORECASE,
)

_LM_EVAL_RE = re.compile(
    r"(lm_eval\s+[^\n`]+)",
    re.IGNORECASE,
)

# ── Flag extraction patterns ─────────────────────────────────────────────

_SEP = r"[\s=]+"  # flag-value separator: space(s) or `=`
_DATASET_NAME_RE = re.compile(rf"--dataset[_-]name{_SEP}(\S+)")
_DATASET_PATH_RE = re.compile(rf"--dataset[_-]path{_SEP}(\S+)")
_RANDOM_INPUT_LEN_RE = re.compile(rf"--random[_-]input[_-]len{_SEP}(\d+)")
_RANDOM_OUTPUT_LEN_RE = re.compile(rf"--random[_-]output[_-]len{_SEP}(\d+)")
_NUM_PROMPTS_RE = re.compile(rf"--num[_-]prompts{_SEP}(\S+)")
_REQUEST_RATE_RE = re.compile(rf"--request[_-]rate{_SEP}(\S+)")
_MAX_CONCURRENCY_RE = re.compile(rf"--max[_-]concurrency{_SEP}(\d+)")
_BURSTINESS_RE = re.compile(rf"--burstiness{_SEP}([\d.]+)")
_RANDOM_PREFIX_LEN_RE = re.compile(rf"--random[_-]prefix[_-]len{_SEP}(\d+)")
_RANDOM_RANGE_RATIO_RE = re.compile(rf"--random[_-]range[_-]ratio{_SEP}([\d.]+)")
_INPUT_LEN_RE = re.compile(rf"--input[_-]len{_SEP}(\d+)")
_OUTPUT_LEN_RE = re.compile(rf"--output[_-]len{_SEP}(\d+)")

# Dataset names that indicate synthetic workloads.
_SYNTHETIC_DATASETS = frozenset({
    "random", "random-mm", "random-rerank", "prefix_repetition",
})

# Dataset names that indicate recorded/real traces.
_TRACE_DATASETS = frozenset({
    "sharegpt", "hf", "timed_trace", "timed-trace", "burstgpt",
    "custom", "custom_audio", "custom_image",
    "sonnet", "spec_bench", "speed_bench",
})


# ── Extraction helpers ───────────────────────────────────────────────────


def _normalize_command(cmd: str) -> str:
    """Collapse `\\<newline>` continuations and excess whitespace."""
    s = cmd.replace("\\\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_params(cmd: str) -> dict[str, Any]:
    """Extract recognized benchmark parameters from a command string."""
    params: dict[str, Any] = {}
    normalized = _normalize_command(cmd)

    for name, pattern in [
        ("random_input_len", _RANDOM_INPUT_LEN_RE),
        ("random_output_len", _RANDOM_OUTPUT_LEN_RE),
        ("random_prefix_len", _RANDOM_PREFIX_LEN_RE),
        ("max_concurrency", _MAX_CONCURRENCY_RE),
        ("input_len", _INPUT_LEN_RE),
        ("output_len", _OUTPUT_LEN_RE),
    ]:
        m = pattern.search(normalized)
        if m:
            params[name] = int(m.group(1))

    for name, pattern in [
        ("num_prompts", _NUM_PROMPTS_RE),
        ("request_rate", _REQUEST_RATE_RE),
    ]:
        m = pattern.search(normalized)
        if m:
            val = m.group(1)
            try:
                params[name] = int(val)
            except ValueError:
                try:
                    params[name] = float(val)
                except ValueError:
                    params[name] = val

    for name, pattern in [
        ("burstiness", _BURSTINESS_RE),
        ("random_range_ratio", _RANDOM_RANGE_RATIO_RE),
    ]:
        m = pattern.search(normalized)
        if m:
            params[name] = float(m.group(1))

    return params


def _classify_command(cmd: str) -> tuple[WorkloadType, str, str]:
    """Classify a command's workload type from its flags.

    Returns (workload_type, trace_source, trace_dataset_path).
    """
    normalized = _normalize_command(cmd)

    dataset_name = ""
    m = _DATASET_NAME_RE.search(normalized)
    if m:
        dataset_name = m.group(1).strip("\"'")

    dataset_path = ""
    m = _DATASET_PATH_RE.search(normalized)
    if m:
        dataset_path = m.group(1).strip("\"'")

    if dataset_name in _SYNTHETIC_DATASETS:
        return "synthetic", "", ""
    if dataset_name in _TRACE_DATASETS:
        return "recorded_trace", dataset_name, dataset_path
    if dataset_name:
        return "unknown", dataset_name, dataset_path

    # No --dataset-name: check for random-* params as synthetic signal
    if _RANDOM_INPUT_LEN_RE.search(normalized) or _RANDOM_OUTPUT_LEN_RE.search(normalized):
        return "synthetic", "", ""

    # --input-len / --output-len without --dataset-name defaults to
    # synthetic in vllm bench (the implicit default is `random`).
    if _INPUT_LEN_RE.search(normalized) or _OUTPUT_LEN_RE.search(normalized):
        return "synthetic", "", ""

    # vllm bench commands without --dataset-name default to synthetic
    # (the implicit dataset is `random`). Detect by presence of --model,
    # --num-prompts, --request-rate, or other benchmark flags.
    if re.search(r"vllm\s+bench\s+(?:serve|throughput|latency)", normalized, re.IGNORECASE):
        if re.search(r"--model|--num-prompts|--request-rate|--batch-size|--backend", normalized):
            return "synthetic", "", ""

    return "unknown", "", ""


# ── Main extraction ──────────────────────────────────────────────────────


def _extract_from_body(body: str) -> list[BenchmarkEntry]:
    """Extract all benchmark entries from a PR body using regex."""
    entries: list[BenchmarkEntry] = []
    seen_commands: set[str] = set()

    # 1. vllm bench sweep (check before regular bench to avoid double-match)
    for m in _VLLM_BENCH_SWEEP_RE.finditer(body):
        raw = m.group(1)
        norm = _normalize_command(raw)
        if norm in seen_commands:
            continue
        seen_commands.add(norm)
        wtype, trace_src, trace_path = _classify_command(raw)
        entries.append(BenchmarkEntry(
            tool="vllm bench sweep",
            workload_type=wtype,
            trace_source=trace_src,
            trace_dataset_path=trace_path,
            generator_params=_extract_params(raw),
            source_command=norm,
            benchmark_scope="serving",
        ))

    # 2. vllm bench serve/throughput/latency
    for m in _VLLM_BENCH_CMD_RE.finditer(body):
        raw = m.group(1)
        norm = _normalize_command(raw)
        if norm in seen_commands:
            continue
        # Skip if this overlaps with a sweep command already captured
        if any(norm in s or s in norm for s in seen_commands):
            continue
        # Skip prose references that aren't actual commands (no flags, no model)
        if "--" not in norm and "/" not in norm:
            continue
        seen_commands.add(norm)

        subcommand = "serve"
        sm = re.search(r"vllm\s+bench\s+(serve|throughput|latency)", raw, re.IGNORECASE)
        if sm:
            subcommand = sm.group(1).lower()

        wtype, trace_src, trace_path = _classify_command(raw)
        entries.append(BenchmarkEntry(
            tool=f"vllm bench {subcommand}",
            workload_type=wtype,
            trace_source=trace_src,
            trace_dataset_path=trace_path,
            generator_params=_extract_params(raw),
            source_command=norm,
            benchmark_scope="serving",
        ))

    # 3. benchmark_serving.py
    for m in _BENCHMARK_SERVING_RE.finditer(body):
        raw = m.group(1)
        norm = _normalize_command(raw)
        if norm in seen_commands:
            continue
        seen_commands.add(norm)
        wtype, trace_src, trace_path = _classify_command(raw)
        entries.append(BenchmarkEntry(
            tool="benchmark_serving.py",
            workload_type=wtype,
            trace_source=trace_src,
            trace_dataset_path=trace_path,
            generator_params=_extract_params(raw),
            source_command=norm,
            benchmark_scope="serving",
        ))

    # 4. Kernel micro-benchmarks
    for m in _KERNEL_BENCH_RE.finditer(body):
        raw = m.group(1)
        norm = _normalize_command(raw)
        if norm in seen_commands:
            continue
        seen_commands.add(norm)
        entries.append(BenchmarkEntry(
            tool=norm.split()[1] if len(norm.split()) > 1 else norm,
            workload_type="synthetic",
            generator_params=_extract_params(raw),
            source_command=norm,
            benchmark_scope="kernel",
        ))

    # 5. Generic python benchmarks/ scripts (non-kernel)
    for m in _GENERIC_BENCH_RE.finditer(body):
        raw = m.group(1)
        norm = _normalize_command(raw)
        if norm in seen_commands:
            continue
        # Skip if already captured as kernel
        if "benchmarks/kernels/" in norm:
            continue
        seen_commands.add(norm)
        wtype, trace_src, trace_path = _classify_command(raw)
        if wtype == "unknown" and not trace_src:
            wtype = "synthetic"
        entries.append(BenchmarkEntry(
            tool=norm.split()[1] if len(norm.split()) > 1 else norm,
            workload_type=wtype,
            trace_source=trace_src,
            trace_dataset_path=trace_path,
            generator_params=_extract_params(raw),
            source_command=norm,
            benchmark_scope="serving",
        ))

    # 6. lm_eval
    for m in _LM_EVAL_RE.finditer(body):
        raw = m.group(1)
        norm = _normalize_command(raw)
        if norm in seen_commands:
            continue
        seen_commands.add(norm)
        entries.append(BenchmarkEntry(
            tool="lm_eval",
            workload_type="recorded_trace",
            trace_source="lm_eval",
            source_command=norm,
            benchmark_scope="accuracy",
        ))

    return entries


def characterize_pr(pr: dict) -> PRCharacterization:
    """Characterize a single PR's benchmark workload(s)."""
    body = pr.get("body") or ""
    entries = _extract_from_body(body)

    confidence: Literal["high", "medium", "low"] = "high"
    if not entries:
        confidence = "low"
    elif any(e.workload_type == "unknown" for e in entries):
        confidence = "medium"

    return PRCharacterization(
        pr_number=pr["pr_number"],
        title=pr.get("title", ""),
        entries=entries,
        confidence=confidence,
        extraction_method="regex",
    )


# ── Aggregation ──────────────────────────────────────────────────────────


def _pr_level_type(entries: list[BenchmarkEntry], include_kernel: bool, include_accuracy: bool) -> WorkloadType | None:
    """Determine overall workload type for a PR from its entries.

    Returns None if all entries are excluded by scope filters.
    """
    types: set[WorkloadType] = set()
    for e in entries:
        if e.benchmark_scope == "kernel" and not include_kernel:
            continue
        if e.benchmark_scope == "accuracy" and not include_accuracy:
            continue
        types.add(e.workload_type)

    if not types:
        return None
    if "synthetic" in types and "recorded_trace" in types:
        return "combination"
    if "recorded_trace" in types:
        return "recorded_trace"
    if "synthetic" in types:
        return "synthetic"
    if "combination" in types:
        return "combination"
    return "unknown"


@dataclass(frozen=True)
class AggregateStats:
    n_total: int
    n_with_benchmarks: int
    type_counts: dict[str, int]
    trace_sources: dict[str, list[int]]
    tool_counts: dict[str, int]
    top_synthetic_configs: list[dict[str, Any]]
    n_synthetic_default_params: int
    n_synthetic_with_params: int
    n_kernel_excluded: int
    n_accuracy_excluded: int


def aggregate_characterizations(
    results: list[PRCharacterization],
    *,
    include_kernel: bool = False,
    include_accuracy: bool = False,
) -> AggregateStats:
    """Aggregate per-PR characterizations into summary statistics."""
    type_counter: Counter[str] = Counter()
    trace_sources: dict[str, list[int]] = defaultdict(list)
    tool_counter: Counter[str] = Counter()
    synthetic_configs: list[tuple[dict[str, Any], int]] = []
    param_counter: Counter[str] = Counter()
    n_with_benchmarks = 0
    n_synthetic_default_params = 0
    n_synthetic_with_params = 0
    n_kernel_excluded = 0
    n_accuracy_excluded = 0

    for pr in results:
        scoped_entries = []
        for e in pr.entries:
            if e.benchmark_scope == "kernel" and not include_kernel:
                n_kernel_excluded += 1
                continue
            if e.benchmark_scope == "accuracy" and not include_accuracy:
                n_accuracy_excluded += 1
                continue
            scoped_entries.append(e)

        if not scoped_entries:
            continue
        n_with_benchmarks += 1

        pr_type = _pr_level_type(pr.entries, include_kernel, include_accuracy)
        if pr_type:
            type_counter[pr_type] += 1

        for e in scoped_entries:
            tool_counter[e.tool] += 1
            if e.trace_source:
                source_key = e.trace_source
                if e.trace_dataset_path:
                    source_key = f"{e.trace_source}/{e.trace_dataset_path}"
                trace_sources[source_key].append(pr.pr_number)
            if e.workload_type == "synthetic":
                if e.generator_params:
                    n_synthetic_with_params += 1
                    param_key = json.dumps(e.generator_params, sort_keys=True)
                    param_counter[param_key] += 1
                else:
                    n_synthetic_default_params += 1

    # Build top synthetic configs
    top_configs: list[dict[str, Any]] = []
    for param_json, count in param_counter.most_common(15):
        params = json.loads(param_json)
        top_configs.append({"params": params, "count": count})

    return AggregateStats(
        n_total=len(results),
        n_with_benchmarks=n_with_benchmarks,
        type_counts=dict(type_counter.most_common()),
        trace_sources=dict(trace_sources),
        tool_counts=dict(tool_counter.most_common()),
        top_synthetic_configs=top_configs,
        n_synthetic_default_params=n_synthetic_default_params,
        n_synthetic_with_params=n_synthetic_with_params,
        n_kernel_excluded=n_kernel_excluded,
        n_accuracy_excluded=n_accuracy_excluded,
    )


# ── Markdown rendering ───────────────────────────────────────────────────


def render_summary_md(
    stats: AggregateStats,
    results: list[PRCharacterization],
    *,
    include_kernel: bool = False,
    include_accuracy: bool = False,
) -> str:
    lines: list[str] = []
    lines.append(f"# Workload Characterization — {stats.n_total} PRs analyzed\n")
    lines.append(f"**{stats.n_with_benchmarks}** PRs contain recognizable benchmark commands.\n")

    # Summary
    lines.append("## Summary\n")
    for wtype in ("synthetic", "recorded_trace", "combination", "unknown"):
        count = stats.type_counts.get(wtype, 0)
        pct = 100 * count / max(1, stats.n_with_benchmarks)
        lines.append(f"- **{wtype}**: {count} PRs ({pct:.1f}%)")
    if stats.n_kernel_excluded:
        lines.append(f"- _Kernel micro-benchmarks: {stats.n_kernel_excluded} entries excluded_")
    if stats.n_accuracy_excluded:
        lines.append(f"- _Accuracy evals (lm_eval): {stats.n_accuracy_excluded} entries excluded_")
    lines.append("")

    # Trace sources
    if stats.trace_sources:
        lines.append("## Trace sources (recorded + combination)\n")
        lines.append("| Source | Count | Example PRs |")
        lines.append("|--------|------:|-------------|")
        for source, pr_nums in sorted(stats.trace_sources.items(), key=lambda kv: -len(kv[1])):
            unique_prs = sorted(set(pr_nums))
            examples = ", ".join(f"#{n}" for n in unique_prs[:5])
            if len(unique_prs) > 5:
                examples += f" +{len(unique_prs) - 5} more"
            lines.append(f"| {source} | {len(unique_prs)} | {examples} |")
        lines.append("")

    # Synthetic configs
    lines.append("## Synthetic workload parameters\n")
    if stats.n_synthetic_default_params > 0:
        lines.append(
            f"**{stats.n_synthetic_default_params}** synthetic entries use "
            f"default parameters (no explicit input/output length, concurrency, "
            f"or rate specified in the command).\n"
        )
    if stats.top_synthetic_configs:
        lines.append(
            f"**{stats.n_synthetic_with_params}** synthetic entries specify "
            f"explicit parameters. Top configurations:\n"
        )
        lines.append("| Input len | Output len | Num prompts | Request rate | Concurrency | Count |")
        lines.append("|----------:|----------:|------------:|:-------------|:------------|------:|")
        for cfg in stats.top_synthetic_configs[:10]:
            p = cfg["params"]
            il = p.get("random_input_len", p.get("input_len", "—"))
            ol = p.get("random_output_len", p.get("output_len", "—"))
            np_ = p.get("num_prompts", "—")
            rr = p.get("request_rate", "—")
            mc = p.get("max_concurrency", "—")
            lines.append(f"| {il} | {ol} | {np_} | {rr} | {mc} | {cfg['count']} |")
        lines.append("")

    # Tools used
    if stats.tool_counts:
        lines.append("## Benchmark tools used\n")
        lines.append("| Tool | Count |")
        lines.append("|------|------:|")
        for tool, count in sorted(stats.tool_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {tool} | {count} |")
        lines.append("")

    return "\n".join(lines)


# ── CSV rendering ────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "pr_number", "title", "workload_type", "benchmark_scope", "tool",
    "trace_source", "trace_dataset_path",
    "random_input_len", "random_output_len", "input_len", "output_len",
    "num_prompts", "request_rate", "max_concurrency", "burstiness",
    "confidence", "extraction_method", "source_command",
]


def render_csv(
    results: list[PRCharacterization],
    *,
    include_kernel: bool = False,
    include_accuracy: bool = False,
) -> str:
    """Render one row per benchmark entry across all PRs."""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    for pr in results:
        for e in pr.entries:
            if e.benchmark_scope == "kernel" and not include_kernel:
                continue
            if e.benchmark_scope == "accuracy" and not include_accuracy:
                continue
            row: dict[str, Any] = {
                "pr_number": pr.pr_number,
                "title": pr.title,
                "workload_type": e.workload_type,
                "benchmark_scope": e.benchmark_scope,
                "tool": e.tool,
                "trace_source": e.trace_source,
                "trace_dataset_path": e.trace_dataset_path,
                "confidence": pr.confidence,
                "extraction_method": pr.extraction_method,
                "source_command": e.source_command,
            }
            # Flatten generator_params into columns
            for key in ("random_input_len", "random_output_len", "input_len",
                        "output_len", "num_prompts", "request_rate",
                        "max_concurrency", "burstiness"):
                row[key] = e.generator_params.get(key, "")
            writer.writerow(row)

    return buf.getvalue()


# ── Public API ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CharacterizationHandle:
    json_path: Path
    md_path: Path
    csv_path: Path
    n_total: int
    n_with_benchmarks: int
    type_counts: dict[str, int]


def characterize(
    *,
    window_id: str,
    view_path: Path,
    out_dir: Path | None = None,
    run_dir: Path | None = None,
    data_root_override: Path | None = None,
    include_kernel: bool = False,
    include_accuracy: bool = False,
) -> CharacterizationHandle:
    """Run workload characterization over a filtered view.

    Reads:
      `view_path`                          — the run's filtered PRs
      `<data_root>/raw/<window>/prs.jsonl` — raw PR bodies

    Writes:
      `<out_dir>/workload_characterization.json`
      `<out_dir>/workload_characterization.md`
    """
    cache_root = data_root_override or data_root()
    raw_path = raw_dir(window_id, root=cache_root) / "prs.jsonl"
    output_dir = out_dir or run_dir
    if output_dir is None:
        raise ValueError("Either out_dir or run_dir must be provided")

    if not view_path.exists():
        raise FileNotFoundError(f"view not found: {view_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw scrape not found: {raw_path}")

    # Load view PR numbers
    view_pr_numbers: set[int] = set()
    for line in view_path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            view_pr_numbers.add(int(json.loads(line)["pr_number"]))

    # Load raw PR bodies for the view
    raw_by_n: dict[int, dict] = {}
    for d in read_jsonl_lenient(raw_path):
        pr_n = d.get("pr_number")
        if pr_n in view_pr_numbers:
            raw_by_n[pr_n] = d

    log.info(
        "characterize: %d view PRs, %d raw bodies loaded",
        len(view_pr_numbers), len(raw_by_n),
    )

    # Extract per-PR
    results: list[PRCharacterization] = []
    for pr_n in sorted(raw_by_n):
        results.append(characterize_pr(raw_by_n[pr_n]))

    # Aggregate
    stats = aggregate_characterizations(
        results,
        include_kernel=include_kernel,
        include_accuracy=include_accuracy,
    )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    full_output = {
        "window_id": window_id,
        "n_total": stats.n_total,
        "n_with_benchmarks": stats.n_with_benchmarks,
        "type_counts": stats.type_counts,
        "trace_sources": {k: sorted(set(v)) for k, v in stats.trace_sources.items()},
        "tool_counts": stats.tool_counts,
        "top_synthetic_configs": stats.top_synthetic_configs,
        "include_kernel": include_kernel,
        "include_accuracy": include_accuracy,
        "per_pr": [r.model_dump() for r in results],
    }
    json_path = output_dir / "workload_characterization.json"
    write_json(json_path, full_output)

    md_content = render_summary_md(
        stats, results,
        include_kernel=include_kernel,
        include_accuracy=include_accuracy,
    )
    md_path = output_dir / "workload_characterization.md"
    atomic_write_text(md_path, md_content)

    csv_content = render_csv(
        results,
        include_kernel=include_kernel,
        include_accuracy=include_accuracy,
    )
    csv_path = output_dir / "workload_characterization.csv"
    atomic_write_text(csv_path, csv_content)

    log.info(
        "characterize: %d PRs analyzed, %d with benchmarks → %s",
        stats.n_total, stats.n_with_benchmarks, json_path,
    )

    return CharacterizationHandle(
        json_path=json_path,
        md_path=md_path,
        csv_path=csv_path,
        n_total=stats.n_total,
        n_with_benchmarks=stats.n_with_benchmarks,
        type_counts=stats.type_counts,
    )


__all__ = [
    "BenchmarkEntry",
    "CharacterizationHandle",
    "PRCharacterization",
    "aggregate_characterizations",
    "characterize",
    "characterize_pr",
]
