"""Analyze the filtered view's PRs to extract a workload portfolio.

For each PR, scan title + body + diff for structured workload
signals: models, parallelism flags, hardware, benchmark commands,
quantization, hot-path category. Aggregate to surface which workload
configurations the perf community actively measures wins against.

Output (under `run_dir`):
  workload_analysis.json   # full per-PR extraction
  workload_summary.md      # human-readable top-N tables

Read-only / regex-based — no LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from spotlights_engine.repo_bench.storage import (
    data_root,
    raw_dir,
)

log = logging.getLogger(__name__)

# ── Signal patterns ───────────────────────────────────────────────────

# Model families seen across vLLM PRs. Order matters: longer / more
# specific names come first so partial matches don't shadow them.
MODEL_PATTERNS = [
    (r"\b(Qwen3-30B-A3B(?:-(?:Thinking|Instruct))?(?:-2507)?(?:-FP8|-fp8)?)\b", "qwen3-30b-a3b"),
    (r"\b(Qwen3-(?:235B|72B|32B|14B|8B|4B|1\.7B|0\.6B)(?:-FP8)?(?:-Instruct)?)\b", "qwen3-other"),
    (r"\b(DeepSeek-V3(?:\.2)?(?:-Lite)?(?:-FP8)?)\b", "deepseek-v3"),
    (r"\b(DeepSeek-R1(?:-(?:Distill-)?(?:Qwen|Llama)(?:-\d+B)?)?)\b", "deepseek-r1"),
    (r"\b(LongCat[- ]?Flash(?:-Chat)?)\b", "longcat-flash"),
    (r"\b(MiniMax-M\d+)\b", "minimax-m"),
    (r"\b(Mistral[- ]?7B(?:-Instruct(?:-v\d+\.\d+)?)?)\b", "mistral-7b"),
    (r"\b(Mixtral-8x(?:7B|22B))\b", "mixtral-8x"),
    (r"\b(Llama-?3(?:\.\d+)?-(?:70B|8B|3B|1B)(?:-Instruct)?)\b", "llama-3"),
    (r"\b(Llama-?2-\d+B)\b", "llama-2"),
    (r"\b(gpt-oss(?:-20b|-120b)?)\b", "gpt-oss"),
    (r"\b(Gemma-?\d+)\b", "gemma"),
    (r"\b(Phi-?\d+)\b", "phi"),
    (r"\b(GLM-?4(?:\.\dV)?)\b", "glm-4"),
    (r"\b(Whisper(?:-large(?:-v\d)?)?)\b", "whisper"),
    (r"\b(Medusa)\b", "medusa"),
    (r"\b(Eagle\d?)\b", "eagle"),
]

# Parallelism / config flags from `vllm serve` commands.
PARALLELISM_FLAGS = [
    (r"\b-tp[ =](\d+)\b|--tensor-parallel-size[ =](\d+)\b|tensor_parallel_size=(\d+)\b", "tp"),
    (r"\b-pp[ =](\d+)\b|--pipeline-parallel-size[ =](\d+)\b|pipeline_parallel_size=(\d+)\b", "pp"),
    (r"\b-dp[ =](\d+)\b|--data-parallel-size[ =](\d+)\b|data_parallel_size=(\d+)\b", "dp"),
    (r"--max-num-seqs[ =](\d+)\b|max_num_seqs=(\d+)\b", "max_num_seqs"),
]

# Boolean / categorical features.
FEATURE_PATTERNS = [
    (r"\benable-expert-parallel\b|enable_expert_parallel|--ep\b", "expert_parallel"),
    (r"\benable-eplb\b|enable_eplb|EPLB", "eplb"),
    (r"\bspec(?:ulative)?[- _]?decod", "spec_decode"),
    (r"\basync[- _]?schedul", "async_scheduling"),
    (r"\bchunked[- _]?prefill\b", "chunked_prefill"),
    (r"\bprefix[- _]?cach", "prefix_cache"),
    (r"\bcuda[- _]?graph\b", "cuda_graph"),
    (r"\btorch\.compile\b", "torch_compile"),
    (r"\bFP8|fp8\b", "fp8"),
    (r"\b(?:INT4|int4|W4A16|w4a16)\b", "int4_quant"),
    (r"\b(?:INT8|int8|W8A8|w8a8)\b", "int8_quant"),
    (r"\bBF16|bf16\b", "bf16"),
    (r"\bFP16|fp16\b", "fp16"),
    (r"\bMoE\b|fused.?moe\b|mixture.?of.?experts", "moe"),
    (r"\bMLA\b|multi.?head.?latent", "mla"),
    (r"\bRoPE|rotary.?embed", "rotary"),
    (r"\bRMSNorm|rms_norm", "rmsnorm"),
    (r"\battention\b", "attention"),
    (r"\bscheduler\b|preempt", "scheduler"),
    (r"\bKV[- ]?cache\b|kv_cache", "kv_cache"),
    (r"\boffload", "offload"),
]

# Hardware mentions.
HARDWARE_PATTERNS = [
    (r"\bH100\b|H200\b", "nvidia-hopper"),
    (r"\bB200\b|B100\b|GB200\b", "nvidia-blackwell"),
    (r"\bA100\b|A800\b", "nvidia-ampere"),
    (r"\bL40S?\b|L4\b", "nvidia-l4"),
    (r"\b(?:MI300X?|MI250X?|MI355X?)\b|\bROCm\b|\bAITER\b", "amd"),
    (r"\bTPU\b|XLA", "tpu"),
    (r"\bSM(?:90|100|120)\b", "sm_arch"),
    (r"\bGFX(?:11|11\d\d)\b", "gfx_arch"),
]

# Benchmark commands embedded in PR bodies.
VLLM_SERVE_RE = re.compile(
    r"vllm\s+serve\s+([\w/.\-]+)([^\n`]*)",
    re.IGNORECASE,
)
VLLM_BENCH_RE = re.compile(
    r"vllm\s+bench\s+(serve|throughput|latency)\s+([^\n`]*)",
    re.IGNORECASE,
)
LM_EVAL_RE = re.compile(r"lm_eval\s+([^\n`]*)", re.IGNORECASE)


@dataclass
class PRSignals:
    pr_number: int
    title: str
    models: set[str] = field(default_factory=set)
    model_canonical: set[str] = field(default_factory=set)
    parallelism: dict[str, int] = field(default_factory=dict)
    features: set[str] = field(default_factory=set)
    hardware: set[str] = field(default_factory=set)
    serve_cmds: list[str] = field(default_factory=list)
    bench_cmds: list[str] = field(default_factory=list)
    has_benchmark_table: bool = False
    files_touched_categories: set[str] = field(default_factory=set)


# ── File-path → component category ───────────────────────────────────


FILE_CATEGORY_PATTERNS = [
    (r"vllm/v1/spec_decode/|spec_decode\.py", "spec_decode"),
    (r"vllm/model_executor/layers/fused_moe/|fused_moe\.py", "moe_kernel"),
    (r"vllm/distributed/eplb/", "eplb"),
    (r"vllm/distributed/", "distributed_comm"),
    (r"vllm/v1/core/sched/|scheduler\.py", "scheduler"),
    (r"vllm/v1/attention/|vllm/attention/", "attention"),
    (r"vllm/model_executor/layers/quantization/", "quantization"),
    (r"vllm/v1/kv_offload/|kv_cache_manager", "kv_cache"),
    (r"vllm/compilation/|torch.compile", "torch_compile"),
    (r"vllm/model_executor/layers/(?:layernorm|rotary)", "norm_rope"),
    (r"vllm/v1/engine/", "engine_v1"),
    (r"vllm/lora/|punica", "lora"),
    (r"csrc/", "cuda_kernels"),
    (r"vllm/multimodal/|vllm/inputs/", "multimodal"),
    (r"vllm/platforms/", "platform"),
]


def _categorize_file(path: str) -> str | None:
    for pat, name in FILE_CATEGORY_PATTERNS:
        if re.search(pat, path, re.IGNORECASE):
            return name
    return None


# ── Extraction ────────────────────────────────────────────────────────


def _extract_first_int(match: re.Match) -> int | None:
    for g in match.groups():
        if g and g.isdigit():
            return int(g)
    return None


def extract_signals(pr: dict, diff_text: str | None) -> PRSignals:
    s = PRSignals(pr_number=pr["pr_number"], title=pr.get("title", "") or "")
    blob = (pr.get("title", "") or "") + "\n" + (pr.get("body", "") or "")

    for pat, canonical in MODEL_PATTERNS:
        for m in re.finditer(pat, blob):
            s.models.add(m.group(0))
            s.model_canonical.add(canonical)

    for pat, name in PARALLELISM_FLAGS:
        for m in re.finditer(pat, blob):
            n = _extract_first_int(m)
            if n is not None:
                # Keep the largest seen per axis
                s.parallelism[name] = max(s.parallelism.get(name, 0), n)

    for pat, name in FEATURE_PATTERNS:
        if re.search(pat, blob, re.IGNORECASE):
            s.features.add(name)

    for pat, name in HARDWARE_PATTERNS:
        if re.search(pat, blob):
            s.hardware.add(name)

    for m in VLLM_SERVE_RE.finditer(blob):
        cmd = (m.group(0) or "")[:300]
        s.serve_cmds.append(cmd.strip())

    for m in VLLM_BENCH_RE.finditer(blob):
        cmd = (m.group(0) or "")[:300]
        s.bench_cmds.append(cmd.strip())

    # Has a "Serving Benchmark Result" table or similar?
    if re.search(r"Serving Benchmark Result|Benchmark duration|Output token throughput|Request throughput",
                 blob, re.IGNORECASE):
        s.has_benchmark_table = True

    # File-touch categories from the diff
    if diff_text:
        for path in re.findall(r"^diff --git a/(\S+)", diff_text, re.MULTILINE):
            cat = _categorize_file(path)
            if cat:
                s.files_touched_categories.add(cat)

    return s


# ── Aggregation ───────────────────────────────────────────────────────


def aggregate(signals: list[PRSignals]) -> dict:
    n = len(signals)
    model_counts = Counter()
    feature_counts = Counter()
    hardware_counts = Counter()
    file_cat_counts = Counter()
    parallelism_axes = Counter()
    benchmark_pr_count = 0
    serve_cmd_pr_count = 0

    # Co-occurrence: how often each model appears with each feature
    model_x_feature: dict[tuple[str, str], int] = defaultdict(int)

    for s in signals:
        for m in s.model_canonical:
            model_counts[m] += 1
        for f in s.features:
            feature_counts[f] += 1
            for m in s.model_canonical:
                model_x_feature[(m, f)] += 1
        for h in s.hardware:
            hardware_counts[h] += 1
        for c in s.files_touched_categories:
            file_cat_counts[c] += 1
        for axis in s.parallelism:
            parallelism_axes[axis] += 1
        if s.has_benchmark_table:
            benchmark_pr_count += 1
        if s.serve_cmds:
            serve_cmd_pr_count += 1

    return {
        "n_prs": n,
        "n_with_benchmark_table": benchmark_pr_count,
        "n_with_serve_cmd": serve_cmd_pr_count,
        "model_counts": dict(model_counts.most_common()),
        "feature_counts": dict(feature_counts.most_common()),
        "hardware_counts": dict(hardware_counts.most_common()),
        "file_category_counts": dict(file_cat_counts.most_common()),
        "parallelism_axes": dict(parallelism_axes.most_common()),
        "model_x_feature": {
            f"{m}|{f}": v
            for (m, f), v in sorted(
                model_x_feature.items(), key=lambda kv: -kv[1]
            )[:50]
        },
    }


# ── Workload portfolio synthesis ─────────────────────────────────────


def synthesize_workload_portfolio(signals: list[PRSignals], agg: dict) -> list[dict]:
    """Greedy set-cover over PRs.

    A workload is anchored by (model, feature). A PR is "covered" by
    that workload only if the PR mentions BOTH the model AND the
    feature (or touches the corresponding file category). This is
    stricter than the original OR-match — covered PRs are ones whose
    perf claims are about that exact model+feature combo.

    Tied candidates: prefer the one whose anchor is itself the most
    common in agg["model_counts"]; secondary tiebreak alphabetical.
    """
    # Build the strict candidate pool: (model, feature) -> covered PR set
    candidates: dict[tuple[str, str], set[int]] = {}
    for s in signals:
        if not s.model_canonical:
            continue
        for model in s.model_canonical:
            for feat in s.features:
                candidates.setdefault((model, feat), set()).add(s.pr_number)

    # Filter: keep only candidates whose intersection has >=3 PRs
    # (single PRs are noise; same goes for two-PR pairs).
    pruned = {k: v for k, v in candidates.items() if len(v) >= 3}

    # Greedy cover
    selected: list[dict] = []
    covered_so_far: set[int] = set()
    n = len(signals)
    model_rank = {m: i for i, m in enumerate(agg["model_counts"])}
    while pruned and len(selected) < 8:
        # Score each remaining candidate by uncovered yield
        ranked = sorted(
            pruned.items(),
            key=lambda kv: (
                -len(kv[1] - covered_so_far),
                model_rank.get(kv[0][0], 999),
                kv[0][0],
                kv[0][1],
            ),
        )
        (model, feat), covered = ranked[0]
        new_cov = covered - covered_so_far
        if not new_cov:
            break
        selected.append({
            "model": model,
            "feature": feat,
            "n_prs_total_intersection": len(covered),
            "n_new_covered": len(new_cov),
            "cumulative_share": (len(covered_so_far) + len(new_cov)) / max(1, n),
            "example_pr_numbers": sorted(new_cov)[:5],
        })
        covered_so_far |= new_cov
        pruned.pop((model, feat))
    return selected


# ── Markdown render ───────────────────────────────────────────────────


def render_summary(agg: dict, portfolio: list[dict], n: int) -> str:
    lines: list[str] = []
    lines.append(f"# Workload analysis — {n} PRs in view\n")
    lines.append(
        f"- **{agg['n_with_serve_cmd']} of {n}** PRs include an explicit "
        f"`vllm serve` command in body (~ {100*agg['n_with_serve_cmd']/n:.0f}%)\n"
    )
    lines.append(
        f"- **{agg['n_with_benchmark_table']} of {n}** PRs include a "
        f"benchmark result table (~ {100*agg['n_with_benchmark_table']/n:.0f}%)\n\n"
    )

    def _table(header: str, items: dict, top: int = 15) -> None:
        lines.append(f"## {header}\n\n")
        lines.append(f"| Item | Count | % of {n} |\n|---|---:|---:|\n")
        for k, v in list(items.items())[:top]:
            lines.append(f"| `{k}` | {v} | {100*v/n:.1f}% |\n")
        lines.append("\n")

    _table("Top models", agg["model_counts"])
    _table("Top features", agg["feature_counts"])
    _table("Top hardware mentions", agg["hardware_counts"])
    _table("Top file categories touched", agg["file_category_counts"])
    _table("Parallelism axes mentioned", agg["parallelism_axes"], top=10)

    lines.append("## Top model × feature co-occurrences\n\n")
    lines.append("| Model + feature | PRs |\n|---|---:|\n")
    for k, v in list(agg["model_x_feature"].items())[:25]:
        lines.append(f"| `{k}` | {v} |\n")
    lines.append("\n")

    lines.append("## Suggested workload portfolio (greedy AND-cover)\n\n")
    lines.append(
        "Each entry is (model, feature). A PR is counted as covered "
        "ONLY if it mentions BOTH the model AND the feature. Greedy "
        "cover picks the highest-yield (model, feature) combo, marks "
        "those PRs covered, and repeats. Pairs covering fewer than 3 "
        "PRs are pruned as noise.\n\n"
    )
    lines.append("| # | Model | Feature | Total PRs at intersection | New covered | Cumulative share | Example PRs |\n")
    lines.append("|---:|---|---|---:|---:|---:|---|\n")
    for i, e in enumerate(portfolio, 1):
        examples = ", ".join(f"#{n}" for n in e.get("example_pr_numbers", []))
        lines.append(
            f"| {i} | `{e['model']}` | `{e['feature']}` | "
            f"{e['n_prs_total_intersection']} | {e['n_new_covered']} | "
            f"{100*e['cumulative_share']:.1f}% | {examples} |\n"
        )
    lines.append("\n")

    return "".join(lines)


# ── Public API ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkloadAnalysisHandle:
    json_path: Path
    md_path: Path
    n_prs: int
    portfolio_size: int


def analyze(
    *,
    window_id: str,
    view_path: Path,
    run_dir: Path,
    data_root_override: Path | None = None,
) -> WorkloadAnalysisHandle:
    """Run workload analysis over a view; emit JSON + MD into `run_dir`.

    Reads:
      `view_path` (the run's view file, `<run_dir>/view/prs.jsonl`)
      `<data_root>/raw/<window>/prs.jsonl`
      `<data_root>/raw/<window>/diffs/<pr>.diff`

    Writes:
      `<run_dir>/workload_analysis.json`
      `<run_dir>/workload_summary.md`
    """
    cache_root = data_root_override or data_root()
    raw_path = raw_dir(window_id, root=cache_root) / "prs.jsonl"
    diffs_dir = raw_dir(window_id, root=cache_root) / "diffs"

    if not view_path.exists():
        raise FileNotFoundError(f"view not found: {view_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw scrape not found: {raw_path}")

    view_pr_numbers = {
        int(json.loads(l)["pr_number"])
        for l in view_path.open(encoding="utf-8")
        if l.strip()
    }

    raw_by_n: dict[int, dict] = {}
    for line in raw_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("pr_number") in view_pr_numbers:
            raw_by_n[d["pr_number"]] = d

    signals: list[PRSignals] = []
    for n in sorted(raw_by_n):
        diff_path = diffs_dir / f"{n}.diff"
        diff_text = (
            diff_path.read_text(encoding="utf-8", errors="replace")
            if diff_path.exists()
            else None
        )
        signals.append(extract_signals(raw_by_n[n], diff_text))

    agg = aggregate(signals)
    portfolio = synthesize_workload_portfolio(signals, agg)

    full = {
        "n_prs": len(signals),
        "aggregate": agg,
        "portfolio": portfolio,
        "per_pr": [
            {
                "pr_number": s.pr_number,
                "title": s.title,
                "models": sorted(s.models),
                "model_canonical": sorted(s.model_canonical),
                "parallelism": s.parallelism,
                "features": sorted(s.features),
                "hardware": sorted(s.hardware),
                "serve_cmds": s.serve_cmds[:3],
                "bench_cmds": s.bench_cmds[:3],
                "has_benchmark_table": s.has_benchmark_table,
                "files_touched_categories": sorted(s.files_touched_categories),
            }
            for s in signals
        ],
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "workload_analysis.json"
    md_path = run_dir / "workload_summary.md"
    json_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    md_path.write_text(render_summary(agg, portfolio, len(signals)), encoding="utf-8")

    log.info(
        "workloads: analyzed %d PRs, %d-entry portfolio → %s",
        len(signals), len(portfolio), json_path,
    )

    return WorkloadAnalysisHandle(
        json_path=json_path,
        md_path=md_path,
        n_prs=len(signals),
        portfolio_size=len(portfolio),
    )


__all__ = [
    "WorkloadAnalysisHandle",
    "analyze",
    "extract_signals",
    "aggregate",
    "synthesize_workload_portfolio",
]
