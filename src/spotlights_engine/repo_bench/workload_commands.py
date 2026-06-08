"""LLM workload-command extractor.

Reads each filtered PR's body + diff, asks Sonnet for the structured
benchmark configuration (model, serve command, bench command, hardware
tier). Then clusters extractions and emits a top-N runnable workload
portfolio for the bench-spec.

Repo-agnostic by design: the prompt asks the model to extract whatever
benchmark commands are documented, not specifically `vllm serve`.

Pipeline shape:
  Per PR (parallelizable, cacheable per-PR):
    LLM extracts {model, serve_cmd, bench_cmd, hardware, citation}
  Hallucination guard:
    Every command must be a substring of PR body (whitespace-tolerant)
  Cluster:
    Group by (canonical model, top-N features) using deterministic
    string match — no LLM. Pick most-cited skeleton per cluster.
  Render:
    Top-N clusters become the workload portfolio in the bench-spec.

Output: <run_dir>/workload_commands.json with full per-PR extractions
+ clustered portfolio. The portfolio gets inlined into the bench-spec
MD as a separate section.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spotlights_engine.agent_proposals.claude_exec import (
    CandidateAgentRunResult,
    run_candidate_claude as _run_claude_default,
)
from spotlights_engine.repo_bench.schemas import RawPR
from spotlights_engine.repo_bench.storage import (
    append_jsonl,
    atomic_write_text,
    data_root,
    raw_dir,
    read_jsonl_lenient,
    write_json,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
DEFAULT_WALLCLOCK_S = 240
DEFAULT_MAX_TURNS = 10
DEFAULT_PORTFOLIO_TOP_N = 5


# ── Schemas ───────────────────────────────────────────────────────────


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    serve_command: str = Field(min_length=1)
    bench_command: str = Field(default="", description="May be empty if PR didn't document one.")
    hardware: str = Field(default="", description="GPU count + type if mentioned.")
    citation: str = Field(default="", description="One-line excerpt from PR body where command appeared.")


class PRWorkloadExtraction(BaseModel):
    """LLM output for one PR."""
    model_config = ConfigDict(extra="forbid")
    is_perf_relevant: bool
    tested_against: list[WorkloadConfig] = Field(default_factory=list)
    perf_claim: str = Field(default="", max_length=200)
    perf_category: str = Field(default="", max_length=80)


class _RunnerFn(Protocol):
    def __call__(
        self,
        *,
        candidate_id: str,
        prompt: str,
        schema_text: str,
        repo_path: Path,
        max_turns: int,
        wallclock_s: int,
    ) -> CandidateAgentRunResult: ...


# ── Prompt ────────────────────────────────────────────────────────────


PROMPT_TEMPLATE = """\
You are extracting structured workload information from a merged
performance PR.

# The PR

**Number:** #{pr_number}
**Title:** {title}

**Body:**
```
{body}
```

# Your task

Extract the benchmark configuration the author used to measure their
performance claim. Return strict JSON matching this schema:

```json
{{
  "is_perf_relevant": <true if PR is a perf optimization, false otherwise>,
  "tested_against": [
    {{
      "model": "<model name as appearing in the command, e.g. Qwen/Qwen3-30B-A3B-FP8>",
      "serve_command": "<the exact `vllm serve ...` or equivalent server command from the body>",
      "bench_command": "<the benchmark-driver command, or empty if not documented>",
      "hardware": "<e.g. '8x H100' if author mentions it, else empty>",
      "citation": "<one-line excerpt from PR body where this config appears>"
    }}
  ],
  "perf_claim": "<the headline number, e.g. '2.9% E2E throughput improvement'>",
  "perf_category": "<2-4 word tag, e.g. 'moe-expert-parallel', 'host-sync-fix', 'kernel-fusion'>"
}}
```

# Important rules

- **`serve_command` and `bench_command` must be VERBATIM substrings of
  the PR body.** Do not paraphrase or normalize. If you can't find an
  exact runnable command, leave the entry out.
- **Multiple configs per PR are valid** — if the author tested across
  models or hardware tiers, return one entry per config.
- **Empty `tested_against` is valid** — many PRs (kernel-level
  optimizations, refactors) don't include a runnable command. Just
  set `is_perf_relevant` and the tag.
- **Don't invent** models, GPU counts, or commands. If unclear, omit.
"""


def prompt_sha256() -> str:
    payload = {
        "template": PROMPT_TEMPLATE,
        "schema": PRWorkloadExtraction.model_json_schema(),
        "version": PROMPT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


# ── Hallucination guard ───────────────────────────────────────────────


def _normalize_ws(s: str) -> str:
    """Collapse whitespace + line continuations for substring check."""
    return re.sub(r"\s+", " ", s.replace("\\\n", " ")).strip()


def _command_in_body(cmd: str, body: str) -> bool:
    if not cmd:
        return True
    return _normalize_ws(cmd) in _normalize_ws(body)


def _validate_extraction(
    extraction: PRWorkloadExtraction, body: str
) -> tuple[PRWorkloadExtraction, list[str]]:
    """Drop tested_against entries whose commands aren't in the body.
    Returns (cleaned_extraction, dropped_reasons)."""
    kept: list[WorkloadConfig] = []
    dropped: list[str] = []
    for cfg in extraction.tested_against:
        if not _command_in_body(cfg.serve_command, body):
            dropped.append(f"serve_command not in body: {cfg.serve_command[:60]}")
            continue
        if cfg.bench_command and not _command_in_body(cfg.bench_command, body):
            dropped.append(f"bench_command not in body: {cfg.bench_command[:60]}")
            continue
        kept.append(cfg)
    cleaned = extraction.model_copy(update={"tested_against": kept})
    return cleaned, dropped


# ── Clustering ────────────────────────────────────────────────────────


_MODEL_FAMILY_PATTERNS = [
    (r"Qwen[23]-?\d+B-A\d+B", "qwen3-moe"),
    (r"Qwen[23]-?\d+B", "qwen-dense"),
    (r"DeepSeek-V\d", "deepseek-v"),
    (r"DeepSeek-R\d", "deepseek-r"),
    (r"Llama-?\d+", "llama"),
    (r"Mixtral-\d+x\d+B", "mixtral"),
    (r"Mistral-?\d+B", "mistral"),
    (r"gpt-oss", "gpt-oss"),
    (r"GLM-?\d+", "glm"),
    (r"MiniMax-M\d+", "minimax"),
]


def _model_family(model: str) -> str:
    for pat, fam in _MODEL_FAMILY_PATTERNS:
        if re.search(pat, model, re.IGNORECASE):
            return fam
    # Fallback: first slash-component lowercased
    return model.split("/")[-1].lower().split("-")[0] or "other"


_FEATURE_PATTERNS = [
    (r"\benable-eplb\b|\benable_eplb\b", "eplb"),
    (r"\benable-expert-parallel\b|\benable_expert_parallel\b|--ep\b", "ep"),
    (r"\bspeculative\b|--speculative-model|\beagle\b|\bmedusa\b", "spec-decode"),
    (r"\benable-prefix-caching\b|\benable_prefix_caching\b", "prefix-cache"),
    (r"--cuda-graph", "cuda-graph"),
    (r"\bFP8\b|--quantization fp8\b", "fp8"),
    (r"-tp[ =]\d", "tp"),
    (r"-pp[ =]\d", "pp"),
    (r"-dp[ =]\d", "dp"),
]


def _features_of(serve_cmd: str) -> tuple[str, ...]:
    feats: list[str] = []
    for pat, name in _FEATURE_PATTERNS:
        if re.search(pat, serve_cmd, re.IGNORECASE):
            feats.append(name)
    return tuple(sorted(feats))


@dataclass(frozen=True)
class WorkloadCluster:
    cluster_id: str  # e.g. "qwen3-moe + ep + eplb"
    n_prs: int
    pr_numbers: tuple[int, ...]
    canonical_model: str
    canonical_serve_cmd: str
    canonical_bench_cmd: str
    canonical_hardware: str
    features: tuple[str, ...]


def _cluster(extractions: dict[int, PRWorkloadExtraction]) -> list[WorkloadCluster]:
    """Group (model_family, features) → most-cited canonical commands."""
    buckets: dict[tuple[str, tuple[str, ...]], list[tuple[int, WorkloadConfig]]] = defaultdict(list)
    for pr_n, ext in extractions.items():
        for cfg in ext.tested_against:
            fam = _model_family(cfg.model)
            feats = _features_of(cfg.serve_command)
            buckets[(fam, feats)].append((pr_n, cfg))

    clusters: list[WorkloadCluster] = []
    for (fam, feats), entries in buckets.items():
        # Distinct PR set
        prs = tuple(sorted({pr_n for pr_n, _ in entries}))
        # Pick the most-cited model + serve_command pair
        model_counter = Counter(cfg.model for _, cfg in entries)
        top_model, _ = model_counter.most_common(1)[0]
        cmd_counter = Counter(cfg.serve_command for _, cfg in entries if cfg.model == top_model)
        top_serve, _ = cmd_counter.most_common(1)[0]
        # Pick a bench cmd that pairs with the top serve
        bench_for_top = [cfg.bench_command for _, cfg in entries if cfg.serve_command == top_serve and cfg.bench_command]
        top_bench = Counter(bench_for_top).most_common(1)[0][0] if bench_for_top else ""
        # Pick most-cited hardware
        hw_for_top = [cfg.hardware for _, cfg in entries if cfg.serve_command == top_serve and cfg.hardware]
        top_hw = Counter(hw_for_top).most_common(1)[0][0] if hw_for_top else ""

        cluster_id = f"{fam} + {' + '.join(feats) if feats else 'no-flags'}"
        clusters.append(WorkloadCluster(
            cluster_id=cluster_id,
            n_prs=len(prs),
            pr_numbers=prs,
            canonical_model=top_model,
            canonical_serve_cmd=top_serve,
            canonical_bench_cmd=top_bench,
            canonical_hardware=top_hw,
            features=feats,
        ))

    clusters.sort(key=lambda c: -c.n_prs)
    return clusters


# ── Render ────────────────────────────────────────────────────────────


def render_portfolio_md(clusters: list[WorkloadCluster], top_n: int) -> str:
    """Render the top-N clusters as a runnable workload portfolio."""
    lines: list[str] = ["## Recommended workloads (auto-extracted from PR bodies)\n"]
    if not clusters:
        lines.append(
            "_(No runnable benchmark commands extracted. Inspect the "
            "filtered PR bodies directly for workload context.)_\n"
        )
        return "\n".join(lines)

    lines.append(
        f"Top {min(top_n, len(clusters))} clusters by PR count. Each "
        f"cluster's canonical command is the most-cited skeleton from "
        f"the PRs in that group. Source PR numbers listed.\n"
    )
    for i, c in enumerate(clusters[:top_n], 1):
        lines.append(f"### W{i} — {c.cluster_id}\n")
        lines.append(f"**Covers {c.n_prs} PRs.**" + (
            f" Hardware: {c.canonical_hardware}.\n" if c.canonical_hardware
            else "\n"
        ))
        lines.append("\n```bash")
        lines.append(c.canonical_serve_cmd.strip())
        if c.canonical_bench_cmd:
            lines.append("")
            lines.append(c.canonical_bench_cmd.strip())
        lines.append("```\n")
        prs_excerpt = ", ".join(f"#{n}" for n in c.pr_numbers[:10])
        if len(c.pr_numbers) > 10:
            prs_excerpt += f" + {len(c.pr_numbers) - 10} more"
        lines.append(f"_Source PRs: {prs_excerpt}_\n")
    return "\n".join(lines)


# ── Orchestration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkloadCommandsHandle:
    json_path: Path
    md_path: Path
    n_prs_extracted: int
    n_prs_with_commands: int
    n_clusters: int
    portfolio_md: str  # ready to inline into bench-spec


def extract(
    *,
    window_id: str,
    view_path: Path,
    run_dir: Path,
    judge_model: str = "sonnet",
    runner: _RunnerFn | None = None,
    repo_path: Path | None = None,
    data_root_override: Path | None = None,
    wallclock_s: int = DEFAULT_WALLCLOCK_S,
    max_turns: int = DEFAULT_MAX_TURNS,
    top_n: int = DEFAULT_PORTFOLIO_TOP_N,
) -> WorkloadCommandsHandle:
    """Run LLM workload-command extraction over a filtered view."""
    runner = runner or _run_claude_default
    repo_path = repo_path or Path.cwd()
    cache_root = data_root_override or data_root()

    if not view_path.exists():
        raise FileNotFoundError(f"view not found: {view_path}")

    # Load view PR numbers, then join against raw to get bodies
    view_pr_numbers = {
        int(json.loads(line)["pr_number"])
        for line in view_path.open(encoding="utf-8")
        if line.strip()
    }

    raw_path = raw_dir(window_id, root=cache_root) / "prs.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw scrape not found: {raw_path}")

    raw_by_n: dict[int, RawPR] = {}
    for d in read_jsonl_lenient(raw_path):
        try:
            pr = RawPR.model_validate(d)
        except ValidationError:
            continue
        if pr.pr_number in view_pr_numbers:
            raw_by_n[pr.pr_number] = pr

    schema_text = json.dumps(
        PRWorkloadExtraction.model_json_schema(),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    p_sha = prompt_sha256()

    # Resume from prior partial run: read any existing per-PR cache and
    # skip those PRs. Cache is JSONL, append-only, one row per PR.
    cache_path = run_dir / "workload_commands_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached_per_pr: dict[int, dict] = {}
    if cache_path.exists():
        for row in read_jsonl_lenient(cache_path):
            pr_n = row.get("pr_number")
            if isinstance(pr_n, int):
                # Last write wins for a given PR, in case of a re-run.
                cached_per_pr[pr_n] = row
        log.info(
            "workload_commands: resuming with %d cached PRs in %s",
            len(cached_per_pr), cache_path,
        )

    log.info(
        "workload_commands: extracting from %d PRs (model=%s, prompt_sha=%s)",
        len(raw_by_n), judge_model, p_sha[:12],
    )

    extractions: dict[int, PRWorkloadExtraction] = {}
    n_with_commands = 0
    per_pr_records: list[dict] = []

    for i, pr_n in enumerate(sorted(raw_by_n), 1):
        # Cache hit: load and skip the LLM call.
        if pr_n in cached_per_pr:
            row = cached_per_pr[pr_n]
            per_pr_records.append(row)
            if not row.get("skipped"):
                ext_data = row.get("extraction") or {}
                try:
                    ext = PRWorkloadExtraction.model_validate(ext_data)
                except ValidationError:
                    continue
                extractions[pr_n] = ext
                if ext.tested_against:
                    n_with_commands += 1
            continue

        pr = raw_by_n[pr_n]
        prompt = PROMPT_TEMPLATE.format(
            pr_number=pr.pr_number,
            title=pr.title,
            body=(pr.body or "")[:8000],  # cap to avoid blowing context
        )
        log.info("workload_commands: [%d/%d] PR #%d — extracting",
                 i, len(raw_by_n), pr_n)
        result = runner(
            candidate_id=f"workload-{pr_n}",
            prompt=prompt,
            schema_text=schema_text,
            repo_path=repo_path,
            max_turns=max_turns,
            wallclock_s=wallclock_s,
        )
        if result.error is not None or result.structured_output is None:
            log.warning("workload_commands: PR #%d failed: %s",
                        pr_n, result.error or "no output")
            row = {
                "pr_number": pr_n, "skipped": True,
                "reason": result.error or "no_structured_output",
            }
            per_pr_records.append(row)
            append_jsonl(cache_path, row)
            continue
        try:
            ext = PRWorkloadExtraction.model_validate(result.structured_output)
        except ValidationError as e:
            log.warning("workload_commands: PR #%d invalid: %s", pr_n, e)
            row = {
                "pr_number": pr_n, "skipped": True,
                "reason": f"validation_error: {e}",
            }
            per_pr_records.append(row)
            append_jsonl(cache_path, row)
            continue
        # Hallucination guard
        cleaned, dropped = _validate_extraction(ext, pr.body or "")
        if dropped:
            log.info("workload_commands: PR #%d dropped %d hallucinated commands",
                     pr_n, len(dropped))
        extractions[pr_n] = cleaned
        if cleaned.tested_against:
            n_with_commands += 1
        row = {
            "pr_number": pr_n,
            "title": pr.title,
            "extraction": cleaned.model_dump(),
            "dropped_for_hallucination": dropped,
        }
        per_pr_records.append(row)
        append_jsonl(cache_path, row)

    clusters = _cluster(extractions)
    portfolio_md = render_portfolio_md(clusters, top_n=top_n)

    out = {
        "window_id": window_id,
        "judge_model": judge_model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": p_sha,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "n_prs_extracted": len(extractions),
        "n_prs_with_commands": n_with_commands,
        "top_n": top_n,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "n_prs": c.n_prs,
                "pr_numbers": list(c.pr_numbers),
                "canonical_model": c.canonical_model,
                "canonical_serve_cmd": c.canonical_serve_cmd,
                "canonical_bench_cmd": c.canonical_bench_cmd,
                "canonical_hardware": c.canonical_hardware,
                "features": list(c.features),
            }
            for c in clusters
        ],
        "per_pr": per_pr_records,
    }
    json_path = run_dir / "workload_commands.json"
    write_json(json_path, out)

    md_path = run_dir / "workload_commands.md"
    atomic_write_text(md_path, portfolio_md)

    log.info(
        "workload_commands: %d PRs extracted, %d with commands, %d clusters",
        len(extractions), n_with_commands, len(clusters),
    )

    return WorkloadCommandsHandle(
        json_path=json_path,
        md_path=md_path,
        n_prs_extracted=len(extractions),
        n_prs_with_commands=n_with_commands,
        n_clusters=len(clusters),
        portfolio_md=portfolio_md,
    )


__all__ = [
    "PRWorkloadExtraction",
    "WorkloadCluster",
    "WorkloadCommandsHandle",
    "WorkloadConfig",
    "PROMPT_VERSION",
    "extract",
    "prompt_sha256",
    "render_portfolio_md",
]
