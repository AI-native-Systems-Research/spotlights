# candidate_research_proposer — Stage 2: Deep Research via GPT Researcher

Stage 2 produces `findings.json` — a list of curated external optimization techniques (papers, PRs, issues, blogs, talks, vendor docs) that downstream Stage 3a will map to the Stage 1 candidate locations.

## 1. Provider: GPT Researcher

- Repo: https://github.com/assafelovic/gpt-researcher (Apache-2.0)
- Library import: `from gpt_researcher import GPTResearcher`
- Dependency floor: `gpt-researcher >= 0.14` (lock an exact version after the M0 smoke test)
- Why: importable directly from Python, emits a structured report we coerce to our findings schema, supports arbitrary OpenAI-compatible LLM endpoints via env vars, separates LLM from retriever, permissive Apache-2.0 license.

## 2. LLM configuration — LiteLLM proxy

GPT Researcher honors `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the process environment (and equivalent config-file keys). The wrapper sets:

```bash
export OPENAI_API_KEY="…"                                            # the proxy key
export OPENAI_BASE_URL="https://ete-litellm.ai-models.vpc-int.res.ibm.com"
# Pin the same model name everywhere GPT Researcher looks for one
export FAST_LLM="openai:<MODEL_NAME>"
export SMART_LLM="openai:<MODEL_NAME>"
export STRATEGIC_LLM="openai:<MODEL_NAME>"
export EMBEDDING="openai:<EMBEDDING_MODEL_NAME>"   # required by GPT Researcher's context compression (see §4)
```

…before constructing `GPTResearcher(query=…, report_type="deep", report_format="markdown")`. Upstream's enum value for deep research is `deep` (`ReportType.DeepResearch.value`), not `deep_research`. The orchestrator must isolate these vars from the Claude Code / Codex subprocesses using the same `_clean_env` pattern described in the [Stage 1 env-scrubbing section](candidate_research_proposer_stage1_candidate_discovery.md#4-subprocess-invocation).

Stage 2 is text-only; no image / vision input.

## 3. Retriever (search backend) — default `arxiv`, Tavily recommended

Stage 2 has two retriever modes:

- **Default — `RETRIEVER=arxiv`.** No API key, no install-time secret, no paid search API. Calls the arxiv API directly; authors / abstracts / arxiv categories influence the synthesized markdown report but are dropped by coercion (only the 5 canonical fields survive — anything else can be re-fetched from `url` by Stage 3). Always on; works out of the box on a node with internet access.
- **Recommended add-on — `RETRIEVER=tavily,arxiv`.** Adds Tavily's AI-curated web search alongside the arxiv API. Surfaces what arxiv can't: engineering blogs (Cloudflare/Meta/Netflix/etc.), GitHub issues and PRs, vendor docs (CUDA/PyTorch/JAX/MLIR), Stack Overflow, conference talks, news. Requires `TAVILY_API_KEY`. Paid — see costs below.

**Startup behavior (no flag needed in the common case):** the orchestrator inspects the environment at startup. If `TAVILY_API_KEY` is set, it exports `RETRIEVER=tavily,arxiv`; otherwise it exports `RETRIEVER=arxiv` and prints a one-line warning to stderr:

```
[stage2] running with arxiv only — set TAVILY_API_KEY to enable Tavily for blog / GitHub / vendor-doc coverage
```

An explicit `--retriever <value>` flag remains available for overrides (e.g., `--retriever duckduckgo` for a laptop smoke test, `--retriever custom,arxiv` for an in-VPC redeploy).

GPT Researcher accepts multiple retrievers as a comma-separated `RETRIEVER` value and uses each specified retriever in sequence ([retriever docs, verified 2026-05-15](https://docs.gptr.dev/docs/gpt-researcher/search-engines)).

**What the default costs in coverage:**

| Source type | `arxiv` default | `tavily,arxiv` |
|---|---|---|
| Peer-reviewed / preprint papers | ✓ (clean metadata via API) | ✓ (arxiv primary, Tavily as backup) |
| Engineering blogs (Cloudflare, Meta, Netflix, …) | — | ✓ |
| GitHub issues, PRs, discussions | — | ✓ |
| Vendor docs (CUDA, PyTorch, JAX, MLIR, …) | — | ✓ |
| Conference talks, slide decks (non-arxiv) | — | ✓ |
| Stack Overflow / Reddit / HN | — | ✓ |

Running arxiv-only is a **known degraded mode**, not a failure: smaller findings sets, higher Stage 3 mapping-orphan rates for optimization opportunities documented only in blogs/issues. The pipeline still runs end-to-end. The Stage 2 wrapper records `retriever_limitations: "arxiv_only"` in the run-level sidecar `findings.meta.json` (see §8) — the canonical `findings.json` schema is unchanged.

**Cost.**

- **arxiv API:** free. Rate-limited to ~1 request / 3 seconds; the underlying `arxiv` Python package defaults to that delay and retries transient failures. ~300K result cap per query (irrelevant for our sub-query sizes).
- **Tavily:** free tier is 1K credits/month with no card required; the $30 Project plan currently includes 4K credits/month; PAYG is ~$0.008/credit ([pricing, verified 2026-05-15](https://docs.tavily.com/documentation/api-credits)). Working estimate: one Stage 2 `deep` run with Tavily burns ~20–60 credits through search/extract calls, so the $30 tier covers roughly 65–200 runs/month. Treat this as an M0-measured number, not a hard guarantee.

**Not used:**

- **DuckDuckGo** — rate-limited / blocked from cloud IPs in practice. Available via `--retriever duckduckgo` for laptop bring-up only.
- **Exa, Serper, SerpAPI, Bing, Google** — Tavily is GPT Researcher's reference retriever and the best-tested integration. No evidence the others would materially improve recall here. Revisit only if M1 shows Tavily relevance is poor.

**If Tavily egress is ever lost** (e.g., deployment moves into a restricted VPC), in-VPC alternatives: `RETRIEVER=custom,arxiv` with an adapter for an internal search index ([custom retriever docs](https://docs.gptr.dev/docs/gpt-researcher/search-engines#custom-retrievers)), or `RETRIEVER=searx,arxiv` against a self-hosted [SearXNG](https://github.com/searxng/searxng). The arxiv default keeps working in either case.

## 4. Embeddings — required by GPT Researcher's context compression

GPT Researcher initializes its memory/context-compression stack with an embedding provider, and deep mode recursively invokes the standard research flow for subqueries. Therefore `EMBEDDING` (see §2) MUST point at a working embeddings model on the LiteLLM proxy. The M0 proxy selftest probes `/v1/embeddings` and treats a failure as a hard blocker for Stage 2 (unlike the chat-completions endpoint, there is no fallback).

Embeddings are **not** used by Stage 3 mapping — the per-finding fan-out (see Stage 3a plan) judges applicability inside each coding-agent call. The MVP's only `--prefilter` option is `bm25` (pure `rank_bm25`, no embedding endpoint touched). A future `--prefilter embeddings` could reuse the same `EMBEDDING` model via `client.embeddings.create(...)` against the proxy.

## 5. Single-call vs deep-research mode

GPT Researcher's relevant `report_type` options (source: `gpt_researcher/utils/enum.py`; deep-mode docs verified 2026-05-15):

- `research_report` — single retrieval pass, single synthesis; fastest, cheapest.
- `detailed_report` — section-by-section, deeper.
- `deep` — recursive breadth/depth search, deepest. Best for our use case.

Pin: `report_type="deep"`. Expect roughly 5–15 minutes wall-clock per call at GPT Researcher's default depth/breadth (`DEEP_RESEARCH_BREADTH=3`, `DEEP_RESEARCH_DEPTH=2`, `DEEP_RESEARCH_CONCURRENCY=4` in the current default config). The §7 `--stage2-max-iterations` flag maps onto `DEEP_RESEARCH_DEPTH` in the config file the wrapper writes; raising it widens the cost/time envelope roughly linearly.

## 6. Markdown-to-schema coercion (mandatory)

GPT Researcher emits **markdown**. Our pipeline needs validated JSON matching the `findings.json` schema. The orchestrator therefore makes a **second LLM call** through the same LiteLLM proxy:

```
COERCION CALL
  endpoint: OPENAI_BASE_URL
  model:    <MODEL_NAME>
  system:   "You convert a research report into a strict JSON findings list."
  user:     report_markdown + COERCION_PROMPT (below)
  response_format:
    type: "json_schema"
    json_schema:
      name:   "findings"
      schema: <pydantic model_json_schema() of the finding list>
      strict: true
```

If the proxy's `<MODEL_NAME>` does not support `response_format=json_schema`, the wrapper falls back to plain JSON-mode (`response_format={"type":"json_object"}`) and then validates with pydantic, retrying once on parse failure with a strict reminder. The M0 proxy selftest must verify which response-format path works for the selected model.

The coercion call returns only the non-canonical object `{ "findings": [...] }`. The orchestrator (not the LLM) wraps it into the canonical envelope `{schema_version, module_qualified_name, findings}` and renumbers each `id` deterministically as `find-NNNN` in input order before writing `findings.json` — so the LLM's `id` values in §10 are placeholders.

## 7. Per-call budget cap

Enforce:

- `--stage2-wallclock-s` (default 1800 s = 30 min) — hard timeout on the GPT Researcher call.
- `--stage2-max-iterations` — pass to GPT Researcher `config_path` to bound the deep-research tree.
- `--stage2-budget-usd` — best-effort; enforce with GPT Researcher's `get_costs()` plus LiteLLM `usage` / cost metadata for the coercion call when available.

## 8. Validation

- Schema parse → reject on failure (retry coercion once with strict reminder).
- URL reachability: for every `finding.url`, issue HTTP HEAD with a 10s timeout; on any non-2xx response (arxiv abstract pages, GitHub PR/issue URLs, and several CDN-fronted blogs return 403/405/301-loops on HEAD), fall back to a ranged GET (`Range: bytes=0-0`, same timeout) before declaring the URL unreachable. Warn if `unreachable_count > 0.2 * total_count`.
- Reachability results live in the sidecar `findings.meta.json`, not on the per-finding record — the canonical `findings.json` schema stays at exactly five fields. Sidecar shape (`retriever_limitations` is `null` when Tavily is enabled):
  ```json
  {
    "run_id": "...",
    "module_qualified_name": "foo/bar",
    "retriever_limitations": "arxiv_only",
    "url_reachability": [
      {"finding_id": "find-0001", "reachable": true,  "status": 200, "method": "HEAD"},
      {"finding_id": "find-0002", "reachable": false, "status": 404, "method": "GET"}
    ]
  }
  ```
- Minimum count: ≥8 findings (matches the §9 prompt's lower bound) or Stage 2 is flagged degraded (run still proceeds; Stage 3a will likely produce orphans).

## 9. Stage 2 research prompt (drop-in, fed as `query` to GPTResearcher)

````
You are conducting a focused literature/engineering survey to support a code-optimization
pipeline. Your output will be parsed by a downstream mapper that attaches your findings
to specific code locations in the user's module.

## Module under audit
- repo: {repo_url_or_path}
- module_qualified_name: {module_qualified_name}
- module_path: {module_path}
- one-paragraph summary (derived from `Module.description`, `Module.main_files`,
  and `Module.submodules`; independent of Stage 1 so Stage 1 and Stage 2 can run
  in parallel):
  """
  {module_summary}
  """

## What to look for
Surface, in order of preference:
1. Peer-reviewed papers proposing techniques with measured speedups, memory wins, or
   throughput wins relevant to the module summary above.
2. GitHub pull requests / issues IN ANY REPO (not just {repo_url_or_path}) that landed measured
   wins in similar systems.
3. Engineering blog posts from vendors and frameworks (NVIDIA, PyTorch, JAX, vLLM, SGLang,
   TGI, DeepSpeed, etc.) describing concrete optimizations with numbers.
4. Conference talks (MLSys, OSDI, SOSP, ASPLOS, SC, PyTorch Conf, GTC) with slides/recordings.

Avoid: pure marketing posts, tutorials with no benchmarks, social-media speculation.

## Bias and constraints
- Prefer sources <= 36 months old unless the technique is foundational.
- For every finding, you MUST resolve a real, fetchable URL. No paraphrased
  recollections without a URL.
- Aim for 8-20 findings. Quality > quantity.

Produce a thorough markdown report. Group findings into logical sections. For each finding,
include at minimum: title, URL, source_type (paper/pr/issue/blog/talk/docs/codebase), and a
≤80-word summary of the technique the source proposes. You may include additional context
(gains, components, suggested changes, evidence assessment) in the prose if helpful for
the reader, but only title, url, source_type, and technique_summary survive coercion;
the orchestrator adds the finding id. The downstream Stage-3 agents fetch the source URL
themselves when they need implementation detail.
````

## 10. Stage 2 coercion prompt (drop-in, fed to second LLM call)

````
You are converting a research report into a strict JSON findings list.

## Input
A markdown report on optimization techniques relevant to a code module. The report is
appended below the `---` separator.

## Output
A single JSON object matching the schema at @findings.schema.json. The top-level object
has exactly one key, `findings`. Every finding has exactly five fields — no more:
- `id` — placeholder of the form `find-NNNN` (zero-padded, sequential within the report); the orchestrator will renumber deterministically post-coercion, so do not worry about gaps
- `title`
- `url` — a real, fetchable URL present in the report; do NOT invent URLs
- `source_type` ∈ {paper, pr, issue, blog, talk, docs, codebase}
- `technique_summary` (≤ 80 words)

- Only include findings whose URL is present in the report.
- Drop any finding for which you cannot fill all five fields.
- Do not emit prose outside the JSON object.
- Do not emit any field other than the five above.

---

{report_markdown}
````
