# Stage 2 Mechanics — Deep Research via GPT Researcher

## 6.1 Provider: GPT Researcher

- Repo: https://github.com/assafelovic/gpt-researcher (Apache-2.0)
- Library import: `from gpt_researcher import GPTResearcher`
- Pin: `gpt-researcher >= 0.13`
- Why: importable directly from Python, emits a structured report we coerce to our findings schema, supports arbitrary OpenAI-compatible LLM endpoints via env vars, separates LLM from retriever, MIT-friendly license.

## 6.2 LLM configuration — LiteLLM proxy

GPT Researcher honors `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the process environment (and equivalent config-file keys). The wrapper sets:

```bash
export OPENAI_API_KEY="…"                                            # the proxy key
export OPENAI_BASE_URL="https://ete-litellm.ai-models.vpc-int.res.ibm.com"
# Pin the same model name everywhere GPT Researcher looks for one
export FAST_LLM="openai:<MODEL_NAME>"
export SMART_LLM="openai:<MODEL_NAME>"
export STRATEGIC_LLM="openai:<MODEL_NAME>"
export EMBEDDING="openai:<EMBEDDING_MODEL_NAME>"   # see §6.4
```

…before constructing `GPTResearcher(query=…, report_type="deep_research", report_format="markdown")`. The orchestrator must isolate these vars from the Claude Code / Codex subprocesses (see §4 _clean_env note).

Stage 2 is text-only; no image / vision input.

## 6.3 Retriever (search backend) — default `arxiv`, Tavily recommended

Stage 2 has two retriever modes:

- **Default — `RETRIEVER=arxiv`.** No API key, no install-time secret, no paid egress. Calls the arxiv API directly; authors / abstracts / arxiv categories survive into `findings.json` unchanged. Always on; works out of the box on a node with internet access.
- **Recommended add-on — `RETRIEVER=tavily,arxiv`.** Adds Tavily's AI-curated web search in parallel with the arxiv API. Surfaces what arxiv can't: engineering blogs (Cloudflare/Meta/Netflix/etc.), GitHub issues and PRs, vendor docs (CUDA/PyTorch/JAX/MLIR), Stack Overflow, conference talks, news. Requires `TAVILY_API_KEY`. Paid — see costs below.

**Startup behavior (no flag needed in the common case):** the orchestrator inspects the environment at startup. If `TAVILY_API_KEY` is set, it exports `RETRIEVER=tavily,arxiv`; otherwise it exports `RETRIEVER=arxiv` and prints a one-line warning to stderr:

```
[stage2] running with arxiv only — set TAVILY_API_KEY to enable Tavily for blog / GitHub / vendor-doc coverage
```

An explicit `--retriever <value>` flag remains available for overrides (e.g., `--retriever duckduckgo` for a laptop smoke test, `--retriever custom,arxiv` for an in-VPC redeploy).

GPT Researcher accepts a list of retrievers, runs them in parallel, and merges results before re-ranking ([retriever docs, verified 2026-05-13](https://docs.gptr.dev/docs/gpt-researcher/search-engines/retrievers)).

**What the default costs in coverage:**

| Source type | `arxiv` default | `tavily,arxiv` |
|---|---|---|
| Peer-reviewed / preprint papers | ✓ (clean metadata via API) | ✓ (arxiv primary, Tavily as backup) |
| Engineering blogs (Cloudflare, Meta, Netflix, …) | — | ✓ |
| GitHub issues, PRs, discussions | — | ✓ |
| Vendor docs (CUDA, PyTorch, JAX, MLIR, …) | — | ✓ |
| Conference talks, slide decks (non-arxiv) | — | ✓ |
| Stack Overflow / Reddit / HN | — | ✓ |

Running arxiv-only is a **known degraded mode**, not a failure: smaller findings sets, higher Stage 3 mapping-orphan rates for optimization opportunities documented only in blogs/issues. The pipeline still runs end-to-end. The Stage 2 wrapper sets `findings.query_context.retriever_limitations = "arxiv_only"` in that mode so downstream consumers can interpret the lower hit rate.

**Cost.**

- **arxiv API:** free. Rate-limited to ~1 request / 3 seconds; the underlying `arxiv` Python package handles back-off automatically. ~30K result cap per query (irrelevant for our sub-query sizes).
- **Tavily:** $30/mo for 10K credits, ~$0.008/credit on PAYG ([pricing, verified 2026-05-13](https://docs.tavily.com/documentation/api-credits)). One `deep_research` call burns ~20–60 credits (per-subquery search + extraction). The $30 tier comfortably absorbs ~150–500 runs/month. Free tier is 1K credits/month with no card required, enough for early dev work.

**Not used:**

- **DuckDuckGo** — rate-limited / blocked from cloud IPs in practice. Available via `--retriever duckduckgo` for laptop bring-up only.
- **Exa, Serper, SerpAPI, Bing, Google** — Tavily is GPT Researcher's reference retriever and the best-tested integration. No evidence the others would materially improve recall here. Revisit only if M1 shows Tavily relevance is poor.

**If Tavily egress is ever lost** (e.g., deployment moves into a restricted VPC), in-VPC alternatives: `RETRIEVER=custom,arxiv` with an adapter for an internal search index ([custom retriever docs](https://docs.gptr.dev/docs/gpt-researcher/search-engines/retrievers#custom-retriever)), or `RETRIEVER=searx,arxiv` against a self-hosted [SearXNG](https://github.com/searxng/searxng). The arxiv default keeps working in either case.

## 6.4 Embeddings — not used by MVP mapping; deferred

The per-finding fan-out (§7.1) judges applicability inside each coding-agent call, so no embeddings endpoint is needed for the MVP. This section is a scaffold for a possible v2 hybrid-retrieval pre-filter:

- A v2 `--prefilter embeddings` (not implemented) would call `client.embeddings.create(model="<EMBEDDING_MODEL_NAME>", input=[...])` against the LiteLLM proxy, with a `sentence-transformers` / `BAAI/bge-m3` local fallback.
- The M0 proxy selftest probes `/v1/embeddings` so the endpoint's availability is recorded; result is informational only.

The MVP's only `--prefilter` option is `bm25` (pure `rank_bm25`, no embedding endpoint touched).

## 6.5 Single-call vs deep-research mode

GPT Researcher's `report_type` options (https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started-with-docker, source: `gpt_researcher/utils/enum.py`):

- `research_report` — single retrieval pass, single synthesis; fastest, cheapest.
- `detailed_report` — section-by-section, deeper.
- `deep_research` — multi-agent, tree-of-thought style search, deepest. Best for our use case.

Pin: `report_type="deep_research"`. Expect 5–15 minutes wall-clock per call.

## 6.6 Markdown-to-schema coercion (mandatory)

GPT Researcher emits **markdown**. Our pipeline needs validated JSON matching the `findings.json` schema. The orchestrator therefore makes a **second LLM call** through the same LiteLLM proxy:

```
COERCION CALL
  endpoint: OPENAI_BASE_URL
  model:    <MODEL_NAME>
  system:   "You convert a research report into a strict JSON findings list."
  user:     report_markdown + COERCION_PROMPT (below)
  response_format: {"type": "json_schema", "json_schema": findings_schema}
```

If the proxy's `<MODEL_NAME>` does not support `response_format=json_schema`, the wrapper falls back to plain JSON-mode (`response_format={"type":"json_object"}`) and then validates with pydantic, retrying once on parse failure with a strict reminder. See §10 Q6.

## 6.7 Per-call budget cap

Enforce:

- `--stage2-wallclock-s` (default 1800 s = 30 min) — hard timeout on the GPT Researcher call.
- `--stage2-max-iterations` — pass to GPT Researcher `config_path` to bound the deep-research tree.
- `--stage2-budget-usd` — best-effort; only enforceable if the LiteLLM proxy returns `usage` blocks (§10 Q5).

## 6.8 Validation

- Schema parse → reject on failure (retry coercion once with strict reminder).
- For every `finding.url`: HTTP HEAD with 10s timeout; mark `url_reachable: true|false`; warn if `unreachable_count > 0.2 * total_count`.
- Minimum count: ≥5 findings or Stage 2 is flagged degraded (run still proceeds; Stage 3a will likely produce orphans).

## 6.9 Stage 2 research prompt (drop-in, fed as `query` to GPTResearcher)

````
You are conducting a focused literature/engineering survey to support a code-optimization
pipeline. Your output will be parsed by a downstream mapper that attaches your findings
to specific code locations in the user's module.

## Module under audit
- repo: {repo_url}
- module_path: {module_path}
- one-paragraph summary (from the bootstrap candidate-discovery agent):
  """
  {module_summary}
  """
- top-level kinds of code present: {top_level_tags}   # e.g., ["scheduler","kv-cache","cuda-kernels"]

## What to look for
Surface, in order of preference:
1. Peer-reviewed papers proposing techniques with measured speedups, memory wins, or
   throughput wins relevant to {top_level_tags}.
2. GitHub pull requests / issues IN ANY REPO (not just {repo_url}) that landed measured
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
include: title, URL, source type (paper/PR/issue/blog/talk/docs/codebase), publication date,
the technique it proposes, the measured or claimed gains (with baseline and conditions),
the components or symbols it would apply to (use concrete keywords that would appear in
source code), suggested changes, and your assessment of evidence quality.
````

## 6.10 Stage 2 coercion prompt (drop-in, fed to second LLM call)

````
You are converting a research report into a strict JSON findings list.

## Input
A markdown report on optimization techniques relevant to a code module. The report is
appended below the `---` separator.

## Output
A single JSON object matching the schema at @findings.schema.json. Every finding must
populate:
- title, url, source_type ∈ {paper, pr, issue, blog, talk, docs, codebase}
- technique_name, technique_summary (≤ 80 words)
- claimed_gains[].(metric|magnitude|baseline|conditions)
- target_components[].(component_hint|file_or_symbol_hints[]|keywords_for_matching[])
- suggested_changes[] (≤ 5 bullets, each ≤ 25 words)
- evidence_quality.(primary|peer_reviewed|has_benchmarks|reproducible|score∈[0,1])

The `keywords_for_matching` array is what the downstream mapper will hybrid-search
against candidate code+rationale. Choose 5-15 specific tokens that would actually
appear in source code or in a code-review rationale (e.g., "kv_cache", "block_table",
"chunked_prefill", NOT "AI", "performance", "speedup").

- Only include findings whose URL is present in the report. Do NOT invent URLs.
- Drop any finding for which you cannot fill the required fields.
- Do not emit prose outside the JSON object.

---

{report_markdown}
````
