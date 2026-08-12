# Spotlights Run — vllm

## Repository
- **Name:** vllm
- **Summary:** vLLM is a high-throughput, memory-efficient inference and serving engine for LLMs. It is a Python library (with C++/CUDA/HIP/Triton kernels and a small Rust component) built on PyTorch, exposing OpenAI/Anthropic/Cohere-compatible REST, gRPC, MCP, and offline APIs. Architecturally it is a layered engine: an entrypoints/API layer over an EngineCore (v1 rewrite; v0 legacy in engine/) that orchestrates a scheduler, KV-cache manager, model executor, and pluggable per-device workers, with hardware-abstracted platforms, torch.compile integration, distributed/parallel state, LoRA, multimodal, quantization, and structured-output/tool/reasoning parser subsystems.
- **External dependencies:** torch, transformers, tokenizers, numpy, fastapi, uvicorn, pydantic, aiohttp, openai, httpx, ray, triton, flashinfer, xgrammar, outlines, llguidance, lm-format-enforcer, prometheus-client, opentelemetry-api, huggingface-hub, safetensors, sentencepiece, mistral-common, tiktoken, msgpack, msgspec, zmq, protobuf, grpcio, cloudpickle, pillow, librosa, soundfile, psutil, pyzmq, boto3, runai-model-streamer, tensorizer, peft
- **Repo path:** /Users/iklamer/ai-native-systems/tmp/vllm
- **Run created:** 2026-08-11T19:35:36+00:00
- **Run status:** COMPLETE
- **Extractor duration (s):** 734.0

## Context
- **Objective:** reduce the median TTFT and median TPOT (Time Per Output Token)
- **Workload hints:**
  - Multi-turn agentic workload
- **Validation plan:**
  - _(none)_

## Modules

| Module | Candidates | High-impact | Relevant findings |
|---|---:|---:|---:|
| [vllm/distributed/kv_transfer](modules/vllm_distributed_kv_transfer.md) | 21 | 10 | 15 |
| [vllm/v1/engine](modules/vllm_v1_engine.md) | 18 | 4 | 12 |
| [vllm/v1/kv_offload](modules/vllm_v1_kv_offload.md) | 19 | 8 | 12 |
| [vllm/v1/core](modules/vllm_v1_core.md) | 21 | 7 | 11 |
| [vllm/v1/executor](modules/vllm_v1_executor.md) | 14 | 6 | 11 |
| [vllm/v1/attention](modules/vllm_v1_attention.md) | 22 | 4 | 9 |
| [vllm/v1/worker](modules/vllm_v1_worker.md) | 32 | 17 | 8 |
| [vllm/multimodal](modules/vllm_multimodal.md) | 26 | 4 | 7 |

## Run manifest

- **Run id:** `run-4b5859bcc6cc3ad1`
- **Date:** 2026-08-11T19:35:36+00:00
- **Target repo:** https://github.com/vllm-project/vllm.git
- **Target commit:** `83ad767eed3be3ee7f2df63be693bfaca5c7c922`
- **Spotlights commit:** `6c23650a2dd309648d1652145eb9f29fa2609206`
- **Pipeline:** deep-research
- **Candidates:** 173
- **Module status:** SUCCEEDED: 2, DEGRADED: 6, FAILED: 0, SKIPPED: 0

### Cost & usage

- **Total cost (USD):** $35.7498  _(source: contracted-rate-table)_
- **External cost (USD):** $57.1842  _(source: public-api-rate-table)_
- **Total tokens:** 266420297
- **Wall clock (s):** 25656.7
- **Accumulated duration (s):** 25655.5
- **API time (s):** 70453.5

| Model | Provider | Role | Input | Output | Cache read | Cache create |
|---|---|---|---:|---:|---:|---:|
| aws/claude-opus-4-7 | anthropic | agent_proposals | 4470 | 491572 | 36482881 | 4064996 |
| aws/claude-opus-4-7 | anthropic | candidate_discovery | 16277 | 269112 | 21833292 | 1610601 |
| aws/claude-opus-4-7 | anthropic | proposal_from_finding_creator | 13439 | 825762 | 138976984 | 27209248 |
| codex | openai | agent_proposals | 2662774 | 238607 | 13031680 | 0 |
| codex | openai | deep_research | 1097362 | 82396 | 1264896 | 0 |
| gpt-5.5 | openai | candidate_discovery | 1714167 | 195573 | 14334208 | 0 |

### Notes

- rates applied: openai:codex (CLI-family fallback for Codex invocations whose stream did not emit a resolvable model id; same rate as openai:gpt-5.5 because gpt-5.5 is the model behind every codex invocation in this deployment. LiteLLM (azure) rate as of 2026-07-07: $2.50/MTok input, $15/MTok output, $0.50/MTok cache read; Codex has no cache-create bucket. Override for a different contract.); openai:gpt-5.5 (LiteLLM (azure) rate as of 2026-07-07: GPT-5.5 $2.50/MTok input, $15/MTok output, $0.50/MTok cache read; no cache-create bucket. Used when the Codex stream emits msg.model=gpt-5.5; identical rates to the openai:codex CLI-family fallback.); PARTIAL cost — no contracted rate for: anthropic:aws/claude-opus-4-7; 181 invocation(s) had an unresolved model id (priced by CLI-family fallback key when available)
- rates applied: openai:codex (CLI-family fallback for Codex invocations whose stream did not emit a resolvable model id; same rate as openai:gpt-5.5 because gpt-5.5 is the model behind every codex invocation in this deployment. Public OpenAI API list price for standard GPT-5.5: $5.00/MTok input, $30.00/MTok output, $0.50/MTok cached input; no cache-create bucket. GPT-5.5 long-context (>272K input tokens) and some regional endpoints have different rates, so override via SPOTLIGHTS_EXTERNAL_RATES_FILE if those apply.); openai:gpt-5.5 (Public OpenAI API list price for standard GPT-5.5: $5.00/MTok input, $30.00/MTok output, $0.50/MTok cached input; no cache-create bucket. Used when the Codex stream emits msg.model=gpt-5.5; identical rates to the openai:codex CLI-family fallback.); PARTIAL cost — no contracted rate for: anthropic:aws/claude-opus-4-7; 181 invocation(s) had an unresolved model id (priced by CLI-family fallback key when available)
- unpriced models excluded from cost: anthropic:aws/claude-opus-4-7; external: unpriced models excluded from cost: anthropic:aws/claude-opus-4-7
