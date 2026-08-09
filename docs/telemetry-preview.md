# Telemetry-driven discovery (preview)

> [!IMPORTANT]
> This path is a **preview** (MVP). Knowledge retrieval and archive are deferred per the design doc, and it is run via the separate `signal-pipeline` command — not `spotlights-engine`.

Given a captured workload's OpenTelemetry traces and the subject repo, the `signal-pipeline` CLI runs five stages — signal extraction, ProjectTree extraction, candidate generation, change generation, execution — to produce evidence-backed code changes with rationales and applied diffs. A canonical run on a vLLM/LRU OTel capture takes ~10 min and ~$2 in API costs and yields a handful of candidates anchored to the captured anomalies.

Its output is the same unified `SpotlightReport` shape emitted by the deep-research path (see commit `c671bbf`, PR #35), so `/spotlights-sort-candidates` and the rendered `index.md`/module tree work on it too.

← Back to [README](../README.md)

```bash
signal-pipeline \
    --output-folder ./spotlights-out \
    --artifacts-dir ./artifacts \
    --repo ../vllm \
    --telemetry-from <path-to-otel-capture-or-signals.json>
```

`--telemetry-from` accepts either a directory of raw OTel files or a pre-cooked `01_signals.json`. For stage-by-stage details, prompt iteration, the artifacts-dir layout, the `--inject` workflow for hand-edited intermediates, model selection, and troubleshooting, see [`../src/spotlights_engine/signal_pipeline/README.md`](../src/spotlights_engine/signal_pipeline/README.md). Architecture and contracts live in [`signal-based/`](signal-based/).
