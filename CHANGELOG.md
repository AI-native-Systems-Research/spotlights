# Changelog

## v0 — first release (tag pending)

The first end-to-end version: point it at a repo and a goal, get a ranked, browsable map of the few places worth optimizing, each with evidence-backed proposals.

### Engine — `spotlights-engine`

- **Structural map** — static analysis extracts the repo's module tree.
- **Candidate discovery** — per module, picks the functions and code regions most worth investigating for the stated objective.
- **Deep research** — per module, retrieves findings from the web, arXiv, blogs, and docs.
- **Proposals** — turns findings into concrete change proposals with citations, plus a second agent pass for proposals the literature doesn't cover.
- **`SpotlightReport` output pydantic model** — the full Spotlights output.
- **Module scoping** — `--include` limits a run to the modules you name (slash-form, e.g. `vllm/v1/kv_offload`). Nothing outside the scope is extracted or researched. Narrower scope reduces run time and cost.
- **Cost tracking** — `run_manifest.json` records provenance, token usage, and price.
- **`prep-evolve`** — packages a chosen candidate as a ready-to-run bundle for an external evolver (skydiscover, coral, nous).
- **`apply`** — one Claude Code session turns a candidate into a reviewable patch.

### Skills

- **`/spotlights-objective-setting`** — interview that turns a vague goal into ready-to-paste `--objective` / `--hint` flags.
- **`/spotlights-sort-candidates`** — ranks a finished run's candidates by estimated impact into `sorted_candidates.md` / `.json`.
- **`/spotlights-share-candidates`** — packages the top-N candidates into a self-contained offline ZIP.
- **`/spotlights-apply-candidate`** — drives `spotlights-engine apply` in-session.

### Known limitations

- **Some candidates will be irrelevant to your setup.** v0 reasons over the code as written and knows nothing about your workload or configuration, so it can surface candidates in paths your deployment never executes — a disabled backend, an unset feature flag, another platform's code. Ranking orders candidates by estimated impact; it does not check reachability. Expect to discard some on review.
- **For now, use `--include`** to limit the analysis to modules you know are relevant.

## In progress

- **Extend the objective/hint** — extend the input to capture deployment and workload profile: how the system is configured and what workload/data it processes.
- **Module exclusion** — `--exclude` takes a package/module list to skip, so the engine never runs on it.
- **Exclusion agent** — infers which modules can't be reached under the given setup, workload, and configuration, and populates the exclusion list.
