# spotlight-engine

The spine of the Spotlight Engine project: schemas, orchestration, integration tests, and the demo path.

## What this is

This repo is the spine. It owns the canonical schemas (`Candidate`, `Change`, and re-exports of the `Signal` types from `spotlight-observability`), the orchestration layer, and the integration-test surface that ties the bundles together. Bundles C (candidates), D (changes), E (validation), and F (orchestration) live here as subpackages until they earn their own repos.

## How it fits

The Spotlight Engine project is split across three repos:

- `spotlight-engine` (this repo) — spine and in-flight bundles (C, D, E, F).
- `spotlight-observability` — Bundle A: telemetry collection and signal extraction.
- `spotlight-knowledge` — Bundle B: literature index and experimental archive.

Dependency direction: `spotlight-engine` depends on `spotlight-observability` and `spotlight-knowledge`. The two leaf repos do not depend on each other and do not depend on the engine. The dependency graph is a tree, not a cycle.

See `discovery_engine_proposal.md` for the full architecture.

## Public API

Only the `schemas` module is part of the contract surface. Everything else is internal and may change without notice.

```python
from spotlights_engine.schemas import Candidate, Change
from spotlights_engine.schemas import TraceSummary, WorkloadProfile, Anomaly  # re-exported from spotlight-observability
```

## Getting started

```bash
git clone <this-repo>
cd spotlight-engine
uv sync --all-extras
uv run pytest
```

## Status

Stage 0 scaffolding. No functional code yet.
