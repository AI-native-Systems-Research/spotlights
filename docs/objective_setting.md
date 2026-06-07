# Objective Setting Skill

The objective-setting skill is the starting point of a Discovery Engine session. It interviews a Product Manager to establish what to optimize, then produces a frozen Objective that all downstream modules use to scope their work.

## How to use

After installing the package and running `spotlights-engine init` (see [README → Install the Spotlights skill](../README.md#install-the-spotlights-skill)), the slash command is available in Claude Code:

```
/spotlights-objective-setting
```

Claude will walk you through a short interview (typically 2-3 minutes). You'll be asked:

1. **What metric to optimize** — e.g., request latency (p99), throughput, time to first token, GPU memory usage
2. **Direction** — minimize or maximize
3. **Workload classes** — what kind of usage matters (agentic, batch-inference, long-context, mixed)
4. **Priorities** (optional) — secondary concerns for ranking optimizations (e.g., memory efficiency, code simplicity)
5. **Notes** (optional) — any context about why this matters or what's driving the goal

No technical knowledge is required. You don't need to know about tests, benchmarks, code paths, or components — the system figures that out autonomously.

## What happens next

After collecting your answers, the skill assembles a proposal and presents it back to you in plain language. You can:

- **Approve** — locks the objective for the session
- **Adjust** — change workload classes, priorities, or other inputs and see a revised proposal

Once approved, the skill produces an `objective.json` file in the current directory. This is the immutable objective for the session.

## Output format

The `objective.json` file contains:

```json
{
  "objective_id": "uuid",
  "session_id": "uuid",
  "created_at": "2026-05-24T12:00:00Z",
  "approved_by": "your name",
  "intent": {
    "target_metric": "request_latency_p99",
    "target_direction": "minimize",
    "workload_classes": ["agentic"],
    "priorities": ["memory_efficiency"],
    "notes": ""
  },
  "rationale": "This session will focus on reducing request_latency_p99 for agentic workloads...",
  "frozen": true
}
```

## How downstream modules use it

- **Bundle C (candidate generation)** — uses `target_metric` and `workload_classes` to decide what to optimize and where to look
- **Bundle E (validation)** — uses the objective as the evaluation frame; determines appropriate tests autonomously; enforces non-regression on all metrics as a system invariant
- **Bundle F (orchestration)** — distributes the objective and manages session lifecycle

## Non-regression guarantee

You do not need to specify what must not regress. The system enforces a universal invariant: **no change may regress any metric**. This is automatic and unconditional.

## Programmatic usage

The module can also be used directly from Python:

```python
from spotlights_engine.objectives import build_intent, assemble_proposal, finalize_objective

intent = build_intent(
    target_metric="request_latency_p99",
    target_direction="minimize",
    workload_classes=["agentic"],
)
proposal = assemble_proposal(intent)
objective = finalize_objective(proposal, session_id="sess-001", approved_by="alice")
```

Or via CLI:

```bash
spotlights-objectives build-intent '{"target_metric": "request_latency_p99", "target_direction": "minimize", "workload_classes": ["agentic"]}'
```
