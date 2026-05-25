<!-- Stage 03 — Bundle C candidate-generation prompt.

Filled in step 5 of the implementation order. Per the approved plan,
this prompt MUST tell the model:

- `projecttree_by_path` and `read_module_source` are real tools — use them
  freely to drill into the implicated modules.

- `get_raw_trace` is currently UNAVAILABLE. A call returns the canonical
  "unavailable" JSON shape (status/reason/fallback) — treat it as
  non-fatal and continue using `TraceSummary` fields plus source drills.
  Do not retry. Do not abort the pass.

- The output is `list[Candidate]` matching `spotlights_engine.schemas.candidate.Candidate`
  (id pattern `^cand-\d{4}$`, no fixes proposed — findings only).

Inputs available in the prompt context: WorkloadProfile, TraceSummary[],
Anomaly[] (Bundle A output), and a serialized ProjectTree summary.
-->
