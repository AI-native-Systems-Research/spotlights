# Module deep research

`spotlights_engine.module_deep_research` implements the per-module research step
for the Spotlights deep-research path. The review-critical contract is the pair
of Pydantic models:

- `ModuleDeepResearchInput`
- `ModuleDeepResearchOutput`

The step takes a `ProjectTree`, a target module qualified name, and caller
context. It resolves the module, builds the research prompt internally from the
repository and module fields, and returns source-backed findings plus any step
issues.

## Input

```python
from spotlights_engine.schemas import ModuleDeepResearchInput, SpotlightContext
from spotlights_engine.schemas.modules import ProjectTree

request = ModuleDeepResearchInput(
    project_tree=ProjectTree.from_json(project_tree_path),
    module_qualified_name="inference.attention",
    context=SpotlightContext(
        objective="reduce decode latency for long-context serving",
        workload_hints=["batch size 1-8", "single-node 8xH100"],
        validation_plan=["compare tokens/sec on the existing benchmark harness"],
    ),
    max_findings_per_module=10,
)
```

`SpotlightContext` is caller supplied. The module deep-research step reads
`objective` and `workload_hints` to filter sources toward the run goal. It does
not mutate the context and does not pass context to module extraction.

## Output

`research_module()` returns the architecture contract object
`ModuleDeepResearchOutput`. Callers should treat that object, not an artifact
folder or free-form text, as the step output.

```python
from spotlights_engine.module_deep_research import research_module
from spotlights_engine.schemas import ModuleDeepResearchOutput

output = research_module(request)
assert isinstance(output, ModuleDeepResearchOutput)

for finding in output.findings:
    print(finding.finding_id, finding.title, finding.source_type, finding.url)
for issue in output.issues:
    print(issue.severity, issue.message)
```

The returned object has this shape:

```python
ModuleDeepResearchOutput(
    findings=[
        Finding(
            finding_id="find-0001",
            title="Source-backed optimization technique",
            url="https://example.com/source",
            source_type="paper",
            technique_summary="Short summary of the technique and why it applies.",
            supporting_evidence="Brief excerpt, paraphrase, or source note.",
        )
    ],
    issues=[],
)
```

`ModuleDeepResearchOutput.findings` may be empty. Empty findings means the step
ran but found no supportable source for the target module. Recoverable or fatal
problems are represented as `StepIssue` entries.

## Qualified names

`ProjectTree` currently uses slash-qualified module names. The module-deep-
research API also accepts dot-qualified names from the architecture document and
normalizes them for lookup, so both of these resolve to the same nested module
when present:

- `inference/attention`
- `inference.attention`

## Offline validation

The unit-testable path does not call live web search or Codex. It validates:

- schema shape and defaults,
- prompt construction from repository/module/context fields,
- parsing and normalization of `ModuleDeepResearchOutput`,
- missing-module and nonzero-runner issue handling.

Run the focused checks with:

```bash
PYTHONPATH=src pytest tests/unit/schemas/test_deep_research.py tests/unit/module_deep_research -q
```
