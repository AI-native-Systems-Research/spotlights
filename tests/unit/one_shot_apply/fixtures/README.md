# `apply` manifest fixtures

Two `manifest.json` instances, one fully-priced model each, used by
`test_usage_manifest.py` to catch schema drift: a field renamed in
`ApplyUsageManifest` or a key left behind here fails `model_validate_json`
because the model sets `extra="forbid"`.

**The token counts are synthetic.** They were written by hand and fed through
`build_apply_usage_manifest`, so the dollar figures are a faithful computation
of them at the committed rate tables — but nothing measured the tokens. No
apply run produced these numbers.

The provenance fields *are* real, copied from
`examples/vllm_subset/run_manifest.json`: both commit SHAs, `repo_url`,
`objective`, and each `candidate_id` / `module_qualified_name` pair. The two
candidates named here have real sibling artifacts under
`examples/vllm_subset/apply/`, from a run that predates this manifest.

These lived under `examples/` until it became clear that a synthetic record
beside three genuine artifacts reads as a measurement. Do not move them back:
`examples/` is where someone looks to see what a real run produced, which is
the one contract invented numbers cannot keep.
