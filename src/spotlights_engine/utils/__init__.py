"""Shared, producer-agnostic helpers for the Spotlight Engine.

`schema_compat` isolates the read/write shape changes introduced when the
flat-location `Candidate` and the two split proposal lists were replaced by the
nested-location `Candidate` and a single unified `Proposal` list. Keeping these
accessors here (rather than under a producer package) lets every consumer —
renderer, cli, prep_evolve, the per-step modules — import them without taking a
dependency on a package they otherwise don't use.
"""
