"""Shared numeric policy constants for the modules extractor.

Single home for the caps that prompt text, schema bounds, and validators must
agree on — interpolating these into prompts is what keeps prompt and schema
from drifting apart.

`MAX_MAIN_FILES` and `MAX_REPRESENTATIVE_FILES` are deliberately separate.
The first is a Stage-3 output constraint (how many reading-order files a
module may cite); the second is the shared cap for deterministic
prompt-reading seeds (Stage-2 per-node representatives and the Stage-3B
territory summary). They currently share the value 5 but are not the same
policy.
"""

from __future__ import annotations

# Assignment-contract size rule: a folder whose `subtree_source_file_count`
# exceeds this is a `MODULE`; at or below it, `MODULE` needs a `keep_reason`.
MERGE_THRESHOLD_DEFAULT = 15

# Stage-3 output constraint: at most this many `main_files` per module.
MAX_MAIN_FILES = 5

# Deterministic reading-seed cap (skeleton representatives, metadata scopes).
MAX_REPRESENTATIVE_FILES = 5

# Warning-only lint floor: a leaf module owning fewer source files than this
# is flagged `tiny_leaf` (and drives `fragmented_children`).
MIN_LEAF_FILES = 3

__all__ = [
    "MAX_MAIN_FILES",
    "MAX_REPRESENTATIVE_FILES",
    "MERGE_THRESHOLD_DEFAULT",
    "MIN_LEAF_FILES",
]
