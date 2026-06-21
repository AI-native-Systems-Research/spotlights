from __future__ import annotations

from spotlights_engine.modules_extractor.prompts import EXTRACTION_PROMPT


def test_extraction_prompt_forbids_conceptual_submodule_paths() -> None:
    assert "Do not split a directory into conceptual children" in EXTRACTION_PROMPT
    assert (
        "Submodule `path` must be a real nested path below its parent"
        in EXTRACTION_PROMPT
    )
