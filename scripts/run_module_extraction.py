"""Run modules_extractor against the vllm repo.

Usage:
    uv run --no-sync python scripts/run_module_extraction.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from spotlights_engine.modules_extractor import (
    ExtractorConfig,
    extract_with_telemetry,
)
from spotlights_engine.schemas.pipeline import ModulesExtractorInput

REPO_PATH = Path("/Users/ophir/PycharmProjects/vllm/vllm")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "tmp" / "module_extraction"
OUTPUT_PATH = ARTIFACTS_DIR / "vllm_modules.json"


def main() -> None:
    run_dir = ARTIFACTS_DIR / "modules_extractor"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    inp = ModulesExtractorInput(repo_path=REPO_PATH)
    cfg = ExtractorConfig(artifacts_dir=ARTIFACTS_DIR)

    result = extract_with_telemetry(inp, config=cfg)
    result.project_tree.to_json(OUTPUT_PATH)

    inv = result.invocation
    print(f"modules: {len(result.project_tree.modules)}")
    print(f"duration_s: {inv.duration_s:.1f}")
    print(f"cost_usd: {inv.cost_usd}")
    print(f"input_tokens: {inv.input_tokens}")
    print(f"output_tokens: {inv.output_tokens}")
    print(f"output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
