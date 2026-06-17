"""Generated-config validation against the live evolver schemas (plan §9.5).

Gated behind availability so CI without the sibling repos / without jsonschema
still passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spotlights_engine.prep_evolve.adapters.nous import NousAdapter

from .test_adapters import _by_path, _spec

_SCHEMA_DIR = (
    Path(__file__).resolve().parents[3]
    / "other_repos"
    / "agentic-strategy-evolution"
    / "orchestrator"
    / "schemas"
)

jsonschema = pytest.importorskip("jsonschema")
_have_schemas = (_SCHEMA_DIR / "campaign.schema.yaml").exists()
pytestmark = pytest.mark.skipif(
    not _have_schemas, reason="agentic-strategy-evolution schemas not present"
)


def test_campaign_validates(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    files = _by_path(NousAdapter().render(spec))
    campaign = yaml.safe_load(files["campaign.yaml"].text)
    schema = yaml.safe_load((_SCHEMA_DIR / "campaign.schema.yaml").read_text())
    jsonschema.validate(campaign, schema)


def test_bundle_validates(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    files = _by_path(NousAdapter().render(spec))
    bundle = yaml.safe_load(files["bundle.yaml"].text)
    schema = yaml.safe_load((_SCHEMA_DIR / "bundle.schema.yaml").read_text())
    jsonschema.validate(bundle, schema)
