"""Unit tests for finding-related schemas."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.search import SearchQueryLog
from spotlights_engine.utils.id_helpers import parse_id, prefix_local_id


def test_finding_id_accepts_module_prefixed() -> None:
    # `finding_id` is a free-form string (the schema no longer enforces a
    # pattern); the manager mints the module-prefixed `find-<segment>-NNNN` form.
    finding = Finding(
        finding_id="find-attention-0001",
        title="Paged attention",
        url="https://example.com/paged-attention",
        source_type="paper",
        technique_summary="Use paged KV allocation to reduce fragmentation.",
    )
    assert finding.finding_id == "find-attention-0001"


def _finding(**overrides: Any) -> Finding:
    kwargs: dict[str, Any] = {
        "finding_id": "find-attention-0001",
        "title": "Paged attention",
        "url": "https://example.com/paged-attention",
        "source_type": "paper",
        "technique_summary": "Use paged KV allocation to reduce fragmentation.",
    }
    kwargs.update(overrides)
    return Finding(**kwargs)


# candidate_id (candidate deep-research mode) --------------------------------


def test_candidate_id_defaults_to_none_so_module_mode_sites_still_construct() -> None:
    assert _finding().candidate_id is None


def test_candidate_id_round_trips() -> None:
    finding = _finding(candidate_id="cand-v1_kv_offload-0002")

    reloaded = Finding.model_validate_json(finding.model_dump_json())

    assert reloaded.candidate_id == "cand-v1_kv_offload-0002"
    assert reloaded == finding


def test_candidate_id_rejects_a_non_candidate_id() -> None:
    with pytest.raises(ValidationError):
        _finding(candidate_id="find-v1_kv_offload-0002")


def test_search_query_log_candidate_id_round_trips() -> None:
    # `SearchQueryLog` uses extra="ignore", so an undeclared key would be
    # silently dropped; the field must survive a dump/load cycle.
    log = SearchQueryLog(
        agent="codex",
        query="paged attention",
        candidate_id="cand-v1_kv_offload-0002",
    )

    reloaded = SearchQueryLog.model_validate_json(log.model_dump_json())

    assert reloaded.candidate_id == "cand-v1_kv_offload-0002"


def test_search_query_log_candidate_id_defaults_to_none() -> None:
    assert SearchQueryLog(agent="codex", query="paged attention").candidate_id is None


# candidate-mode finding ids --------------------------------------------------


def test_candidate_mode_finding_id_parses_and_prefixes_idempotently() -> None:
    # `find-<module_segment>-<candidate_counter>-NNNN`: `parse_id` is
    # right-anchored, so the candidate counter stays part of the segment.
    fid = "find-v1_kv_offload-0002-0001"

    assert _finding(finding_id=fid).finding_id == fid

    id_type, segment, counter = parse_id(fid)
    assert (id_type, segment, counter) == ("find", "v1_kv_offload-0002", 1)
    assert prefix_local_id(fid, segment=segment, expected_type="find") == fid
