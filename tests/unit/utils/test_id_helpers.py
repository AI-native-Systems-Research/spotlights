"""Unit tests for the module-name-prefix id helpers (decision D3)."""

from __future__ import annotations

import pytest

from spotlights_engine.utils.id_helpers import (
    IdAllocationError,
    module_segment,
    parse_id,
    prefix_local_id,
    slug_for,
)


# --- slug_for ---------------------------------------------------------------


def test_slug_for_replaces_unsafe_chars() -> None:
    assert slug_for("v1/kv_offload") == "v1_kv_offload"
    assert slug_for("v1/attention/paged_kv") == "v1_attention_paged_kv"
    assert slug_for("a.b c") == "a.b_c"


# --- parse_id (right-anchored) ----------------------------------------------


def test_parse_id_recovers_type_segment_counter() -> None:
    assert parse_id("cand-auth_login-0001") == ("cand", "auth_login", 1)
    assert parse_id("find-auth_login.s2-0007") == ("find", "auth_login.s2", 7)
    assert parse_id("prop-v1_kv_offload-0123") == ("prop", "v1_kv_offload", 123)


def test_parse_id_handles_slug_with_hyphen() -> None:
    # Parse from the right: the last `-\d{4}` is the counter, so a slug
    # containing '-' is recovered intact as the opaque segment.
    assert parse_id("cand-a-b_c-0010") == ("cand", "a-b_c", 10)


@pytest.mark.parametrize(
    "bad",
    ["cand-0001", "cand--0001", "cand-slug-12", "candidate-slug-0001", "cand-slug-001a"],
)
def test_parse_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(IdAllocationError):
        parse_id(bad)


# --- module_segment ---------------------------------------------------------


def test_module_segment_first_session_is_bare_slug() -> None:
    assert module_segment("auth_login", None) == "auth_login"
    assert module_segment("auth_login", 1) == "auth_login"


def test_module_segment_subsequent_session_appends_suffix() -> None:
    assert module_segment("auth_login", 2) == "auth_login.s2"
    assert module_segment("auth_login", 5) == "auth_login.s5"


def test_module_segment_rejects_empty_slug_or_bad_session() -> None:
    with pytest.raises(IdAllocationError):
        module_segment("", 1)
    with pytest.raises(IdAllocationError):
        module_segment("slug", 0)


# --- prefix_local_id --------------------------------------------------------


def test_prefix_local_id_inserts_segment() -> None:
    assert (
        prefix_local_id("cand-0001", expected_type="cand", segment="auth_login")
        == "cand-auth_login-0001"
    )
    assert (
        prefix_local_id("find-0007", expected_type="find", segment="auth_login.s2")
        == "find-auth_login.s2-0007"
    )


def test_prefix_local_id_is_idempotent_on_exact_segment() -> None:
    once = prefix_local_id("cand-0001", expected_type="cand", segment="auth_login")
    assert (
        prefix_local_id(once, expected_type="cand", segment="auth_login")
        == "cand-auth_login-0001"
    )


def test_prefix_local_id_rejects_segment_prefix_collision() -> None:
    # `ab` already-prefixed must NOT pass through as if it were segment `a`.
    already = "cand-ab-0001"
    with pytest.raises(IdAllocationError):
        prefix_local_id(already, expected_type="cand", segment="a")


def test_prefix_local_id_rejects_wrong_type() -> None:
    with pytest.raises(IdAllocationError):
        prefix_local_id("find-0001", expected_type="cand", segment="auth_login")
    with pytest.raises(IdAllocationError):
        prefix_local_id("cand-0001", expected_type="bogus", segment="auth_login")
