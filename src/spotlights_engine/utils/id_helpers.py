"""Repo-wide id helpers for the module-name-prefix id scheme.

Every minted id carries the producing module's slug, so ids are **globally
unique by construction** across parallel modules and remain unique when one
module produces several sessions within a run:

    <type>-<segment>-NNNN

    type    := cand | find | prop
    segment := <slug>[.s<k>]          # the module slug, optional `.s<k>` session
    NNNN    := \\d{4}                  # per-(module, session) counter, resets to 0001

Examples::

    cand-auth_login-0001              # module auth_login, first/only session
    find-auth_login-0003
    prop-auth_login-0007
    cand-auth_login.s2-0001           # second discovery session of the same module

Because both the slug and the counter are hyphen-delimited and the slug may
itself contain ``-``, ids are **parsed from the right**: the trailing
``-\\d{4}`` is the counter, the leading ``cand``/``find``/``prop`` is the type,
and everything between is the opaque ``segment``. Never split an id on ``-``
naively — route through :func:`parse_id`.

Producers (the discovery and deep-research agents) keep minting bare local ids
(``cand-0001``/``find-0001``); the manager hands each producer step its module
segment and the local ids are promoted to final prefixed ids via
:func:`prefix_local_id` *before* a persisted schema object is constructed. The
promotion is idempotent: an id that already carries the exact expected segment
passes through unchanged, which makes resume a no-op.
"""

from __future__ import annotations

import re

_TYPES = ("cand", "find", "prop")

# Characters allowed verbatim in a slug; everything else collapses to `_`. This
# matches the slug char class embedded in the schema id patterns
# (`[A-Za-z0-9_.-]`), so a `slug_for(qn)` value always parses back cleanly.
_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def slug_for(qualified_name: str) -> str:
    """Slug used both as the per-module directory name and as the id segment.

    Slash-form qualified name with any char outside `[A-Za-z0-9._-]` replaced by
    `_` (so path separators collapse to `_`). Slash-form is unique by
    construction, so two modules never share a slug; the manager still checks at
    filter-resolution time. This is the canonical slug source — `persistence`
    re-exports it for back-compat.
    """
    return _SLUG_SAFE.sub("_", qualified_name)

# Right-anchored: capture the leading type, the opaque middle (slug[.session]),
# and the trailing four-digit counter. The middle is `.+` (greedy stops at the
# last `-\d{4}`) so slugs containing `-` parse correctly.
_ID_RE = re.compile(r"^(cand|find|prop)-(.+)-(\d{4})$")


class IdAllocationError(ValueError):
    """Raised when an id is malformed or cannot be prefixed with a segment."""


def parse_id(id_str: str) -> tuple[str, str, int]:
    """Decompose a prefixed id into ``(type, segment, counter)``.

    ``parse_id("cand-auth_login.s2-0007")`` -> ``("cand", "auth_login.s2", 7)``.

    Parses from the right so a slug containing ``-`` (e.g. ``a-b_c``) is
    recovered intact as the opaque segment. The segment is **not** split into
    slug/session here — only :func:`module_segment` constructs the expected
    segment for a given module/session, which avoids ambiguity for slugs that
    legitimately contain ``.`` or look like ``.s2``.
    """
    m = _ID_RE.match(id_str)
    if m is None:
        raise IdAllocationError(f"malformed prefixed id: {id_str!r}")
    return m.group(1), m.group(2), int(m.group(3))


def module_segment(slug: str, session: int | None) -> str:
    """Build the id segment for a module/session.

    The first/only session uses the bare ``<slug>`` (``session`` is ``None`` or
    ``1``); subsequent sessions append ``.s<k>`` so two sessions of the same
    module never collide. ``slug`` must be non-empty.
    """
    if not slug:
        raise IdAllocationError("module slug must be non-empty")
    if session is None or session == 1:
        return slug
    if session < 1:
        raise IdAllocationError(f"session index must be >= 1, got {session}")
    return f"{slug}.s{session}"


def prefix_local_id(local_id: str, *, expected_type: str, segment: str) -> str:
    """Promote a bare local id to its module-prefixed final form.

    ``prefix_local_id("cand-0001", expected_type="cand", segment="auth_login")``
    -> ``"cand-auth_login-0001"``.

    Idempotent for resume: an id that already carries the exact expected
    ``(type, segment)`` is returned unchanged. Anything else — a wrong type, a
    different segment (segment-prefix collisions like ``a`` vs ``ab`` are
    rejected, not silently accepted), or a malformed id — raises.
    """
    if expected_type not in _TYPES:
        raise IdAllocationError(f"unknown id type {expected_type!r}")

    bare = re.fullmatch(rf"{expected_type}-(\d{{4}})", local_id)
    if bare is not None:
        return f"{expected_type}-{segment}-{bare.group(1)}"

    # Already prefixed? Accept only an exact-segment match (idempotent resume).
    parsed_type, parsed_segment, counter = parse_id(local_id)
    if parsed_type == expected_type and parsed_segment == segment:
        return local_id

    raise IdAllocationError(
        f"cannot prefix {local_id!r} as a {expected_type!r} id with segment "
        f"{segment!r}: it is neither the bare local form {expected_type}-NNNN "
        f"nor already carrying the expected segment"
    )


__all__ = [
    "IdAllocationError",
    "module_segment",
    "parse_id",
    "prefix_local_id",
    "slug_for",
]
