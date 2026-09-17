"""Step 5 relaunching a rate-limited agent call.

The IOCR run lost 44 candidates in this step and reported 30 of them as
"claude timed out after 600.0s" while their streams said `rate_limit`. These
tests pin the two halves of the fix: the call is retried when the stream proves
a rate limit, and it is *not* retried for anything else.

Delays are configured to milliseconds rather than patched out, so the real
backoff path -- including the jitter -- executes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine.agent_proposals import (
    AgentProposalsConfig,
    create_agent_proposals_with_telemetry,
)
from spotlights_engine.agent_proposals.claude_exec import CandidateAgentRunResult
from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.utils.schema_compat import proposals_from
from tests.unit.agent_proposals._fakes import (
    fake_codex_runner_factory,
    make_input,
    make_proposal_payload,
)

CAND = "cand-v1_kv_offload-0001"

# Real captures from the archived run.
RATE_LIMITED = (
    b'{"type":"system","subtype":"api_retry","attempt":10,"max_retries":10,'
    b'"retry_delay_ms":562,"error_status":429,"error":"rate_limit"}\n'
)
QUOTA_GONE = (
    b'{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. '
    b'Upgrade to Pro for more access. You can try again at Jul 1st, 2026 '
    b'12:00 AM."}}\n'
)
JUST_SLOW = b'{"type":"assistant","message":{"content":"still working"}}\n'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    a = tmp_path / "artifacts"
    a.mkdir()
    return a


def _claude_runner(streams: list[bytes], *, calls: list[int]):
    """Fails with each stream in turn, then succeeds. `calls` records launches."""

    def _runner(*, candidate_id: str, **_: object) -> CandidateAgentRunResult:
        calls.append(1)
        i = len(calls) - 1
        if i < len(streams):
            return CandidateAgentRunResult(
                candidate_id=candidate_id,
                duration_s=0.001,
                error=f"claude timed out after 600.0s (attempt {i + 1})",
                stdout=streams[i],
                stderr=b"",
            )
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=0.001,
            structured_output=make_proposal_payload(agent_name="claude"),
        )

    return _runner


def _cfg(repo: Path, artifacts: Path, **kw: object) -> AgentProposalsConfig:
    return AgentProposalsConfig(
        repo_path=repo,
        artifacts_dir=artifacts,
        agent_retry_base_s=0.001,
        agent_retry_max_s=0.002,
        **kw,  # type: ignore[arg-type]
    )


def _run(inp, cfg, claude_runner):
    return create_agent_proposals_with_telemetry(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=fake_codex_runner_factory(
            payloads={CAND: make_proposal_payload(agent_name="codex")}
        ),
    )


def test_a_rate_limited_call_is_relaunched_and_the_proposal_survives(
    repo: Path, artifacts: Path
) -> None:
    """The outcome the whole change exists for: a 429 costs a wait, not a
    candidate. In the archive this candidate was simply lost."""
    calls: list[int] = []
    result = _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts, agent_retry_attempts=3),
        _claude_runner([RATE_LIMITED], calls=calls),
    )
    assert len(calls) == 2
    proposals = proposals_from(
        result.output.candidates.candidates[0], "agent_knowledge"
    )
    assert {p.author for p in proposals} == {"claude", "codex"}
    assert [i for i in result.output.issues if i.severity == "error"] == []


def test_off_by_default(repo: Path, artifacts: Path) -> None:
    """Backoff makes a bad run longer, so it must not arrive by surprise."""
    calls: list[int] = []
    result = _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts),
        _claude_runner([RATE_LIMITED], calls=calls),
    )
    assert len(calls) == 1
    assert any("timed out" in i.message for i in result.output.issues)


def test_quota_exhaustion_is_not_waited_out(repo: Path, artifacts: Path) -> None:
    """No wait inside a run fixes an account out of budget until next month."""
    calls: list[int] = []
    _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts, agent_retry_attempts=3),
        _claude_runner([QUOTA_GONE], calls=calls),
    )
    assert len(calls) == 1


def test_a_genuinely_slow_call_is_not_relaunched(repo: Path, artifacts: Path) -> None:
    """Same error text as the rate-limited case -- the stream is the only thing
    that separates them, which is exactly what the archive taught."""
    calls: list[int] = []
    _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts, agent_retry_attempts=3),
        _claude_runner([JUST_SLOW], calls=calls),
    )
    assert len(calls) == 1


def test_attempts_are_bounded_and_every_attempt_keeps_its_evidence(
    repo: Path, artifacts: Path
) -> None:
    """A rate limit that never clears must still terminate, and each superseded
    attempt has to leave its stream on disk -- a retry that hid the 429s would
    recreate the blind spot this whole branch is about."""
    calls: list[int] = []
    result = _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts, agent_retry_attempts=3),
        _claude_runner([RATE_LIMITED] * 9, calls=calls),
    )
    assert len(calls) == 3

    debug = artifacts / "agent_proposals.last_messages"
    kept = sorted(p.name for p in debug.glob(f"{CAND}.claude*.stdout"))
    assert kept == [
        f"{CAND}.claude.attempt1.stdout",
        f"{CAND}.claude.attempt2.stdout",
        f"{CAND}.claude.stdout",
    ]
    assert (debug / f"{CAND}.claude.attempt1.stdout").read_bytes() == RATE_LIMITED
    # And the failure still reaches the caller rather than being swallowed.
    assert any("rate_limit" in i.message or "timed out" in i.message for i in result.output.issues)


def _paying_claude_runner(streams: list[bytes], *, calls: list[int]):
    """Like `_claude_runner`, but every attempt reports what it burned.

    Not a hypothetical: the failing branches of `claude_exec` and `codex_exec`
    both parse usage out of the partial stream, so a 429 that arrived after the
    prompt was read has already been paid for.
    """

    def _runner(*, candidate_id: str, **_: object) -> CandidateAgentRunResult:
        calls.append(1)
        i = len(calls) - 1
        usage = AgentUsage(input=100, output=10, api_time_s=2.0, model="m")
        if i < len(streams):
            return CandidateAgentRunResult(
                candidate_id=candidate_id,
                duration_s=5.0,
                error="claude timed out after 600.0s",
                stdout=streams[i],
                stderr=b"",
                usage=usage,
            )
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=5.0,
            structured_output=make_proposal_payload(agent_name="claude"),
            usage=usage,
        )

    return _runner


def test_tokens_burned_by_a_superseded_attempt_are_still_billed(
    repo: Path, artifacts: Path
) -> None:
    """Three launches, three bills.

    The caller turns one `CandidateAgentRunResult` into this candidate's usage
    record, so a retry that returned only the winning attempt would quietly
    under-report cost -- and cost is the number this engine quotes to people.
    """
    calls: list[int] = []
    result = _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts, agent_retry_attempts=3),
        _paying_claude_runner([RATE_LIMITED, RATE_LIMITED], calls=calls),
    )
    assert len(calls) == 3

    claude = [u for u in result.usages_by_candidate[CAND] if u.cli == "claude"]
    assert len(claude) == 1
    assert (claude[0].usage.input, claude[0].usage.output) == (300, 30)
    assert claude[0].usage.api_time_s == 6.0
    assert claude[0].usage.model == "m"


def test_time_spent_on_superseded_attempts_is_still_reported(
    repo: Path, artifacts: Path
) -> None:
    """A candidate that took three attempts did not take one attempt's time.

    `duration_s` feeds the manifest's `accumulated_duration_s`; dropping two
    thirds of it would make a run that waited on purpose look fast.
    """
    calls: list[int] = []
    result = _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts, agent_retry_attempts=3),
        _paying_claude_runner([RATE_LIMITED, RATE_LIMITED], calls=calls),
    )
    assert result.per_candidate_durations_s[CAND]["claude"] >= 15.0


def test_a_call_that_needed_no_retry_is_returned_untouched(
    repo: Path, artifacts: Path
) -> None:
    """The default path must be byte-for-byte what it was before the retry."""
    calls: list[int] = []
    result = _run(
        make_input(n_candidates=1),
        _cfg(repo, artifacts, agent_retry_attempts=3),
        _paying_claude_runner([], calls=calls),
    )
    assert len(calls) == 1
    claude = [u for u in result.usages_by_candidate[CAND] if u.cli == "claude"]
    assert (claude[0].usage.input, claude[0].usage.api_time_s) == (100, 2.0)
    assert result.per_candidate_durations_s[CAND]["claude"] == 5.0


def test_a_cap_below_the_base_delay_is_refused(repo: Path, artifacts: Path) -> None:
    """Clamping it instead would run a policy the manifest does not record."""
    with pytest.raises(ValidationError):
        AgentProposalsConfig(
            repo_path=repo,
            artifacts_dir=artifacts,
            agent_retry_attempts=3,
            agent_retry_base_s=60.0,
            agent_retry_max_s=30.0,
        )
