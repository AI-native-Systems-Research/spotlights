"""Classification of agent-CLI streams.

Every fragment below is the shape the CLIs really emit, taken from the IOCR run's
persisted streams (`page_latency-2026-08-19-ophir-new-module-extractor`), because
the whole point of this module is that hand-imagined signatures are what produced
"schema parse failed twice" for eight rate limits.
"""

from __future__ import annotations

from spotlights_engine.utils.agent_stream import (
    describe,
    explain,
    retry_class,
    scan,
)

# --- real event shapes -------------------------------------------------------

# codex, on giving up: its own retry loop is already spent by this point.
CODEX_429 = (
    b'{"type":"error","message":"exceeded retry limit, last status: '
    b'429 Too Many Requests"}\n'
)

# claude, mid-flight. Note `retry_delay_ms`: 562ms of backoff, 10 attempts, then
# it stops -- which is why an engine-level retry has to wait far longer to help.
CLAUDE_RATE_LIMIT_RETRY = (
    b'{"type":"system","subtype":"api_retry","attempt":1,"max_retries":10,'
    b'"retry_delay_ms":562,"error_status":429,"error":"rate_limit"}\n'
)
CLAUDE_SERVER_ERROR_RETRY = (
    b'{"type":"system","subtype":"api_retry","attempt":2,"max_retries":10,'
    b'"retry_delay_ms":1200,"error_status":500,"error":"server_error"}\n'
)
CLAUDE_COMPACTING = b'{"type":"system","status":"compacting"}\n'


# claude, on a failure no backoff can fix: 401 is absent from the retryable
# status list on purpose.
CLAUDE_AUTH_RETRY = (
    b'{"type":"system","subtype":"api_retry","attempt":1,"max_retries":10,'
    b'"error_status":401,"error":"authentication_error"}\n'
)
# The CLI retrying without knowing why: `error_status` is null and the reason is
# the literal string "unknown". Verbatim from
# cand-document_understanding_workflow-0007.claude.stdout.
CLAUDE_UNKNOWN_RETRY = (
    b'{"type":"system","subtype":"api_retry","attempt":1,"max_retries":10,'
    b'"retry_delay_ms":528,"error_status":null,"error":"unknown"}\n'
)
CLAUDE_CLEAN_RESULT = (
    b'{"type":"result","subtype":"success","is_error":false,'
    b'"result":"{\\"candidates\\":[]}"}\n'
)

# The one failure no backoff can fix. Shape taken from an archived turn.failed.
QUOTA_EXHAUSTED = (
    b'{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. '
    b'Upgrade to Pro for more access. You can try again at Jul 1st, 2026 '
    b'12:00 AM."}}\n'
)

NOISE = b'{"type":"assistant","message":{"content":"thinking"}}\nnot json at all\n'


# --- describe ----------------------------------------------------------------


def test_describe_names_the_hard_error():
    assert describe(CODEX_429) == (
        "agent reported: exceeded retry limit, last status: 429 Too Many Requests"
    )


def test_describe_counts_retries_and_names_the_last_reason():
    stream = CLAUDE_RATE_LIMIT_RETRY * 3
    assert describe(stream) == "3 API retries in stream (last: rate_limit)"


def test_describe_reports_compaction():
    assert describe(CLAUDE_COMPACTING) == (
        "stream hit context compaction before finishing"
    )


def test_describe_falls_back_to_stderr_when_the_stream_says_nothing():
    # The 14 `exit=1, stderr=''` failures are this case with an empty stderr:
    # nothing at all to report, which is worth saying explicitly.
    assert describe(NOISE, b"") is None
    assert describe(b"", b"  connector warning: foo  ") == (
        'stderr tail: \'connector warning: foo\''
    )


def test_describe_ignores_unparseable_lines():
    assert describe(NOISE + CODEX_429) is not None


# --- retry_class -------------------------------------------------------------


def test_codex_429_is_retryable():
    assert retry_class(CODEX_429) == "retryable"


def test_claude_rate_limit_retries_that_never_recover_are_retryable():
    assert retry_class(CLAUDE_RATE_LIMIT_RETRY * 4) == "retryable"


def test_server_error_is_retryable():
    assert retry_class(CLAUDE_SERVER_ERROR_RETRY) == "retryable"


def test_quota_exhaustion_is_fatal_not_retryable():
    """The expensive mistake would be reading 'limit' and backing off.

    The account is out of budget until a date next month; every retry is a
    guaranteed-failed paid call.
    """
    assert retry_class(QUOTA_EXHAUSTED) == "fatal"


def test_compaction_is_fatal():
    assert retry_class(CLAUDE_COMPACTING) == "fatal"


def test_auth_failure_is_fatal():
    # 401 is not in the retryable status list, so it must not be retried -- this
    # is the WSL misconfiguration that masquerades as a schema failure.
    assert retry_class(CLAUDE_AUTH_RETRY) == "fatal"


def test_unknown_retry_reason_gets_the_benefit_of_the_doubt():
    """"unknown" is the CLI declining to classify, not a diagnosis.

    Contrast `test_auth_failure_is_fatal`: an unrecognised reason we can read is
    a reason not to retry, but the CLI only emits `api_retry` for conditions it
    believed were transient, so its own shrug must not be read as a dead end.
    Replaying the IOCR run is what surfaced this -- one stream whose sole
    evidence was this event came back `fatal`.
    """
    assert retry_class(CLAUDE_UNKNOWN_RETRY) == "retryable"


def test_unknown_reason_does_not_override_a_recorded_status():
    """A named status still decides, whatever the reason string says."""
    stream = (
        b'{"type":"system","subtype":"api_retry","attempt":1,"max_retries":10,'
        b'"error_status":401,"error":"unknown"}\n'
    )
    assert retry_class(stream) == "fatal"


def test_recovered_transient_does_not_rescue_a_trailing_auth_failure():
    """A 429 the stream survived must not vouch for the 401 that killed it.

    Found by review of the first Stage A commit. `retry_statuses` is cumulative,
    so feeding the whole list to the verdict let the historical 429 match the
    rate-limit pattern and return `retryable` -- retrying an auth failure
    forever, which is precisely what the fatal rule exists to prevent. The
    verdict reads the final retry event alone.
    """
    stream = CLAUDE_RATE_LIMIT_RETRY + CLAUDE_CLEAN_RESULT + CLAUDE_AUTH_RETRY
    assert scan(stream)["retry_statuses"] == [429, 401]  # history is still kept
    assert retry_class(stream) == "fatal"


def test_an_earlier_auth_failure_does_not_poison_a_trailing_rate_limit():
    """The converse: reading the tail must not mean "any fatal signal wins"."""
    stream = CLAUDE_AUTH_RETRY + CLAUDE_CLEAN_RESULT + CLAUDE_RATE_LIMIT_RETRY
    assert retry_class(stream) == "retryable"


def test_a_stale_reason_does_not_decide_for_a_later_retry():
    """`retry_reason` keeps the last reason it could read, for the human text.

    The verdict cannot use that: here the trailing retry names no reason at all,
    so the earlier "rate_limit" would be read as this failure's reason. Only the
    tail event's own 401 may decide.
    """
    unnamed_401 = (
        b'{"type":"system","subtype":"api_retry","attempt":2,"max_retries":10,'
        b'"error_status":401}\n'
    )
    stream = CLAUDE_RATE_LIMIT_RETRY + CLAUDE_CLEAN_RESULT + unnamed_401
    assert scan(stream)["retry_reason"] == "rate_limit"  # human text unchanged
    assert retry_class(stream) == "fatal"


def test_clean_stream_has_no_verdict():
    assert retry_class(CLAUDE_CLEAN_RESULT) is None
    assert retry_class(NOISE) is None
    assert retry_class(b"") is None


def test_retries_survived_then_a_shape_failure_is_not_retryable():
    """A recovered rate limit is history, not the cause.

    This is the trap `99626cc` had to fix once already: treating any stream that
    mentions a rate limit as salvageable hides a deterministic contract bug. Here
    the retries are followed by a clean terminal event, so the stream reports no
    failure at all and the caller's failure must be explained elsewhere.
    """
    stream = CLAUDE_RATE_LIMIT_RETRY * 2 + CLAUDE_CLEAN_RESULT
    assert retry_class(stream) is None


def test_hard_error_after_recovered_retries_is_judged_on_the_error():
    """The agent's own last word wins over earlier retry noise."""
    stream = (
        CLAUDE_RATE_LIMIT_RETRY
        + CLAUDE_CLEAN_RESULT
        + b'{"type":"error","message":"tool_use ids must be unique"}\n'
    )
    assert retry_class(stream) == "fatal"


def test_unrecognised_error_while_retries_in_flight_is_a_cutoff():
    stream = CLAUDE_RATE_LIMIT_RETRY + b'{"type":"error","message":"stream closed"}\n'
    assert retry_class(stream) == "retryable"


def test_codex_events_nested_under_msg_are_read():
    # codex --json nests some events; claude does not. Both must classify.
    stream = (
        b'{"id":"0","msg":{"type":"error","message":"exceeded retry limit, '
        b'last status: 429 Too Many Requests"}}\n'
    )
    assert retry_class(stream) == "retryable"


# --- scan / explain ----------------------------------------------------------


def test_scan_reports_nothing_for_a_healthy_stream():
    assert scan(CLAUDE_CLEAN_RESULT) == {}


def test_scan_collects_retry_statuses():
    found = scan(CLAUDE_RATE_LIMIT_RETRY + CLAUDE_SERVER_ERROR_RETRY)
    assert found["api_retries"] == 2
    assert found["retry_statuses"] == [429, 500]
    assert found["retries_at_tail"] is True


def test_explain_keeps_the_caller_message_as_a_prefix():
    """Anything grepping for the old text must keep matching."""
    out = explain("claude timed out after 600.0s", CLAUDE_RATE_LIMIT_RETRY * 10)
    assert out.startswith("claude timed out after 600.0s; ")
    assert "10 API retries in stream (last: rate_limit)" in out


def test_explain_is_unchanged_when_there_is_nothing_to_add():
    assert explain("codex exit=1: stderr=''", NOISE) == "codex exit=1: stderr=''"
