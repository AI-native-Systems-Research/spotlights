"""Classification of agent-CLI output streams.

Both agent CLIs emit newline-delimited JSON events and both are capable of
dying without saying so in their exit status or on stderr. The engine used to
describe those deaths by whatever the *caller* happened to know -- step 2 said
"schema parse failed twice" and step 5 says "timed out after 600.0s" -- and on
the IOCR run every one of those labels was wrong: the streams show rate limits.
The terminal event is in the stream, so read the stream.

This module deliberately contains no subprocess, concurrency or retry code. It
turns bytes into a description or a verdict and nothing else, which is what lets
it be shared by every step's failure path without changing how any call is made.

Two entry points:

- `describe` -- a human one-line cause, for error messages and `status.json`.
- `retry_class` -- whether the failure is worth another attempt. Nothing calls
  this yet; it is here so the classification table can be reviewed and tested
  before any code starts acting on it.
"""

from __future__ import annotations

import json
import re
from typing import Literal

RetryClass = Literal["retryable", "fatal"]

# Quota exhaustion. Checked *before* the rate-limit patterns and deliberately so:
# a quota message may also carry a 429, but no amount of backoff inside a single
# run can fix an account that is out of budget until a date next month. An
# archived run holds the real shape of this event:
#   {"type":"turn.failed","error":{"message":"You've hit your usage limit ...
#    Upgrade to Pro ... try again at Jul 1st, 2026 12:00 AM"}}
_QUOTA_RE = re.compile(
    r"usage limit|quota exceeded|upgrade to (?:pro|max)|try again at", re.IGNORECASE
)

# Rate limiting. `exceeded retry limit, last status: 429 Too Many Requests` is
# codex's wording; claude reports `"error":"rate_limit"` on its retry events.
_RATE_LIMIT_RE = re.compile(
    r"rate.?limit|too many requests|\b429\b", re.IGNORECASE
)

# Transient server-side failures. The numeric alternation lists only the codes
# worth retrying, so anything else that shows up as a status (401 auth, 400 bad
# request) falls through to `fatal` rather than being retried forever.
_TRANSIENT_RE = re.compile(
    r"server_error|overloaded|internal server error|bad gateway"
    r"|service unavailable|gateway timeout|\b(?:408|500|502|503|504|529)\b",
    re.IGNORECASE,
)

# What a CLI says when it retried without knowing why. The IOCR run's streams
# carry `{"error_status":null,"error":"unknown"}` on 4 of their retry events:
# that is the CLI declining to classify, not a diagnosis, so it must not be read
# as a reason *not* to retry the way `authentication_error` is.
_UNINFORMATIVE_REASONS = frozenset({"unknown", "none", "unspecified"})


def scan(stdout: bytes) -> dict[str, object]:
    """Collect the terminal-failure signals present in an agent stream.

    Returns only the keys it actually found, so an empty dict means "this stream
    contains no evidence of trouble".
    """
    found: dict[str, object] = {}
    api_retries = 0
    statuses: list[int] = []
    # Event ordering matters for one decision: a stream that retried and then
    # finished cleanly is healthy, while one whose last word was a retry was cut
    # off mid-flight. Track where each was last seen rather than merely whether.
    last_retry_i = -1
    last_clean_i = -1
    # `retry_reason` below is the last reason we could *read*, which is what the
    # human text wants. A verdict needs something stricter: the reason and status
    # of the final retry event alone. Keeping the cumulative list and reusing it
    # for the verdict let a recovered 429 earlier in the stream rescue a trailing
    # 401 -- i.e. retry an auth failure forever, the exact outcome the fatal rule
    # exists to prevent. So track the tail event separately, resetting both on
    # every retry so nothing carries over from one that was survived.
    last_retry_reason: str | None = None
    last_retry_status: int | None = None

    for i, raw in enumerate(stdout.splitlines()):
        raw = raw.strip()
        if not raw or not raw.startswith(b"{"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # codex --json nests some events under "msg"; claude does not.
        for event in (obj, obj.get("msg")):
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            if etype == "error":
                message = event.get("message")
                if isinstance(message, str) and message:
                    found["hard_error"] = message
            elif etype == "turn.failed":
                err = event.get("error")
                message = err.get("message") if isinstance(err, dict) else err
                if isinstance(message, str) and message:
                    found["hard_error"] = message
            elif etype == "turn.completed":
                last_clean_i = i
            elif etype == "result":
                if event.get("is_error"):
                    found["result_error"] = str(
                        event.get("result") or event.get("subtype") or "unspecified"
                    )
                else:
                    last_clean_i = i
            elif etype == "system":
                if event.get("subtype") == "api_retry":
                    api_retries += 1
                    last_retry_i = i
                    reason = event.get("error")
                    named = isinstance(reason, str) and bool(reason)
                    if named:
                        found["retry_reason"] = reason
                    last_retry_reason = reason if named else None
                    status = event.get("error_status")
                    if isinstance(status, int):
                        statuses.append(status)
                        last_retry_status = status
                    else:
                        last_retry_status = None
                if event.get("status") == "compacting":
                    found["compacting"] = True

    if api_retries:
        found["api_retries"] = api_retries
    if statuses:
        found["retry_statuses"] = statuses
    if last_retry_reason is not None:
        found["last_retry_reason"] = last_retry_reason
    if last_retry_status is not None:
        found["last_retry_status"] = last_retry_status
    if api_retries and last_retry_i > last_clean_i:
        found["retries_at_tail"] = True
    return found


def describe(stdout: bytes, stderr: bytes = b"") -> str | None:
    """Best-effort one-line explanation of why an agent produced no output."""
    found = scan(stdout)
    parts: list[str] = []
    if isinstance(found.get("hard_error"), str):
        parts.append(f"agent reported: {found['hard_error']}")
    elif isinstance(found.get("result_error"), str):
        parts.append(f"agent result was an error: {found['result_error']}")
    retries = found.get("api_retries")
    if isinstance(retries, int):
        reason = found.get("retry_reason")
        suffix = f" (last: {reason})" if isinstance(reason, str) else ""
        parts.append(f"{retries} API retries in stream{suffix}")
    if found.get("compacting"):
        parts.append("stream hit context compaction before finishing")
    if not parts:
        tail = stderr[-300:].decode("utf-8", "replace").strip()
        if tail:
            parts.append(f"stderr tail: {tail!r}")
    return "; ".join(parts) or None


def explain(message: str, stdout: bytes, stderr: bytes = b"") -> str:
    """`message`, with the stream's own account of the failure appended.

    The caller's message stays the prefix so anything that greps for the old
    text ("timed out after", "exit=") keeps matching.
    """
    cause = describe(stdout, stderr)
    return f"{message}; {cause}" if cause else message


def retry_class(stdout: bytes, stderr: bytes = b"") -> RetryClass | None:
    """Whether a failed agent call is worth another attempt.

    `None` means the stream carries no evidence either way -- the caller should
    treat that as "do not retry", but it is reported distinctly from `fatal` so a
    silent stream is never mistaken for a diagnosed dead end.

    Note both CLIs already retry internally, fast (claude backs off ~562ms) and
    shallow (10 attempts) before reporting failure. A `retryable` verdict is
    therefore only actionable by a caller prepared to wait far longer than that.
    """
    found = scan(stdout)
    if not found:
        return None

    def _classify(text: str) -> RetryClass | None:
        # Quota first: it can carry a 429 of its own, and it is the one failure
        # where retrying is guaranteed to waste money.
        if _QUOTA_RE.search(text):
            return "fatal"
        if _RATE_LIMIT_RE.search(text) or _TRANSIENT_RE.search(text):
            return "retryable"
        return None

    def _tail_verdict() -> RetryClass | None:
        """The verdict from retries the stream never recovered from.

        `None` when the retries were survived: a call that retried, completed its
        turn and then failed on output shape is a contract problem, and calling
        that retryable is exactly the over-broad salvage that hid a real bug once
        already.
        """
        if not found.get("retries_at_tail"):
            return None
        tail_parts: list[str] = []
        reason = found.get("last_retry_reason")
        if (
            isinstance(reason, str)
            and reason.strip().lower() not in _UNINFORMATIVE_REASONS
        ):
            tail_parts.append(reason)
        status = found.get("last_retry_status")
        if status is not None:
            tail_parts.append(str(status))
        tail = " ".join(tail_parts)
        verdict = _classify(tail)
        if verdict is not None:
            return verdict
        # A reason we can read but do not recognise as transient is a reason
        # *not* to retry -- an authentication_error/401 retried forever is a
        # paid call that cannot ever succeed. A cut-off that recorded nothing,
        # or nothing better than `unknown`, gets the benefit of the doubt, and
        # then only because the CLI itself emits `api_retry` solely for
        # conditions it believed were transient.
        return "fatal" if tail.strip() else "retryable"

    reported = " ".join(
        str(found[key]) for key in ("hard_error", "result_error") if key in found
    )
    if reported:
        # The agent said why it died, so that is the answer -- whatever happened
        # earlier in the stream is history it evidently survived.
        verdict = _classify(reported)
        if verdict is not None:
            return verdict
        # An unrecognised error that arrived while retries were still in flight
        # is a cut-off rather than a considered failure.
        return _tail_verdict() or "fatal"

    if found.get("compacting"):
        return "fatal"

    return _tail_verdict()


__all__ = ["RetryClass", "describe", "explain", "retry_class", "scan"]
