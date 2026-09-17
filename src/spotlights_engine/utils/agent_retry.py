"""Backoff policy for relaunching an agent CLI that a rate limit cut off.

Why this exists
---------------
On the IOCR run (`page_latency-2026-08-19-ophir-new-module-extractor`) the agent
CLIs took 149 rate limits, and their own retry loops were not built for a window
that lasts minutes: 169 retry events, a **median backoff of 601 ms**, and 316
seconds of deliberate waiting across a 4.3-hour run. Against that, 30 step-5
candidates each burned their full 600-second wall and were then reported as
timeouts. Five hours lost to timeouts against five minutes spent waiting on
purpose.

Measurement ruled out the alternative fix. Neither the run's call count nor its
token throughput predicts the refusals -- correlation is negative at every bucket
width, the minutes that took a 429 were *lighter* than the minutes that did not
(209,545 vs 244,189 tok/min), and a live ramp to 64 genuinely concurrent requests
against the same gateway drew no refusal at all. So the engine cannot out-pace
these 429s by launching fewer calls; it can only outlast them.

Sizing, from those numbers rather than from taste
------------------------------------------------
- **base 30 s** -- fifty times the ~0.6 s the CLI already tried and failed with.
  A retry that waits about as long as the last one adds cost and changes nothing.
- **cap 300 s** -- the observed spikes persisted for minutes, and the longest
  single backoff any CLI chose for itself was 32.7 s.
- **full jitter** -- `uniform(0, ceiling)`, not `ceiling`. Several candidates run
  in parallel and are throttled by the same event, so a fixed delay would march
  them back in lockstep and rebuild the very burst we are waiting out.

Opt-in: `attempts` defaults to 1, which is exactly today's behaviour and takes
the same code path. Nothing changes for a run that does not ask for it.

This module holds no subprocess or stream-parsing code. It decides *whether* and
*how long*, reading the verdict from `utils.agent_stream.retry_class`, so the
classification table stays in one place and this policy stays testable without
launching anything.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from spotlights_engine.utils.agent_stream import retry_class

# Defaults, justified in the module docstring. Exposed as constants so the config
# layer and the tests quote one source rather than three.
DEFAULT_ATTEMPTS = 1
DEFAULT_BASE_S = 30.0
DEFAULT_MAX_S = 300.0


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to relaunch a rate-limited agent call, and how long to wait.

    `attempts` counts *total* launches, not retries, so 1 means "no retry" and is
    the default. Frozen because a policy is recorded in `run_manifest.json` and
    must describe what the run actually did.
    """

    attempts: int = DEFAULT_ATTEMPTS
    base_s: float = DEFAULT_BASE_S
    max_s: float = DEFAULT_MAX_S

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be >= 1, got {self.attempts}")
        if self.base_s <= 0 or self.max_s <= 0:
            raise ValueError(f"delays must be > 0, got base={self.base_s} max={self.max_s}")
        if self.max_s < self.base_s:
            raise ValueError(f"max_s {self.max_s} is below base_s {self.base_s}")

    @property
    def enabled(self) -> bool:
        return self.attempts > 1

    def delay_s(self, attempt: int, *, rand: random.Random | None = None) -> float:
        """Full-jitter exponential wait before `attempt`, which is 1-based.

        `attempt` is the number of the launch about to be made, so the first
        retry is attempt 2 and waits within [0, base_s].
        """
        if attempt < 2:
            raise ValueError(f"attempt {attempt} needs no delay; retries start at 2")
        ceiling = min(self.max_s, self.base_s * 2 ** (attempt - 2))
        return (rand or random).uniform(0.0, ceiling)


def should_retry(
    stdout: bytes,
    stderr: bytes = b"",
    *,
    policy: RetryPolicy,
    attempt: int,
) -> bool:
    """Whether a failed launch earns another one.

    The whole decision is delegated to `retry_class`, which is what keeps the
    two rules the archive demands from having to be restated here:

    - a **timeout is not by itself retryable**. `retry_class` returns None for a
      clean stream, so a call that was merely slow is never relaunched -- only
      one whose stream carries rate-limit or transient-server evidence is.
    - **quota exhaustion is fatal.** No wait inside a run fixes an account that
      is out of budget until a date next month, so it classifies `fatal` and
      stops here rather than burning the remaining attempts.
    """
    if attempt >= policy.attempts:
        return False
    return retry_class(stdout, stderr) == "retryable"


__all__ = [
    "DEFAULT_ATTEMPTS",
    "DEFAULT_BASE_S",
    "DEFAULT_MAX_S",
    "RetryPolicy",
    "should_retry",
]
