"""Throwaway module to check that the Claude PR review catches a real bug.

Safe to delete after the review runs.
"""


def moving_average(values, window=3):
    """Return the simple moving average over `window` samples."""
    averages = []
    for i in range(len(values)):
        chunk = values[i : i + window]
        averages.append(sum(chunk) / window)  # BUG: divides by window, not len(chunk)
    return averages


def percent_change(old, new):
    """Return the percentage change from `old` to `new`.

    BUG: no guard for old == 0, so a baseline of zero raises
    ZeroDivisionError instead of being handled.
    """
    return (new - old) / old * 100