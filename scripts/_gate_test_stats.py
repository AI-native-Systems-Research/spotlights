"""Throwaway helper used to exercise the claude-review gate. Safe to delete.

Deliberately contains a division-by-zero bug so the review has something to find.
"""


def mean_latency_ms(samples: list[float]) -> float:
    """Return the arithmetic mean of `samples`.

    BUG: an empty list divides by zero instead of being handled.
    """
    total = 0.0
    for s in samples:
        total += s
    return total / len(samples)


def percent_of_budget(latency_ms: float, budget_ms: float) -> float:
    """Return `latency_ms` as a percentage of `budget_ms`.

    BUG: a zero budget divides by zero.
    """
    return (latency_ms / budget_ms) * 100.0


if __name__ == "__main__":
    print(mean_latency_ms([12.0, 15.5, 9.25]))
    print(percent_of_budget(30.0, 50.0))
    print(mean_latency_ms([]))  # ZeroDivisionError
