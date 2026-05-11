"""
Benchmark utilities
===================
Thin wrappers around time.perf_counter so timing logic stays out of
the scenario code.  perf_counter is the highest-resolution clock
available in Python and is not affected by system time adjustments.
"""

import time
from typing import Any, Callable


def time_once(fn: Callable, *args) -> tuple[float, Any]:
    """
    Run fn(*args) exactly once.

    Returns
    -------
    (elapsed_seconds, return_value)
    """
    start = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - start, result


def time_average(fn: Callable, *args, runs: int = 7) -> tuple[float, Any]:
    """
    Run fn(*args) `runs` times and return the mean elapsed time.

    The last return value is kept (all runs should produce the same result).
    Averaging reduces noise from OS scheduling jitter, which matters a lot
    for very fast functions (< 1 ms).

    Returns
    -------
    (mean_elapsed_seconds, last_return_value)
    """
    total = 0.0
    result = None
    for _ in range(runs):
        t, result = time_once(fn, *args)
        total += t
    return total / runs, result