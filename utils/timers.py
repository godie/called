"""Timing and time-related utilities."""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def Timer(name: str = "operation") -> Generator[dict, None, None]:
    """Context manager to measure elapsed time of a block.

    Usage:
        with Timer("my-op") as t:
            do_work()
        print(f"Took {t['elapsed']:.3f}s")

    Args:
        name: Label for the timing measurement.

    Yields:
        A mutable dict with 'name' and 'elapsed' keys. 'elapsed' is
        populated after the block executes.
    """
    result: dict = {"name": name, "elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def get_timestamp() -> str:
    """Get current timestamp in ISO 8601 format.

    Returns:
        ISO 8601 formatted timestamp string.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to HH:MM:SS.mmm format.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
