"""A single shared, bounded thread pool for all blocking work.

Centralizing the pool keeps memory predictable on an 8GB / 4-core box: every
module offloads blocking calls here instead of spawning its own threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


class SharedExecutor:
    """Owns one ``ThreadPoolExecutor`` and exposes lifecycle helpers."""

    def __init__(self, max_workers: int) -> None:
        self._max_workers = max(1, max_workers)
        self._executor: ThreadPoolExecutor | None = None

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="jero-worker",
            )
        return self._executor

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def shutdown(self, *, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
