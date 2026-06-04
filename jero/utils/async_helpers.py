"""Helpers for bridging blocking work into the asyncio event loop.

Jero offloads every blocking call (model inference, audio I/O, OS calls) to a
shared bounded thread pool so the main event loop never stalls.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_blocking(
    executor: Executor | None,
    func: Callable[..., T],
    /,
    *args: object,
    **kwargs: object,
) -> T:
    """Run a blocking ``func`` in ``executor`` without blocking the event loop.

    ``functools.partial`` is used so keyword arguments are supported (the stdlib
    ``loop.run_in_executor`` only accepts positional args).
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, call)


async def retry_async(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``coro_factory`` with exponential backoff.

    ``coro_factory`` must return a fresh awaitable each call, since an awaitable
    can only be awaited once.
    """
    delay = base_delay
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except exceptions as exc:  # noqa: BLE001 - re-raised below after retries
            last_exc = exc
            if attempt == attempts:
                break
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs",
                attempt,
                attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
    assert last_exc is not None
    raise last_exc
