from __future__ import annotations

import asyncio

import pytest

from jero.core.bus import MessageBus


async def test_bounded_queue_applies_backpressure() -> None:
    """When a subscriber is slow, publish blocks instead of overflowing."""
    bus = MessageBus(maxsize=2)
    _q = bus.subscribe("t")  # never drained -> simulates a slow consumer

    await bus.publish("t", 1)
    await bus.publish("t", 2)  # queue now full (maxsize=2)

    # A third publish must block (backpressure), not raise or grow unbounded.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(bus.publish("t", 3), timeout=0.2)


async def test_publish_unblocks_when_consumer_drains() -> None:
    bus = MessageBus(maxsize=1)
    q = bus.subscribe("t")
    await bus.publish("t", "a")  # full

    async def drain_later() -> None:
        await asyncio.sleep(0.05)
        await q.get()  # frees a slot

    asyncio.create_task(drain_later())
    # Initially blocked, then unblocks once the consumer drains.
    await asyncio.wait_for(bus.publish("t", "b"), timeout=1.0)
