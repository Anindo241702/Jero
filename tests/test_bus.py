from __future__ import annotations

import asyncio

from jero.core.bus import MessageBus


async def test_publish_delivers_to_subscriber() -> None:
    bus = MessageBus()
    queue = bus.subscribe("topic.a")
    await bus.publish("topic.a", "payload")
    msg = await asyncio.wait_for(queue.get(), timeout=1)
    assert msg.topic == "topic.a"
    assert msg.payload == "payload"


async def test_multiple_subscribers_each_get_message() -> None:
    bus = MessageBus()
    q1 = bus.subscribe("t")
    q2 = bus.subscribe("t")
    await bus.publish("t", 42)
    assert (await q1.get()).payload == 42
    assert (await q2.get()).payload == 42


async def test_publish_without_subscribers_is_noop() -> None:
    bus = MessageBus()
    await bus.publish("nobody", "x")  # should not raise
