"""A minimal async message bus implementing the producer-consumer pattern.

Modules communicate by publishing to named topics; subscribers each get their
own bounded ``asyncio.Queue`` for backpressure. Bounded queues keep memory flat
under load rather than growing without limit.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    topic: str
    payload: Any


# Well-known topics used across the pipeline.
TOPIC_TRANSCRIPT = "audio.transcript"   # STT output -> Brain
TOPIC_RESPONSE = "brain.response"       # Brain output -> TTS
TOPIC_SPEAK = "audio.speak"             # arbitrary text -> TTS


class MessageBus:
    def __init__(self, maxsize: int = 100) -> None:
        self._maxsize = maxsize
        self._subscribers: dict[str, list[asyncio.Queue[Message]]] = defaultdict(list)

    def subscribe(self, topic: str) -> asyncio.Queue[Message]:
        """Register a subscriber and return its dedicated queue."""
        queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers[topic].append(queue)
        return queue

    async def publish(self, topic: str, payload: Any) -> None:
        """Publish a payload to all subscribers of ``topic``.

        Applies backpressure: if a subscriber's queue is full, this awaits.
        """
        message = Message(topic=topic, payload=payload)
        subscribers = self._subscribers.get(topic, [])
        if not subscribers:
            logger.debug("No subscribers for topic '%s'; dropping message", topic)
            return
        await asyncio.gather(*(q.put(message) for q in subscribers))
