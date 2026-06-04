from __future__ import annotations

import asyncio

from jero.audio.pipeline import AudioPipeline
from jero.brain.brain import Brain
from jero.core.bus import TOPIC_RESPONSE, TOPIC_TRANSCRIPT, MessageBus
from tests.test_brain import FakeProvider


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


async def test_transcript_flows_through_brain_to_response() -> None:
    bus = MessageBus()
    nim = FakeProvider("nvidia-nim", "hi there")
    groq = FakeProvider("groq", "unused")
    brain = Brain(nim=nim, groq=groq)
    tts = FakeTTS()
    pipeline = AudioPipeline(bus=bus, stt=None, tts=tts, brain=brain)

    response_q = bus.subscribe(TOPIC_RESPONSE)
    await pipeline.start()
    try:
        await bus.publish(TOPIC_TRANSCRIPT, "hello jero")
        msg = await asyncio.wait_for(response_q.get(), timeout=2)
        assert msg.payload == "hi there"
        # Speaking loop should also have spoken the response.
        await asyncio.wait_for(_wait_for(lambda: tts.spoken), timeout=2)
        assert "hi there" in tts.spoken
    finally:
        await pipeline.stop()


async def _wait_for(predicate, interval: float = 0.02) -> None:
    while not predicate():
        await asyncio.sleep(interval)


async def test_empty_transcript_ignored() -> None:
    bus = MessageBus()
    nim = FakeProvider("nvidia-nim", "should not be called")
    groq = FakeProvider("groq", "x")
    brain = Brain(nim=nim, groq=groq)
    pipeline = AudioPipeline(bus=bus, stt=None, tts=FakeTTS(), brain=brain)
    await pipeline.start()
    try:
        await bus.publish(TOPIC_TRANSCRIPT, "   ")
        await asyncio.sleep(0.1)
        assert not nim.calls
    finally:
        await pipeline.stop()
