"""The audio producer-consumer pipeline: mic -> STT -> Brain -> TTS.

Each stage is an independent asyncio task communicating over the message bus,
so no stage blocks another. The microphone producer is optional (it requires a
working input device); the consumers can also be driven programmatically by
publishing to the bus.
"""

from __future__ import annotations

import asyncio
import logging

from jero.audio.stt import STTEngine
from jero.audio.tts import TTSEngine
from jero.brain.brain import Brain
from jero.core.bus import (
    TOPIC_RESPONSE,
    TOPIC_SPEAK,
    TOPIC_TRANSCRIPT,
    MessageBus,
)

logger = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(
        self,
        bus: MessageBus,
        stt: STTEngine,
        tts: TTSEngine,
        brain: Brain,
        sample_rate: int = 16000,
    ) -> None:
        self._bus = bus
        self._stt = stt
        self._tts = tts
        self._brain = brain
        self._sample_rate = sample_rate
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Spawn consumer tasks. Returns immediately; use ``stop`` to tear down.

        Subscriptions are registered synchronously here (before any await) so
        messages published right after ``start`` are never dropped due to a
        consumer task not having run yet.
        """
        self._stop.clear()
        # Models reside in memory only while the pipeline is running: load on
        # start, free on stop. Engines may be absent (e.g. text-only mode) or
        # lightweight stand-ins (console) without a load method.
        await self._load_engines()
        transcript_q = self._bus.subscribe(TOPIC_TRANSCRIPT)
        response_q = self._bus.subscribe(TOPIC_RESPONSE)
        speak_q = self._bus.subscribe(TOPIC_SPEAK)
        self._tasks = [
            asyncio.create_task(
                self._reasoning_loop(transcript_q), name="jero-reasoning"
            ),
            asyncio.create_task(
                self._speaking_loop(response_q, speak_q), name="jero-speaking"
            ),
        ]
        logger.info("Audio pipeline started (%d consumer tasks)", len(self._tasks))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._unload_engines()
        logger.info("Audio pipeline stopped")

    async def _load_engines(self) -> None:
        for engine in (self._stt, self._tts):
            load = getattr(engine, "load", None)
            if callable(load):
                await load()

    def _unload_engines(self) -> None:
        for engine in (self._stt, self._tts):
            unload = getattr(engine, "unload", None)
            if callable(unload):
                unload()

    async def _reasoning_loop(self, queue: asyncio.Queue) -> None:
        """Consume transcripts, ask the Brain, publish responses."""
        while not self._stop.is_set():
            message = await queue.get()
            text = str(message.payload).strip()
            if not text:
                continue
            logger.info("User said: %s", text)
            try:
                reply = await self._brain.reason(text)
            except Exception:  # noqa: BLE001 - keep the loop alive
                logger.exception("Brain failed to produce a reply")
                continue
            await self._bus.publish(TOPIC_RESPONSE, reply)

    async def _speaking_loop(
        self, response_q: asyncio.Queue, speak_q: asyncio.Queue
    ) -> None:
        """Consume responses (and ad-hoc speak requests) and play them via TTS."""
        while not self._stop.is_set():
            get_response = asyncio.create_task(response_q.get())
            get_speak = asyncio.create_task(speak_q.get())
            try:
                done, pending = await asyncio.wait(
                    {get_response, get_speak}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                get_response.cancel()
                get_speak.cancel()
                raise
            for task in pending:
                task.cancel()
            for task in done:
                message = task.result()
                text = str(message.payload).strip()
                logger.info("Jero says: %s", text)
                try:
                    await self._tts.speak(text)
                except Exception:  # noqa: BLE001 - keep the loop alive
                    logger.exception("TTS failed to speak")
