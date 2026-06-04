"""Wires the modules together and owns the application lifecycle.

For the MVP this boots the Brain + Audio pipeline. Autonomy and OS-interaction
modules plug in here in later iterations.
"""

from __future__ import annotations

import asyncio
import logging

from jero.audio.pipeline import AudioPipeline
from jero.audio.stt import STTEngine
from jero.audio.tts import TTSEngine
from jero.brain.brain import Brain
from jero.core.bus import MessageBus
from jero.core.config import Config, load_config
from jero.core.executor import SharedExecutor
from jero.core.logging import setup_logging

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._executor = SharedExecutor(config.max_workers)
        self._bus = MessageBus()
        self._brain = Brain.from_config(config)
        self._stt = STTEngine(
            config.audio.stt,
            self._executor.executor,
            sample_rate=config.audio.sample_rate,
        )
        self._tts = TTSEngine(config.audio.tts, self._executor.executor)
        self._pipeline = AudioPipeline(
            bus=self._bus,
            stt=self._stt,
            tts=self._tts,
            brain=self._brain,
            sample_rate=config.audio.sample_rate,
        )

    @classmethod
    def from_default_config(cls) -> Orchestrator:
        config = load_config()
        setup_logging(config.log_level)
        return cls(config)

    async def startup(self) -> None:
        """Verify/preload models so failures surface before we accept input."""
        logger.info("Jero starting up (max_workers=%d)", self._config.max_workers)
        # Ensure TTS voice files are present (downloads if URLs configured).
        await self._tts.ensure_models()
        # Preload models concurrently to cut first-utterance latency.
        await asyncio.gather(self._stt.load(), self._tts.load())
        await self._pipeline.start()
        logger.info("Jero is ready.")

    async def shutdown(self) -> None:
        logger.info("Jero shutting down...")
        await self._pipeline.stop()
        await self._brain.aclose()
        self._executor.shutdown(wait=True)
        logger.info("Shutdown complete.")

    async def run(self) -> None:
        """Run until cancelled (e.g. Ctrl+C)."""
        await self.startup()
        try:
            await asyncio.Event().wait()  # run forever
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()
