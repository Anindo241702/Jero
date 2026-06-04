"""Wires the modules together and owns the application lifecycle.

For the MVP this boots the Brain + Audio pipeline. Autonomy and OS-interaction
modules plug in here in later iterations.
"""

from __future__ import annotations

import asyncio
import logging

from jero.audio.console import ConsoleTTS, read_console_line
from jero.audio.pipeline import AudioPipeline
from jero.audio.stt import STTEngine
from jero.audio.tts import TTSEngine
from jero.autonomy.engine import AutonomyEngine
from jero.brain.brain import Brain
from jero.core.bus import TOPIC_TRANSCRIPT, MessageBus
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
        self._autonomy = AutonomyEngine(
            brain=self._brain,
            bus=self._bus,
            config=config.autonomy,
            repo_root=config.repo_root,
            executor=self._executor.executor,
        )

    @classmethod
    def from_default_config(cls) -> Orchestrator:
        config = load_config()
        setup_logging(config.log_level)
        return cls(config)

    async def startup(self) -> None:
        """Start the pipeline. Models are loaded by ``pipeline.start`` so they
        reside in memory only while the pipeline is running, and are freed on
        ``pipeline.stop``."""
        logger.info("Jero starting up (max_workers=%d)", self._config.max_workers)
        await self._pipeline.start()
        if self._config.autonomy.enabled:
            self._autonomy.start()
            logger.info("Autonomy engine enabled.")
        logger.info("Jero is ready.")

    async def shutdown(self) -> None:
        logger.info("Jero shutting down...")
        await self._autonomy.stop()
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

    async def run_text(self) -> None:
        """Hardware-free REPL: type input -> Brain -> printed reply.

        Exercises the real bus + reasoning/speaking loops with no microphone,
        speakers, STT, or TTS model files. Only the API keys are required.
        """
        console_tts = ConsoleTTS()
        pipeline = AudioPipeline(
            bus=self._bus,
            stt=None,  # no STT: input comes from the console
            tts=console_tts,
            brain=self._brain,
            sample_rate=self._config.audio.sample_rate,
        )
        await pipeline.start()
        print("Jero text mode. Type a message, or '/exit' to quit.", flush=True)
        try:
            while True:
                line = await read_console_line(self._executor.executor)
                if line.strip().lower() in {"/exit", "/quit"}:
                    break
                if not line.strip():
                    continue
                await self._bus.publish(TOPIC_TRANSCRIPT, line)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            await pipeline.stop()
            await self._brain.aclose()
            self._executor.shutdown(wait=True)
            print("Goodbye.", flush=True)
