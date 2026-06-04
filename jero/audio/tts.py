"""Text-to-speech via Piper (local, Bangla-capable).

Piper voice files (``<voice>.onnx`` and ``<voice>.onnx.json``) are downloaded
into the configured ``models_dir`` on first run when download URLs are provided.
A presence check runs before the pipeline starts so failures surface early.
"""

from __future__ import annotations

import logging
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

import httpx

from jero.core.config import TTSConfig
from jero.utils.async_helpers import run_blocking

logger = logging.getLogger(__name__)


class TTSModelMissingError(RuntimeError):
    """Raised when Piper voice files are absent and cannot be downloaded."""


class TTSEngine:
    """Async wrapper around a Piper ``PiperVoice``."""

    def __init__(self, config: TTSConfig, executor: Executor | None) -> None:
        self._config = config
        self._executor = executor
        self._voice: Any | None = None

    @property
    def models_dir(self) -> Path:
        return Path(self._config.models_dir)

    @property
    def voice_path(self) -> Path:
        return self.models_dir / f"{self._config.voice}.onnx"

    @property
    def config_path(self) -> Path:
        return self.models_dir / f"{self._config.voice}.onnx.json"

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None

    def has_model_files(self) -> bool:
        return self.voice_path.exists() and self.config_path.exists()

    async def ensure_models(self) -> None:
        """Download voice files if missing; raise if they can't be obtained."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self.has_model_files():
            return
        if not (self._config.voice_url and self._config.config_url):
            raise TTSModelMissingError(
                f"Piper voice '{self._config.voice}' not found in {self.models_dir} "
                "and no voice_url/config_url configured for auto-download. "
                "Set audio.tts.voice_url and audio.tts.config_url in config.json."
            )
        logger.info("Downloading Piper voice '%s' into %s", self._config.voice, self.models_dir)
        await self._download(self._config.voice_url, self.voice_path)
        await self._download(self._config.config_url, self.config_path)
        if not self.has_model_files():
            raise TTSModelMissingError("Piper voice download did not produce expected files")
        logger.info("Piper voice ready")

    async def _download(self, url: str, dest: Path) -> None:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    async for chunk in response.aiter_bytes():
                        fh.write(chunk)
                tmp.replace(dest)

    async def load(self) -> None:
        if self._voice is not None:
            return
        await self.ensure_models()
        logger.info("Loading Piper voice from %s", self.voice_path)
        self._voice = await run_blocking(self._executor, self._load_voice)
        logger.info("TTS voice ready")

    def unload(self) -> None:
        """Drop the voice reference so it can be garbage-collected (frees RAM)."""
        if self._voice is not None:
            logger.info("Unloading TTS voice")
            self._voice = None

    def _load_voice(self) -> Any:
        from piper.voice import PiperVoice

        return PiperVoice.load(str(self.voice_path), config_path=str(self.config_path))

    async def synthesize(self, text: str) -> bytes:
        """Return 16-bit PCM audio bytes for ``text`` (no playback)."""
        if self._voice is None:
            await self.load()
        return await run_blocking(self._executor, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        assert self._voice is not None
        chunks = bytearray()
        for audio_bytes in self._voice.synthesize_stream_raw(text):
            chunks.extend(audio_bytes)
        return bytes(chunks)

    async def speak(self, text: str) -> None:
        """Synthesize and play ``text`` through the default output device."""
        if not text.strip():
            return
        pcm = await self.synthesize(text)
        await run_blocking(self._executor, self._play_sync, pcm)

    def _play_sync(self, pcm: bytes) -> None:
        import numpy as np
        import sounddevice as sd

        sample_rate = getattr(getattr(self._voice, "config", None), "sample_rate", 22050)
        samples = np.frombuffer(pcm, dtype=np.int16)
        sd.play(samples, samplerate=sample_rate)
        sd.wait()
