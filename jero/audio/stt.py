"""Speech-to-text via faster-whisper.

The model is downloaded automatically by faster-whisper into the configured
``models_dir`` on first load. All blocking work (model load + transcription)
is offloaded to the shared executor so the event loop stays responsive.
"""

from __future__ import annotations

import logging
from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jero.core.config import STTConfig
from jero.utils.async_helpers import run_blocking

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)


class STTEngine:
    """Async wrapper around a faster-whisper ``WhisperModel``."""

    def __init__(
        self,
        config: STTConfig,
        executor: Executor | None,
        sample_rate: int = 16000,
    ) -> None:
        self._config = config
        self._executor = executor
        self._sample_rate = sample_rate
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def load(self) -> None:
        """Load (and on first run, download) the whisper model off-thread."""
        if self._model is not None:
            return
        models_dir = Path(self._config.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Loading faster-whisper model '%s' (%s/%s) into %s",
            self._config.model_size,
            self._config.device,
            self._config.compute_type,
            models_dir,
        )
        self._model = await run_blocking(self._executor, self._load_model, models_dir)
        logger.info("STT model ready")

    def unload(self) -> None:
        """Drop the model reference so it can be garbage-collected (frees RAM)."""
        if self._model is not None:
            logger.info("Unloading STT model")
            self._model = None

    def _load_model(self, models_dir: Path) -> Any:
        # Imported lazily so the package imports without the heavy dependency.
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._config.model_size,
            device=self._config.device,
            compute_type=self._config.compute_type,
            download_root=str(models_dir),
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 numpy array sampled at ``sample_rate``."""
        if self._model is None:
            await self.load()
        return await run_blocking(self._executor, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        assert self._model is not None
        segments, _info = self._model.transcribe(
            audio,
            language=self._config.language,
            beam_size=self._config.beam_size,
        )
        return "".join(segment.text for segment in segments).strip()
