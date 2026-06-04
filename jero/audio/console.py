"""Lightweight stand-ins for hardware audio, used by ``--text`` mode.

These let you exercise the full bus + Brain + pipeline wiring without a
microphone, speakers, or any local model files.
"""

from __future__ import annotations

import logging
from concurrent.futures import Executor

from jero.utils.async_helpers import run_blocking

logger = logging.getLogger(__name__)


class ConsoleTTS:
    """A TTS sink that prints instead of synthesizing audio.

    Implements the ``speak`` method the pipeline relies on. Deliberately has no
    ``load``/``unload`` so no model is ever loaded in text mode.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.spoken.append(text)
        print(f"\nJero: {text}\n", flush=True)


async def read_console_line(
    executor: Executor | None, prompt: str = "You: "
) -> str:
    """Read one line from stdin without blocking the event loop."""

    def _read() -> str:
        try:
            return input(prompt)
        except EOFError:
            return "/exit"

    return await run_blocking(executor, _read)
