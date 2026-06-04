"""Groq provider: fast research and feature ideation."""

from __future__ import annotations

from jero.brain.base import OpenAICompatProvider
from jero.core.config import ProviderConfig


class GroqProvider(OpenAICompatProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="groq", config=config)
