"""NVIDIA NIM provider: heavy reasoning and code generation."""

from __future__ import annotations

from jero.brain.base import OpenAICompatProvider
from jero.core.config import ProviderConfig


class NvidiaNimProvider(OpenAICompatProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="nvidia-nim", config=config)
