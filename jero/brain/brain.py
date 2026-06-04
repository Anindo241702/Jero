"""The Brain facade.

Routes work to the right provider:
- ``reason`` / ``generate_code`` -> NVIDIA NIM (heavy reasoning, coding)
- ``research`` -> Groq (fast ideation / feature discovery)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from jero.brain.base import LLMProvider, Message
from jero.brain.groq_client import GroqProvider
from jero.brain.nvidia_nim import NvidiaNimProvider
from jero.core.config import Config

logger = logging.getLogger(__name__)

REASONING_SYSTEM_PROMPT = (
    "You are Jero, an autonomous, voice-operated OS-level assistant. "
    "Answer concisely and conversationally. The user may speak Bangla or English; "
    "reply in the same language they used."
)

RESEARCH_SYSTEM_PROMPT = (
    "You are Jero's research module. Given a topic, brainstorm concrete, "
    "implementable feature ideas and summarize relevant approaches succinctly."
)

CODEGEN_SYSTEM_PROMPT = (
    "You are Jero's code-generation module. Produce correct, minimal, "
    "well-structured Python that fits an asyncio-based codebase. "
    "Return only code unless asked otherwise."
)


class Brain:
    def __init__(self, nim: LLMProvider, groq: LLMProvider) -> None:
        self._nim = nim
        self._groq = groq

    @classmethod
    def from_config(cls, config: Config) -> Brain:
        return cls(
            nim=NvidiaNimProvider(config.nvidia),
            groq=GroqProvider(config.groq),
        )

    async def reason(
        self,
        prompt: str,
        context: Sequence[Message] | None = None,
    ) -> str:
        """Conversational reasoning via NVIDIA NIM."""
        messages: list[Message] = [{"role": "system", "content": REASONING_SYSTEM_PROMPT}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        logger.debug("Brain.reason -> NVIDIA NIM")
        return await self._nim.chat(messages)

    async def research(self, topic: str) -> str:
        """Fast feature research / ideation via Groq."""
        messages: list[Message] = [
            {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": topic},
        ]
        logger.debug("Brain.research -> Groq")
        return await self._groq.chat(messages)

    async def generate_code(self, spec: str) -> str:
        """Code generation via NVIDIA NIM."""
        messages: list[Message] = [
            {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
            {"role": "user", "content": spec},
        ]
        logger.debug("Brain.generate_code -> NVIDIA NIM")
        return await self._nim.chat(messages)

    async def aclose(self) -> None:
        await self._nim.aclose()
        await self._groq.aclose()
