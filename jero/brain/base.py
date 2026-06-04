"""Abstract LLM provider interface and a shared OpenAI-compatible client.

Both NVIDIA NIM and Groq expose OpenAI-compatible ``/chat/completions``
endpoints, so a single async httpx-based client serves both with different
base URLs, models, and keys.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx

from jero.core.config import ProviderConfig
from jero.utils.async_helpers import retry_async

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when an LLM request fails."""


class LLMProvider(ABC):
    """Async chat-completion provider."""

    name: str
    model: str = "unknown"

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the assistant's reply text for ``messages``."""

    async def aclose(self) -> None:  # noqa: B027 - optional override, intentional no-op
        """Release any held resources (override if needed)."""


class OpenAICompatProvider(LLMProvider):
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.model = config.model
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self._config.model,
            "messages": list(messages),
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": (
                temperature if temperature is not None else self._config.temperature
            ),
        }

        async def _do_request() -> str:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise LLMError(
                    f"{self.name}: unexpected response shape: {data}"
                ) from exc

        try:
            return await retry_async(
                _do_request,
                attempts=3,
                exceptions=(httpx.TransportError, httpx.HTTPStatusError),
            )
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"{self.name}: HTTP {exc.response.status_code} from {exc.request.url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: request failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
