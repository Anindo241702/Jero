from __future__ import annotations

from collections.abc import Sequence

import pytest

from jero.brain.base import LLMProvider, Message
from jero.brain.brain import Brain


class FakeProvider(LLMProvider):
    def __init__(self, name: str, reply: str) -> None:
        self.name = name
        self._reply = reply
        self.calls: list[list[Message]] = []
        self.closed = False

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(list(messages))
        return self._reply

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture()
def brain_with_fakes() -> tuple[Brain, FakeProvider, FakeProvider]:
    nim = FakeProvider("nvidia-nim", "reasoned answer")
    groq = FakeProvider("groq", "research result")
    return Brain(nim=nim, groq=groq), nim, groq


async def test_reason_routes_to_nim(brain_with_fakes) -> None:
    brain, nim, groq = brain_with_fakes
    out = await brain.reason("hello")
    assert out == "reasoned answer"
    assert len(nim.calls) == 1
    assert not groq.calls
    assert nim.calls[0][-1] == {"role": "user", "content": "hello"}


async def test_research_routes_to_groq(brain_with_fakes) -> None:
    brain, nim, groq = brain_with_fakes
    out = await brain.research("voice features")
    assert out == "research result"
    assert len(groq.calls) == 1
    assert not nim.calls


async def test_generate_code_routes_to_nim(brain_with_fakes) -> None:
    brain, nim, groq = brain_with_fakes
    out = await brain.generate_code("write a function")
    assert out == "reasoned answer"
    assert len(nim.calls) == 1
    assert not groq.calls


async def test_reason_includes_context(brain_with_fakes) -> None:
    brain, nim, _groq = brain_with_fakes
    context: list[Message] = [{"role": "assistant", "content": "prior turn"}]
    await brain.reason("follow up", context=context)
    sent = nim.calls[0]
    assert sent[0]["role"] == "system"
    assert {"role": "assistant", "content": "prior turn"} in sent


async def test_aclose_closes_both(brain_with_fakes) -> None:
    brain, nim, groq = brain_with_fakes
    await brain.aclose()
    assert nim.closed and groq.closed
