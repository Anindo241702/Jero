from __future__ import annotations

import asyncio

from jero.audio.console import ConsoleTTS
from jero.audio.stt import STTEngine
from jero.audio.tts import TTSEngine
from jero.core.config import STTConfig, TTSConfig


def test_stt_unload_clears_model() -> None:
    engine = STTEngine(STTConfig(), executor=None)
    assert not engine.is_loaded
    engine._model = object()  # simulate a loaded model
    assert engine.is_loaded
    engine.unload()
    assert not engine.is_loaded


def test_tts_unload_clears_voice() -> None:
    engine = TTSEngine(TTSConfig(), executor=None)
    assert not engine.is_loaded
    engine._voice = object()  # simulate a loaded voice
    assert engine.is_loaded
    engine.unload()
    assert not engine.is_loaded


async def test_console_tts_records_and_ignores_blank() -> None:
    tts = ConsoleTTS()
    await tts.speak("  ")
    assert tts.spoken == []
    await tts.speak("hello")
    assert tts.spoken == ["hello"]


def test_console_tts_has_no_model_lifecycle() -> None:
    # Text mode must never load a model: ConsoleTTS exposes no load/unload.
    tts = ConsoleTTS()
    assert not hasattr(tts, "load")
    assert not hasattr(tts, "unload")


async def test_pipeline_skips_load_for_none_and_console() -> None:
    from jero.audio.pipeline import AudioPipeline
    from jero.brain.brain import Brain
    from jero.core.bus import MessageBus
    from tests.test_brain import FakeProvider

    brain = Brain(FakeProvider("nim", "ok"), FakeProvider("groq", "ok"))
    pipeline = AudioPipeline(
        bus=MessageBus(), stt=None, tts=ConsoleTTS(), brain=brain
    )
    # Should not raise despite stt=None and a console TTS without load/unload.
    await pipeline.start()
    await asyncio.sleep(0)
    await pipeline.stop()
