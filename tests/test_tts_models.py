from __future__ import annotations

from pathlib import Path

import pytest

from jero.audio.tts import TTSEngine, TTSModelMissingError
from jero.core.config import TTSConfig


async def test_ensure_models_raises_when_missing_and_no_urls(tmp_path: Path) -> None:
    cfg = TTSConfig(voice="bn_BD-test", models_dir=str(tmp_path / "piper"))
    engine = TTSEngine(cfg, executor=None)
    assert not engine.has_model_files()
    with pytest.raises(TTSModelMissingError):
        await engine.ensure_models()


async def test_ensure_models_passes_when_files_present(tmp_path: Path) -> None:
    models_dir = tmp_path / "piper"
    models_dir.mkdir(parents=True)
    voice = "bn_BD-test"
    (models_dir / f"{voice}.onnx").write_bytes(b"fake-onnx")
    (models_dir / f"{voice}.onnx.json").write_text("{}", encoding="utf-8")
    cfg = TTSConfig(voice=voice, models_dir=str(models_dir))
    engine = TTSEngine(cfg, executor=None)
    assert engine.has_model_files()
    await engine.ensure_models()  # should not raise
