from __future__ import annotations

import json
from pathlib import Path

import pytest

from jero.core.config import Config, ConfigError, load_config


def test_load_config_from_example(example_config: Config) -> None:
    cfg = example_config
    assert cfg.nvidia.api_key == "test-nvidia-key"
    assert cfg.groq.api_key == "test-groq-key"
    assert cfg.nvidia.base_url.startswith("https://")
    assert cfg.groq.base_url.startswith("https://")
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.stt.language == "bn"
    assert cfg.max_workers >= 1


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.json", load_env=False)


def test_missing_secret_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example = json.loads((repo_root / "config.example.json").read_text(encoding="utf-8"))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(example), encoding="utf-8")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        load_config(cfg_path, load_env=False, require_secrets=True)


def test_secrets_optional_when_not_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example = json.loads((repo_root / "config.example.json").read_text(encoding="utf-8"))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(example), encoding="utf-8")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = load_config(cfg_path, load_env=False, require_secrets=False)
    assert cfg.nvidia.api_key == ""
