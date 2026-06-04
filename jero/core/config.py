"""Configuration loading and validation.

Secrets come from ``.env`` (git-ignored); non-secret settings come from
``config.json`` (git-ignored, with ``config.example.json`` as the committed
template). Secrets are never written to logs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.json"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "config.example.json"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.2


@dataclass(frozen=True)
class STTConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "bn"
    beam_size: int = 1
    models_dir: str = "models/whisper"


@dataclass(frozen=True)
class TTSConfig:
    voice: str = "bn_BD-custom-medium"
    models_dir: str = "models/piper"
    voice_url: str = ""
    config_url: str = ""


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


@dataclass(frozen=True)
class AutonomyConfig:
    enabled: bool = False
    idle_seconds_before_research: int = 120
    poll_interval_seconds: float = 5.0
    cooldown_seconds: float = 60.0
    backlog_path: str = "autonomy_backlog.md"
    proposals_dir: str = "proposals"


@dataclass(frozen=True)
class Config:
    nvidia: ProviderConfig
    groq: ProviderConfig
    audio: AudioConfig
    autonomy: AutonomyConfig
    max_workers: int
    log_level: str
    repo_root: Path = REPO_ROOT


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required secret '{name}'. Copy .env.example to .env and set it."
        )
    return value


def _load_raw(config_path: Path) -> dict:
    if not config_path.exists():
        raise ConfigError(
            f"Config file not found at {config_path}. "
            f"Copy {EXAMPLE_CONFIG_PATH.name} to {config_path.name} and edit it."
        )
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - trivial
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc


def load_config(
    config_path: Path | str | None = None,
    *,
    load_env: bool = True,
    require_secrets: bool = True,
) -> Config:
    """Load and validate Jero's configuration.

    Args:
        config_path: Path to ``config.json``. Defaults to repo root.
        load_env: Whether to load ``.env`` into the process environment.
        require_secrets: If False, missing API keys become empty strings
            (useful for tests that mock the providers).
    """
    if load_env:
        load_dotenv(REPO_ROOT / ".env")

    raw = _load_raw(Path(config_path) if config_path else DEFAULT_CONFIG_PATH)

    def _secret(name: str) -> str:
        if require_secrets:
            return _require_env(name)
        return os.environ.get(name, "").strip()

    brain = raw.get("brain", {})
    nvidia_raw = brain.get("nvidia", {})
    groq_raw = brain.get("groq", {})

    nvidia = ProviderConfig(
        base_url=nvidia_raw.get("base_url", "https://integrate.api.nvidia.com/v1"),
        model=nvidia_raw.get("model", "meta/llama-3.1-70b-instruct"),
        api_key=_secret("NVIDIA_API_KEY"),
        timeout_seconds=float(nvidia_raw.get("timeout_seconds", 60)),
        max_tokens=int(nvidia_raw.get("max_tokens", 1024)),
        temperature=float(nvidia_raw.get("temperature", 0.2)),
    )
    groq = ProviderConfig(
        base_url=groq_raw.get("base_url", "https://api.groq.com/openai/v1"),
        model=groq_raw.get("model", "llama-3.1-8b-instant"),
        api_key=_secret("GROQ_API_KEY"),
        timeout_seconds=float(groq_raw.get("timeout_seconds", 30)),
        max_tokens=int(groq_raw.get("max_tokens", 1024)),
        temperature=float(groq_raw.get("temperature", 0.4)),
    )

    audio_raw = raw.get("audio", {})
    stt_raw = audio_raw.get("stt", {})
    tts_raw = audio_raw.get("tts", {})
    audio = AudioConfig(
        sample_rate=int(audio_raw.get("sample_rate", 16000)),
        stt=STTConfig(
            model_size=stt_raw.get("model_size", "base"),
            device=stt_raw.get("device", "cpu"),
            compute_type=stt_raw.get("compute_type", "int8"),
            language=stt_raw.get("language", "bn"),
            beam_size=int(stt_raw.get("beam_size", 1)),
            models_dir=stt_raw.get("models_dir", "models/whisper"),
        ),
        tts=TTSConfig(
            voice=tts_raw.get("voice", "bn_BD-custom-medium"),
            models_dir=tts_raw.get("models_dir", "models/piper"),
            voice_url=tts_raw.get("voice_url", ""),
            config_url=tts_raw.get("config_url", ""),
        ),
    )

    autonomy_raw = raw.get("autonomy", {})
    autonomy = AutonomyConfig(
        enabled=bool(autonomy_raw.get("enabled", False)),
        idle_seconds_before_research=int(
            autonomy_raw.get("idle_seconds_before_research", 120)
        ),
        poll_interval_seconds=float(autonomy_raw.get("poll_interval_seconds", 5)),
        cooldown_seconds=float(autonomy_raw.get("cooldown_seconds", 60)),
        backlog_path=autonomy_raw.get("backlog_path", "autonomy_backlog.md"),
        proposals_dir=autonomy_raw.get("proposals_dir", "proposals"),
    )

    executor_raw = raw.get("executor", {})
    max_workers = int(executor_raw.get("max_workers", os.cpu_count() or 4))

    log_level = str(raw.get("logging", {}).get("level", "INFO")).upper()

    return Config(
        nvidia=nvidia,
        groq=groq,
        audio=audio,
        autonomy=autonomy,
        max_workers=max_workers,
        log_level=log_level,
    )
