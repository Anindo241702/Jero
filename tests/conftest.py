"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jero.core.config import Config, load_config


@pytest.fixture()
def example_config(tmp_path: Path) -> Config:
    """Build a Config from config.example.json with dummy secrets."""
    repo_root = Path(__file__).resolve().parents[1]
    example = json.loads((repo_root / "config.example.json").read_text(encoding="utf-8"))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(example), encoding="utf-8")
    import os

    os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")
    os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
    return load_config(cfg_path, load_env=False, require_secrets=True)
