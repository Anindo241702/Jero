"""Jero entrypoint: boots the async orchestrator.

Usage:
    python main.py            # full voice mode (mic + speakers + STT/TTS)
    python main.py --text     # hardware-free text REPL (type -> Brain -> print)

Requires a populated ``.env`` (NVIDIA_API_KEY, GROQ_API_KEY) and a
``config.json`` (copy from ``config.example.json``). Text mode needs only the
API keys -- no microphone, speakers, or model files.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Ensure the repo root is importable when run as a plain script, regardless of
# how the interpreter is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jero.core.orchestrator import Orchestrator  # noqa: E402  (after sys.path setup)

logger = logging.getLogger(__name__)


async def _main(text_mode: bool) -> None:
    orchestrator = Orchestrator.from_default_config()
    if text_mode:
        await orchestrator.run_text()
    else:
        await orchestrator.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jero autonomous assistant")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run a hardware-free text REPL (no mic/speakers/STT/TTS).",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_main(text_mode=args.text))
    except KeyboardInterrupt:
        logger.info("Interrupted by user; exiting.")


if __name__ == "__main__":
    main()
