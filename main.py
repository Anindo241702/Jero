"""Jero entrypoint: boots the async orchestrator.

Usage:
    python main.py

Requires a populated ``.env`` (NVIDIA_API_KEY, GROQ_API_KEY) and a
``config.json`` (copy from ``config.example.json``).
"""

from __future__ import annotations

import asyncio
import logging

from jero.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


async def _main() -> None:
    orchestrator = Orchestrator.from_default_config()
    await orchestrator.run()


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user; exiting.")


if __name__ == "__main__":
    main()
