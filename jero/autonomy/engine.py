"""Idle-time self-improvement engine (scaffold).

Full implementation lands in a follow-up PR. The interface below documents the
intended contract: wait for the system to be idle, research a feature via Groq,
then generate an implementation via NVIDIA NIM.
"""

from __future__ import annotations

import logging

from jero.brain.brain import Brain
from jero.core.config import AutonomyConfig

logger = logging.getLogger(__name__)


class AutonomyEngine:
    def __init__(self, brain: Brain, config: AutonomyConfig) -> None:
        self._brain = brain
        self._config = config

    async def run(self) -> None:  # pragma: no cover - scaffold
        """Run the idle-time research/implement loop.

        Not yet implemented; see the follow-up PR tracked in the project plan.
        """
        raise NotImplementedError(
            "AutonomyEngine.run is implemented in a later iteration."
        )
