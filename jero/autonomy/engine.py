"""Idle-time self-improvement engine.

During idle time (no bus activity for a configurable threshold), Jero:
  1. picks the next unprocessed item from a local backlog file,
  2. researches it via Groq (``Brain.research``),
  3. generates an implementation via NVIDIA NIM (``Brain.generate_code``),
  4. writes everything to a timestamped ``proposals/<ts>-<slug>/`` directory and
     logs a notification.

Safety: the engine NEVER executes generated code and NEVER overwrites existing
files. All output is confined to the proposals directory; each proposal lives in
its own unique, timestamped subdirectory. A human reviews proposals before any
code is merged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from concurrent.futures import Executor
from datetime import datetime, timezone
from pathlib import Path

from jero.brain.brain import Brain
from jero.core.bus import MessageBus
from jero.core.config import AutonomyConfig
from jero.utils.async_helpers import run_blocking

logger = logging.getLogger(__name__)

_PROCESSED_FILE = ".processed.json"
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_TASK_LINE = re.compile(r"^\s*[-*]\s*(?:\[(?P<mark>[ xX])\]\s*)?(?P<text>.+?)\s*$")


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].strip("-")) or "feature"


def extract_python_code(text: str) -> str:
    """Return code from a fenced block if present, else the raw text."""
    matches = _CODE_FENCE.findall(text)
    if matches:
        return "\n\n".join(block.strip() for block in matches).strip() + "\n"
    return text.strip() + "\n"


def parse_backlog(content: str) -> list[str]:
    """Extract candidate feature items from backlog text.

    Recognizes markdown bullets and task list items. Checked items (``[x]``) are
    treated as already done and skipped.
    """
    items: list[str] = []
    for line in content.splitlines():
        match = _TASK_LINE.match(line)
        if not match:
            continue
        if (match.group("mark") or " ").lower() == "x":
            continue
        text = match.group("text").strip()
        if text:
            items.append(text)
    return items


class AutonomyEngine:
    def __init__(
        self,
        brain: Brain,
        bus: MessageBus,
        config: AutonomyConfig,
        *,
        repo_root: Path,
        executor: Executor | None = None,
    ) -> None:
        self._brain = brain
        self._bus = bus
        self._config = config
        self._repo_root = repo_root
        self._executor = executor
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def proposals_dir(self) -> Path:
        path = Path(self._config.proposals_dir)
        return path if path.is_absolute() else self._repo_root / path

    @property
    def backlog_path(self) -> Path:
        path = Path(self._config.backlog_path)
        return path if path.is_absolute() else self._repo_root / path

    # ----- lifecycle -------------------------------------------------------

    def start(self) -> asyncio.Task[None]:
        self._stop.clear()
        self._task = asyncio.create_task(self.run(), name="jero-autonomy")
        return self._task

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def run(self) -> None:
        """Poll for idle time and generate one proposal per cooldown window."""
        threshold = self._config.idle_seconds_before_research
        poll = max(0.1, self._config.poll_interval_seconds)
        logger.info(
            "AutonomyEngine running (idle threshold=%ss, poll=%ss)", threshold, poll
        )
        while not self._stop.is_set():
            await self._sleep(poll)
            if self._stop.is_set():
                break
            if self._bus.seconds_since_last_activity() < threshold:
                continue
            topic = await self._next_topic()
            if topic is None:
                continue  # backlog empty / all processed; wait for new work
            try:
                await self.generate_proposal(topic)
            except Exception:  # noqa: BLE001 - never let the engine die
                logger.exception("Failed to generate proposal for %r", topic)
            # Space out API usage; this loop won't run again until cooldown ends.
            await self._sleep(self._config.cooldown_seconds)

    async def _sleep(self, seconds: float) -> None:
        """Sleep that returns early if stop is requested."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ----- backlog + state -------------------------------------------------

    async def _read_text(self, path: Path) -> str:
        def _read() -> str:
            return path.read_text(encoding="utf-8") if path.exists() else ""

        return await run_blocking(self._executor, _read)

    async def _load_processed(self) -> set[str]:
        path = self.proposals_dir / _PROCESSED_FILE
        raw = await self._read_text(path)
        if not raw.strip():
            return set()
        try:
            return set(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning("Corrupt %s; ignoring", path)
            return set()

    async def _mark_processed(self, topic: str) -> None:
        processed = await self._load_processed()
        processed.add(topic)
        path = self.proposals_dir / _PROCESSED_FILE
        payload = json.dumps(sorted(processed), indent=2)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")

        await run_blocking(self._executor, _write)

    async def _next_topic(self) -> str | None:
        content = await self._read_text(self.backlog_path)
        if not content.strip():
            return None
        processed = await self._load_processed()
        for item in parse_backlog(content):
            if item not in processed:
                return item
        return None

    # ----- proposal generation --------------------------------------------

    async def generate_proposal(self, topic: str) -> Path:
        """Research + generate code for ``topic`` and write a safe proposal.

        Returns the proposal directory. Does not execute or import the code.
        """
        logger.info("Autonomy: researching feature %r via Groq", topic)
        research = await self._brain.research(topic)

        spec = (
            f"Feature to implement for the Jero assistant:\n{topic}\n\n"
            f"Research notes:\n{research}\n\n"
            "Write a self-contained, asyncio-friendly Python module that fits "
            "Jero's architecture (no blocking calls on the event loop). "
            "Return the code in a single ```python block."
        )
        logger.info("Autonomy: generating implementation via NVIDIA NIM")
        raw_code = await self._brain.generate_code(spec)
        code = extract_python_code(raw_code)

        # Timestamp for readability + a short uuid so directories are unique
        # regardless of clock resolution; the exist_ok=False guard below then
        # never has to overwrite an earlier proposal.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        proposal_dir = self.proposals_dir / f"{timestamp}-{_slugify(topic)}-{suffix}"
        metadata = {
            "topic": topic,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "nvidia_model": self._brain.nvidia_model,
            "groq_model": self._brain.groq_model,
            "status": "proposed",
            "note": "Auto-generated by Jero. Review before merging. Not executed.",
        }

        def _write_all() -> None:
            # Unique timestamped dir => never overwrites an existing proposal.
            proposal_dir.mkdir(parents=True, exist_ok=False)
            (proposal_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            (proposal_dir / "research.md").write_text(
                f"# Research: {topic}\n\n{research}\n", encoding="utf-8"
            )
            (proposal_dir / "proposal.md").write_text(
                f"# Proposal: {topic}\n\n## Raw model output\n\n{raw_code}\n",
                encoding="utf-8",
            )
            # ".py.txt" so it is never auto-imported/executed as a module.
            (proposal_dir / "proposal.py.txt").write_text(code, encoding="utf-8")

        await run_blocking(self._executor, _write_all)
        await self._mark_processed(topic)

        logger.warning(
            "Autonomy: NEW PROPOSAL written to %s for review (not applied). "
            "Topic: %s",
            proposal_dir,
            topic,
        )
        return proposal_dir
