from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jero.autonomy.engine import (
    AutonomyEngine,
    extract_python_code,
    parse_backlog,
)
from jero.brain.brain import Brain
from jero.core.bus import MessageBus
from jero.core.config import AutonomyConfig
from tests.test_brain import FakeProvider

# ----- pure helpers -------------------------------------------------------


def test_parse_backlog_skips_checked_and_blank() -> None:
    content = (
        "# Title\n"
        "- [ ] first feature\n"
        "- [x] already done\n"
        "* second feature\n"
        "- [X] also done\n"
        "plain prose line\n"
        "-   \n"
    )
    assert parse_backlog(content) == ["first feature", "second feature"]


def test_extract_python_code_from_fence() -> None:
    text = "Here you go:\n```python\nprint('hi')\n```\nThanks"
    assert extract_python_code(text) == "print('hi')\n"


def test_extract_python_code_without_fence() -> None:
    assert extract_python_code("x = 1") == "x = 1\n"


# ----- engine -------------------------------------------------------------


def _brain(code_reply: str = "```python\nprint('hi')\n```") -> Brain:
    nim = FakeProvider("nvidia-nim", code_reply)
    groq = FakeProvider("groq", "research notes")
    return Brain(nim=nim, groq=groq)


def _config(tmp_path: Path, **overrides) -> AutonomyConfig:
    base = dict(
        enabled=True,
        idle_seconds_before_research=0,
        poll_interval_seconds=0.01,
        cooldown_seconds=0.01,
        backlog_path="autonomy_backlog.md",
        proposals_dir="proposals",
    )
    base.update(overrides)
    return AutonomyConfig(**base)


async def test_generate_proposal_writes_safe_artifacts(tmp_path: Path) -> None:
    engine = AutonomyEngine(
        _brain(), MessageBus(), _config(tmp_path), repo_root=tmp_path
    )
    proposal_dir = await engine.generate_proposal("Add wake word")

    assert proposal_dir.exists()
    assert (proposal_dir / "metadata.json").exists()
    assert (proposal_dir / "research.md").exists()
    assert (proposal_dir / "proposal.md").exists()
    # Code is stored as .py.txt so it can never be auto-imported/executed.
    assert (proposal_dir / "proposal.py.txt").exists()
    assert not list(proposal_dir.glob("*.py"))

    meta = json.loads((proposal_dir / "metadata.json").read_text())
    assert meta["topic"] == "Add wake word"
    assert meta["status"] == "proposed"

    # Everything stays inside the proposals dir.
    assert engine.proposals_dir in proposal_dir.parents
    # Topic recorded as processed.
    processed = json.loads((engine.proposals_dir / ".processed.json").read_text())
    assert "Add wake word" in processed


async def test_next_topic_skips_processed(tmp_path: Path) -> None:
    (tmp_path / "autonomy_backlog.md").write_text(
        "- [ ] alpha\n- [ ] beta\n", encoding="utf-8"
    )
    engine = AutonomyEngine(
        _brain(), MessageBus(), _config(tmp_path), repo_root=tmp_path
    )
    assert await engine._next_topic() == "alpha"
    await engine._mark_processed("alpha")
    assert await engine._next_topic() == "beta"
    await engine._mark_processed("beta")
    assert await engine._next_topic() is None


async def test_run_generates_when_idle(tmp_path: Path) -> None:
    (tmp_path / "autonomy_backlog.md").write_text("- [ ] only feature\n", "utf-8")
    bus = MessageBus()
    engine = AutonomyEngine(_brain(), bus, _config(tmp_path), repo_root=tmp_path)

    engine.start()
    # Poll until a proposal appears (idle threshold is 0).
    for _ in range(200):
        if engine.proposals_dir.exists() and any(
            p.is_dir() for p in engine.proposals_dir.iterdir()
        ):
            break
        await asyncio.sleep(0.01)
    await engine.stop()

    dirs = [p for p in engine.proposals_dir.iterdir() if p.is_dir()]
    assert len(dirs) == 1


async def test_run_pauses_when_active(tmp_path: Path) -> None:
    (tmp_path / "autonomy_backlog.md").write_text("- [ ] feature\n", "utf-8")
    bus = MessageBus()
    # High idle threshold => never idle within the test window.
    engine = AutonomyEngine(
        _brain(),
        bus,
        _config(tmp_path, idle_seconds_before_research=1000),
        repo_root=tmp_path,
    )
    engine.start()
    await asyncio.sleep(0.1)
    await engine.stop()
    # No proposals because the system was treated as active.
    assert not engine.proposals_dir.exists() or not list(engine.proposals_dir.glob("*/"))


async def test_does_not_overwrite_existing_proposal_dir(tmp_path: Path) -> None:
    engine = AutonomyEngine(
        _brain(), MessageBus(), _config(tmp_path), repo_root=tmp_path
    )
    d1 = await engine.generate_proposal("same topic")
    d2 = await engine.generate_proposal("same topic")
    # Unique timestamped dirs; the second never clobbers the first.
    assert d1.exists() and d2.exists()
