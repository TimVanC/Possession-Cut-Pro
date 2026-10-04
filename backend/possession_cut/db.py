"""SQLite storage (SQLModel): jobs, broadcaster templates, clips, exports.

The API and the worker are separate processes sharing this file, so the database runs
in WAL mode with a generous busy timeout.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column, event
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from .config import get_settings

JOB_STATUSES = ("draft", "calibrating", "ready", "analyzing", "review", "exporting", "done", "failed")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    status: str = Field(default="draft", index=True)
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    source_path: str
    source_name: str = ""
    sport: str = "nba"
    game_id: str | None = None
    game: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    team: str | None = None
    start_spec: dict[str, Any] = Field(default_factory=lambda: {"mode": "start"}, sa_column=Column(JSON))
    end_spec: dict[str, Any] = Field(default_factory=lambda: {"mode": "end"}, sa_column=Column(JSON))
    options: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    template_id: int | None = Field(default=None, foreign_key="template.id")
    probe: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    calibration: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    claude_spent_usd: float = 0.0
    # Work queued for the worker: "calibrate" | "analyze" | "export". None = nothing pending.
    task: str | None = Field(default=None, index=True)
    task_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    task_started_at: datetime | None = None
    from_inbox: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Template(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    sport: str = "nba"
    broadcaster: str = ""
    bug: list[float] = Field(default_factory=list, sa_column=Column(JSON))
    fields: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    crop: list[float] = Field(default_factory=list, sa_column=Column(JSON))
    frame_aspect: float = 16 / 9
    reference_image: str = ""  # path under data/templates
    mask_image: str = ""
    source: str = "local"  # claude | local | manual
    use_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    last_used_at: datetime | None = None


class Clip(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    order: int = 0
    src_in: float = 0.0
    src_out: float = 0.0
    # Kept ranges inside [src_in, src_out]. More than one when a replay or the walk-up
    # between free throws was cut out of the middle.
    segments: list[list[float]] = Field(default_factory=list, sa_column=Column(JSON))
    # what analysis produced, so "reset" can undo manual nudges
    auto_segments: list[list[float]] = Field(default_factory=list, sa_column=Column(JSON))
    auto_in: float = 0.0
    auto_out: float = 0.0
    team: str = ""
    period: int | None = None
    clock: float | None = None
    score_before: int | None = None
    score_after: int | None = None
    score_away: int | None = None
    score_home: int | None = None
    points: int = 0
    kind: str = "field_goal"
    scorer: str = ""
    description: str = ""
    confidence: float = 0.0
    enabled: bool = True
    pbp_event_id: str | None = None
    warnings: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    events: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    thumbnail: str = ""


class Export(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    status: str = "pending"  # pending | rendering | done | failed
    path: str = ""
    title: str = ""
    settings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    caption: str = ""
    cutlist_path: str = ""
    caption_path: str = ""
    duration: float = 0.0
    size_bytes: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_dirs()
        _engine = create_engine(
            f"sqlite:///{settings.db_path.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi_conn, _record):  # pragma: no cover - driver hook
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        SQLModel.metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Drop the cached engine (tests point DATA_DIR somewhere new)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def session_scope() -> Iterator[Session]:
    session = Session(get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def touch(job: Job) -> None:
    job.updated_at = utcnow()
