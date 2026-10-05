"""Keeping the disk from filling up on a copy that runs unattended.

Two things grow: uploaded game files (gigabytes each) and exported videos (hundreds of
megabytes each). Both can be given a shelf life in days, and an export about to start can
ask for room, which removes the oldest finished exports first.

Nothing here runs unless a retention is set or room is asked for, so a copy on your own
computer keeps everything, as before.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import select

from ..config import get_settings
from ..db import Export, Job, session_scope

log = logging.getLogger(__name__)
SIDE_FILES = (".cutlist.json", ".caption.txt")
KEEP_FRESH = timedelta(hours=1)  # never remove something made in the last hour to make room


def _aware(moment: datetime | None) -> datetime:
    if moment is None:
        return datetime.fromtimestamp(0, UTC)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _remove_export_files(path: Path) -> None:
    path.unlink(missing_ok=True)
    for suffix in SIDE_FILES:
        path.with_name(path.stem + suffix).unlink(missing_ok=True)


def sweep(now: datetime | None = None) -> dict:
    """Remove uploads and exports past their shelf life. Returns what was removed."""
    settings = get_settings()
    now = now or datetime.now(UTC)
    removed = {"uploads": 0, "exports": 0}

    if settings.upload_retention_days > 0 and settings.uploads_path.is_dir():
        limit = now - timedelta(days=settings.upload_retention_days)
        with session_scope() as s:
            jobs = s.exec(select(Job)).all()
            last_used: dict[str, datetime] = {}
            busy: set[str] = set()
            for job in jobs:
                key = str(Path(job.source_path))
                last_used[key] = max(last_used.get(key, _aware(None)), _aware(job.updated_at))
                if job.task is not None:
                    busy.add(key)
        for path in settings.uploads_path.iterdir():
            if not path.is_file() or path.suffix in (".part", ".json"):
                continue  # unfinished uploads have their own one-day clean-up
            key = str(path.resolve())
            used = last_used.get(key) or datetime.fromtimestamp(path.stat().st_mtime, UTC)
            if key not in busy and used < limit:
                path.unlink(missing_ok=True)
                removed["uploads"] += 1
                log.info("janitor: removed upload %s (unused since %s)", path.name, used.date())

    if settings.export_retention_days > 0:
        limit = now - timedelta(days=settings.export_retention_days)
        with session_scope() as s:
            old = [e for e in s.exec(select(Export).where(Export.status == "done")).all() if e.path and _aware(e.created_at) < limit]
            for export in old:
                path = Path(export.path)
                if path.exists():
                    _remove_export_files(path)
                    removed["exports"] += 1
                    log.info("janitor: removed export %s (made %s)", path.name, _aware(export.created_at).date())
    return removed


def make_room(need_bytes: int, now: datetime | None = None) -> int:
    """Free at least ``need_bytes`` where exports are written by removing the oldest
    finished exports. Returns how many were removed. Does nothing if there is room."""
    settings = get_settings()
    folder = settings.exports_path
    folder.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(UTC)
    removed = 0
    with session_scope() as s:
        done = sorted(s.exec(select(Export).where(Export.status == "done")).all(), key=lambda e: _aware(e.created_at))
        for export in done:
            if shutil.disk_usage(folder).free >= need_bytes:
                break
            path = Path(export.path) if export.path else None
            if path is None or not path.exists() or now - _aware(export.created_at) < KEEP_FRESH:
                continue
            _remove_export_files(path)
            removed += 1
            log.info("janitor: removed export %s to make room", path.name)
    return removed
