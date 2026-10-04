"""Watch ``inbox/`` and turn each finished game file into a draft job.

A file still being copied or downloaded keeps growing, so a file only counts once its
size has been the same on two polls in a row.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlmodel import select

from ..config import get_settings
from ..db import Job, session_scope

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".ts", ".m2ts", ".mov", ".m4v", ".webm", ".avi", ".mpg", ".mpeg"}
POLL_SECONDS = 10.0


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


class InboxWatcher:
    def __init__(self) -> None:
        self._sizes: dict[str, int] = {}

    def scan(self) -> list[int]:
        """One poll. Returns the ids of draft jobs created."""
        inbox = get_settings().inbox_path
        created: list[int] = []
        if not inbox.is_dir():
            return created
        current: dict[str, int] = {}
        for path in sorted(inbox.iterdir()):
            if not path.is_file() or not is_video(path):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            key = str(path.resolve())
            current[key] = size
            stable = size > 0 and self._sizes.get(key) == size
            if not stable:
                continue
            with session_scope() as s:
                exists = s.exec(select(Job).where(Job.source_path == key)).first()
                if exists is None:
                    job = Job(source_path=key, source_name=path.name, status="draft", from_inbox=True,
                              message="Found in inbox. Fill in the game details to start.")
                    s.add(job)
                    s.flush()
                    created.append(job.id)
                    log.info("inbox: new draft job %s for %s", job.id, path.name)
        self._sizes = current
        return created

    def pending(self) -> list[str]:
        """Files seen but not yet stable (still being written)."""
        return [k for k in self._sizes]
