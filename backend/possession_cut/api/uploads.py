"""Receiving a game file from the browser.

A web page cannot hand the engine a path the way the file browser does, so the file is
sent in pieces and written to the uploads folder. Pieces are appended in order and the
page can ask how much has arrived, so a dropped connection resumes where it stopped
instead of starting a multi-gigabyte upload again.

The uploaded copy belongs to the app: deleting the job deletes it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..config import get_settings
from ..db import Job, session_scope
from ..pipeline.probe import ProbeError, probe_file
from ..worker.inbox import VIDEO_EXTENSIONS
from ..worker.runner import job_dir

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/uploads")

CHUNK_BYTES = 8 * 1024 * 1024  # what the page is told to send per request
MAX_CHUNK_BYTES = 64 * 1024 * 1024
STALE_SECONDS = 24 * 3600  # an unfinished upload nobody has touched for a day is dropped
HEADROOM_BYTES = 2 * 1024**3  # disk space to leave free for analysis files and the export
LOCK_WAIT_SECONDS = 6.0  # how long to wait out another program holding the file
UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class UploadStart(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)


def safe_name(name: str) -> str:
    """The file's own name, made safe to create inside the uploads folder."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = FORBIDDEN.sub("_", base).strip(" .")
    stem, ext = os.path.splitext(base)
    return f"{(stem or 'game')[:150]}{ext.lower()}"


def unique_path(folder: Path, name: str) -> Path:
    candidate = folder / name
    stem, ext = os.path.splitext(name)
    n = 2
    while candidate.exists():
        candidate = folder / f"{stem} ({n}){ext}"
        n += 1
    return candidate


def is_upload(path: str | Path) -> bool:
    """Was this source file received by upload (and so owned by the app)?"""
    try:
        Path(path).resolve().relative_to(get_settings().uploads_path.resolve())
        return True
    except (ValueError, OSError):
        return False


def _files(upload_id: str) -> tuple[Path, Path]:
    if not UPLOAD_ID.match(upload_id):
        raise HTTPException(404, "Upload not found")
    folder = get_settings().uploads_path
    return folder / f"{upload_id}.part", folder / f"{upload_id}.json"


def _load(upload_id: str) -> tuple[dict, Path, Path]:
    part, meta_path = _files(upload_id)
    if not (part.exists() and meta_path.exists()):
        raise HTTPException(404, "Upload not found. It may have been cancelled or expired; start it again.")
    return json.loads(meta_path.read_text(encoding="utf-8")), part, meta_path


def _status(upload_id: str, meta: dict, part: Path) -> dict:
    return {
        "id": upload_id,
        "name": meta["name"],
        "size": meta["size"],
        "received": part.stat().st_size,
        "chunk_size": CHUNK_BYTES,
    }


def _busy() -> HTTPException:
    return HTTPException(503, "Another program (antivirus or file indexing) is holding the upload for a moment. Trying again.")


def _when_unlocked(action):
    """Run a file operation, waiting out the short locks Windows antivirus and indexers
    take on a file that is being written."""
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise _busy() from None
            time.sleep(0.1)


def drop_stale(folder: Path, now: float | None = None) -> int:
    """Remove unfinished uploads that have not grown for a day."""
    now = now or time.time()
    dropped = 0
    for part in folder.glob("*.part"):
        try:
            if now - part.stat().st_mtime > STALE_SECONDS:
                part.unlink(missing_ok=True)
                part.with_suffix(".json").unlink(missing_ok=True)
                dropped += 1
        except OSError:
            continue
    return dropped


@router.post("", status_code=201)
def start_upload(body: UploadStart) -> dict:
    name = safe_name(body.name)
    if os.path.splitext(name)[1] not in VIDEO_EXTENSIONS:
        kinds = ", ".join(sorted(e.lstrip(".").upper() for e in VIDEO_EXTENSIONS))
        raise HTTPException(400, f"{body.name} is not a video file this app reads ({kinds}).")
    folder = get_settings().uploads_path
    folder.mkdir(parents=True, exist_ok=True)
    drop_stale(folder)
    free = shutil.disk_usage(folder).free
    if free < body.size + HEADROOM_BYTES:
        raise HTTPException(
            507,
            f"Not enough disk space for this file: it is {body.size / 1024**3:.1f} GB and "
            f"{free / 1024**3:.1f} GB is free where uploads are kept ({folder}).",
        )
    upload_id = uuid.uuid4().hex
    part, meta_path = _files(upload_id)
    meta = {"name": name, "size": body.size, "created": time.time()}
    part.touch()
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return _status(upload_id, meta, part)


@router.get("/{upload_id}")
def upload_status(upload_id: str) -> dict:
    meta, part, _ = _load(upload_id)
    return _status(upload_id, meta, part)


@router.put("/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = Query(ge=0)) -> dict:
    """Append one piece. ``offset`` must be where the file currently ends; if it is not
    (a retry after a half-sent piece), the answer says where to continue from."""
    meta, part, _ = _load(upload_id)
    have = part.stat().st_size
    if offset != have:
        raise HTTPException(409, {"received": have, "message": "Continue from the byte count in 'received'."})
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            f = part.open("ab")
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                raise _busy() from None
            await asyncio.sleep(0.1)
    written = 0
    too_much = False
    with f:
        async for piece in request.stream():
            written += len(piece)
            if written > MAX_CHUNK_BYTES or have + written > meta["size"]:
                too_much = True
                break
            f.write(piece)
    if too_much:
        _when_unlocked(lambda: os.truncate(part, have))
        raise HTTPException(400, "More data arrived than the upload was started for.")
    return {"received": have + written, "size": meta["size"]}


@router.post("/{upload_id}/complete", status_code=201)
def complete_upload(upload_id: str) -> dict:
    """Check the finished file is a video and open a draft job for it."""
    from .routes import job_out  # routes imports this module's helpers

    meta, part, meta_path = _load(upload_id)
    have = part.stat().st_size
    if have != meta["size"]:
        raise HTTPException(409, {"received": have, "message": "The upload is not finished."})
    final = unique_path(part.parent, meta["name"])
    _when_unlocked(lambda: part.replace(final))
    meta_path.unlink(missing_ok=True)
    try:
        probe = probe_file(final)
    except ProbeError as exc:
        final.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:  # ffprobe itself is missing
        raise HTTPException(500, str(exc)) from exc
    with session_scope() as s:
        job = Job(
            source_path=str(final.resolve()), source_name=meta["name"], status="draft", probe=probe.to_dict(),
            message="Uploaded. Fill in the game details to start.",
        )
        s.add(job)
        s.flush()
        (job_dir(job.id) / "probe.json").write_text(json.dumps(probe.to_dict(), indent=1), encoding="utf-8")
        log.info("upload: new draft job %s for %s", job.id, final.name)
        return job_out(job)


@router.delete("/{upload_id}")
def cancel_upload(upload_id: str) -> dict:
    part, meta_path = _files(upload_id)
    part.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return {"cancelled": upload_id}
