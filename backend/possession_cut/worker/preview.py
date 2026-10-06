"""Background thread of the worker: makes preview copies for jobs that want one.

It runs beside the task loop with a few encoder threads, so a calibration or analysis
started meanwhile is barely slowed. A copy is written as ``proxy.part.mp4`` and renamed
when complete, so the page never plays a half-written file. A job deleted mid-way, or a
source wiped by a redeploy, stops the encode quietly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from sqlmodel import select

from ..db import Job, session_scope
from ..pipeline.export import make_proxy
from ..pipeline.probe import Probe
from ..preview import failed_path, part_path, progress_path, proxy_path, wants_preview

log = logging.getLogger("possession_cut.worker.preview")
POLL_SECONDS = 15.0
THREADS = 3


def next_job() -> tuple[int, dict] | None:
    """The newest job that wants a preview copy and has none, with its probe."""
    with session_scope() as s:
        jobs = s.exec(select(Job).order_by(Job.updated_at.desc())).all()
        for job in jobs:
            if job.id is None or not wants_preview(job.probe):
                continue
            if proxy_path(job.id).exists() or failed_path(job.id).exists():
                continue
            if not Path(job.source_path).exists():
                continue
            return job.id, dict(job.probe)
    return None


def build(job_id: int, probe_dict: dict) -> bool:
    probe = Probe.from_dict(probe_dict)
    part, final = part_path(job_id), proxy_path(job_id)
    last_written = [0.0]

    def progress(fraction: float, _message: str) -> None:
        now = time.time()
        if now - last_written[0] < 2.0:
            return
        last_written[0] = now
        try:
            progress_path(job_id).write_text(json.dumps({"fraction": round(fraction, 3), "at": now}), encoding="utf-8")
        except OSError:
            pass

    def gone() -> bool:
        # the job was deleted (its folder with it), or the disk was wiped under us
        return not Path(probe.path).exists() or not part.parent.exists()

    try:
        make_proxy(probe, part, progress=progress, should_stop=gone, seekable=True, threads=THREADS)
        part.replace(final)
        return True
    except Exception as exc:
        part.unlink(missing_ok=True)
        if gone():
            return False
        try:
            failed_path(job_id).write_text(f"{type(exc).__name__}: {exc}"[:500], encoding="utf-8")
        except OSError:
            pass
        log.warning("preview copy for job %s failed: %s", job_id, exc)
        return False
    finally:
        progress_path(job_id).unlink(missing_ok=True)


def preview_loop(stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            found = next_job()
            if found is not None:
                job_id, probe = found
                log.info("preview: making a copy for job %s", job_id)
                started = time.time()
                if build(job_id, probe):
                    log.info("preview: job %s copy ready in %.0fs", job_id, time.time() - started)
                continue  # there may be another waiting
        except Exception:
            log.exception("preview loop failed")
        stop.wait(POLL_SECONDS)
