"""The worker process: polls the jobs table and runs one task at a time.

    python -m possession_cut.worker
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback

from sqlmodel import select

from ..config import get_settings
from ..db import Export, Job, get_engine, session_scope, touch, utcnow
from .inbox import POLL_SECONDS, InboxWatcher
from .janitor import sweep
from .runner import PREVIOUS_STATUS, TASKS, Cancelled, job_dir

log = logging.getLogger("possession_cut.worker")
POLL_INTERVAL = 1.0


def recover() -> int:
    """Tasks that were running when the worker last died go back in the queue."""
    with session_scope() as s:
        stuck = s.exec(select(Job).where(Job.task != None, Job.task_started_at != None)).all()  # noqa: E711
        for job in stuck:
            job.task_started_at = None
            job.message = "Restarting after the worker stopped"
            s.add(job)
        return len(stuck)


def claim_next() -> tuple[int, str] | None:
    with session_scope() as s:
        job = s.exec(
            select(Job).where(Job.task != None, Job.task_started_at == None).order_by(Job.updated_at)  # noqa: E711
        ).first()
        if job is None:
            return None
        if (job.task_payload or {}).get("cancel"):
            job.task = None
            job.task_payload = {}
            s.add(job)
            return None
        job.task_started_at = utcnow()
        s.add(job)
        return job.id, job.task


def run_one(job_id: int, task: str) -> None:
    handler = None
    try:
        logs = job_dir(job_id) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(logs / "worker.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger("possession_cut").addHandler(handler)
    except OSError:
        handler = None
    log.info("job %s: %s started", job_id, task)
    started = time.time()
    status: str | None = None
    error: str | None = None
    try:
        TASKS[task](job_id)
        log.info("job %s: %s finished in %.1fs", job_id, task, time.time() - started)
    except (Cancelled, InterruptedError):
        status = PREVIOUS_STATUS.get(task, "draft")
        log.info("job %s: %s cancelled", job_id, task)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        log.error("job %s: %s failed\n%s", job_id, task, traceback.format_exc())
    finally:
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is not None:
                payload = dict(job.task_payload or {})
                if status is not None:
                    export_id = payload.get("export_id")
                    if task == "export" and export_id:
                        export = s.get(Export, export_id)
                        if export is not None and export.status != "done":
                            export.status = "failed"
                            export.error = error or "cancelled"
                            s.add(export)
                    if status == "failed":
                        job.status = "failed"
                        job.error = error
                        job.message = f"{task.title()} failed"
                    else:
                        job.status = status
                        job.message = "Cancelled"
                        job.progress = 0.0
                job.task = None
                job.task_started_at = None
                job.task_payload = {}
                touch(job)
                s.add(job)
        if handler is not None:
            logging.getLogger("possession_cut").removeHandler(handler)
            handler.close()


def _inbox_loop(stop: threading.Event) -> None:
    watcher = InboxWatcher()
    while not stop.is_set():
        try:
            watcher.scan()
        except Exception:
            log.exception("inbox scan failed")
        stop.wait(POLL_SECONDS)


def _janitor_loop(stop: threading.Event) -> None:
    """Hourly: remove uploads and exports past the shelf life set in the settings."""
    while not stop.is_set():
        try:
            removed = sweep()
            if any(removed.values()):
                log.info("janitor: %s", removed)
        except Exception:
            log.exception("janitor sweep failed")
        stop.wait(3600.0)


def _heartbeat_loop(stop: threading.Event) -> None:
    """Touch a file every few seconds so the API can tell the worker is alive."""
    path = get_settings().heartbeat_path
    while not stop.is_set():
        try:
            path.write_text(str(time.time()))
        except OSError:
            pass
        stop.wait(3.0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    settings = get_settings()
    settings.ensure_dirs()
    get_engine()
    requeued = recover()
    if requeued:
        log.info("requeued %d interrupted task(s)", requeued)
    stop = threading.Event()
    threading.Thread(target=_inbox_loop, args=(stop,), daemon=True, name="inbox").start()
    threading.Thread(target=_janitor_loop, args=(stop,), daemon=True, name="janitor").start()
    log.info("worker ready (analysis workers: %d, sample fps: %s, inbox: %s)",
             settings.workers, settings.ocr_sample_fps, settings.inbox_path)
    beat = threading.Thread(target=_heartbeat_loop, args=(stop,), daemon=True, name="heartbeat")
    beat.start()
    try:
        while True:
            claimed = claim_next()
            if claimed is None:
                time.sleep(POLL_INTERVAL)
                continue
            run_one(*claimed)
    except KeyboardInterrupt:
        log.info("worker stopping")
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
