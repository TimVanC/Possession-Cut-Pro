"""HTTP API. Queues work for the worker and serves media; does no heavy lifting itself.

Two deliberate exceptions to "the worker does everything", both about responsiveness:
game lookup (a small cached HTTP call that must not wait behind a running analysis) and
the calibration screen's live read-out (OCR on a handful of already-extracted frames).
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import select

from .. import __version__
from ..ai.claude import ClaudeClient
from ..config import ffmpeg_bin, get_settings, is_allowed_path
from ..db import Clip, Export, Job, Template, session_scope, touch
from ..edits import move_edges
from ..pipeline import templates as tmpl
from ..pipeline.calibration import Calibration, make_reader, read_frame, recalibrate_manual
from ..pipeline.frames import FrameError, extract_frame
from ..pipeline.geometry import compute_crop, crop_to_norm, norm_box
from ..pipeline.matching import load_sidecar, sides_of
from ..pipeline.ocr import get_engine
from ..pipeline.probe import Probe, ProbeError, probe_file
from ..pipeline.window import biggest_run_start
from ..preview import preview_state, proxy_path
from ..sports import PlayByPlayUnavailable, available_sports, get_adapter
from ..worker.inbox import VIDEO_EXTENSIONS, is_video
from ..worker.janitor import remove_export_files
from ..worker.runner import job_dir, load_calibration, save_calibration
from .auth import signed_in
from .progress import progress_view
from .uploads import is_upload

router = APIRouter(prefix="/api")

BUSY = ("calibrating", "analyzing", "exporting")
# which commit a hosted copy was built from (Railway provides it), so "is the new build live?" has an answer
BUILD = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")[:7]


# -- helpers ----------------------------------------------------------------------------


def _job_or_404(s, job_id: int) -> Job:
    job = s.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _proxy_path(job_id: int) -> Path:
    return proxy_path(job_id)


def _local_only() -> None:
    """Refuse what only makes sense when the engine is on the user's own computer."""
    if get_settings().hosted:
        raise HTTPException(403, "Not available on a hosted copy. Upload the file instead.")


def job_out(job: Job) -> dict[str, Any]:
    probe = job.probe or {}
    cal = job.calibration or {}
    summary = job.summary or {}
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": round(job.progress or 0.0, 4),
        "message": job.message,
        "busy": job.task is not None,
        **progress_view(job),
        "error": job.error,
        "source_path": job.source_path,
        "source_name": job.source_name,
        "source_exists": Path(job.source_path).exists(),
        "sport": job.sport,
        "game_id": job.game_id,
        "game": job.game or {},
        "team": job.team,
        "start_spec": job.start_spec or {"mode": "start"},
        "end_spec": job.end_spec or {"mode": "end"},
        "options": job.options or {},
        "template_id": job.template_id,
        "from_inbox": job.from_inbox,
        "uploaded": is_upload(job.source_path),
        "probe": {k: probe.get(k) for k in ("duration", "width", "height", "display_width", "fps", "video_codec",
                                              "audio_codec", "size_bytes", "browser_playable", "container")} if probe else None,
        "calibration": {k: cal.get(k) for k in ("confidence", "source", "confirmed", "template_name", "teams",
                                                "broadcaster", "warnings", "crop", "bug")} if cal else None,
        "summary": summary or None,
        **(preview_state(job.id, probe) if job.id else {"media_ready": False, "media_source": None, "preview": None}),
        "claude_spent_usd": round(job.claude_spent_usd or 0.0, 4),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


def clip_out(clip: Clip) -> dict[str, Any]:
    return {
        "id": clip.id, "job_id": clip.job_id, "order": clip.order, "enabled": clip.enabled,
        "src_in": clip.src_in, "src_out": clip.src_out, "segments": clip.segments,
        "auto_in": clip.auto_in, "auto_out": clip.auto_out,
        "duration": round(sum(b - a for a, b in clip.segments), 3),
        "team": clip.team, "period": clip.period, "clock": clip.clock,
        "score_before": clip.score_before, "score_after": clip.score_after,
        "score_away": clip.score_away, "score_home": clip.score_home,
        "points": clip.points, "kind": clip.kind, "scorer": clip.scorer, "description": clip.description,
        "confidence": clip.confidence, "pbp_event_id": clip.pbp_event_id, "warnings": clip.warnings or [],
        "events": clip.events or [],
        "thumbnail": f"/api/media/{clip.job_id}/thumb/{clip.order}?v={int((clip.auto_in or 0) * 10)}" if clip.thumbnail else None,
        "edited": abs(clip.src_in - clip.auto_in) > 1e-6 or abs(clip.src_out - clip.auto_out) > 1e-6,
    }


def export_out(export: Export) -> dict[str, Any]:
    return {
        "id": export.id, "job_id": export.job_id, "status": export.status, "title": export.title,
        "path": export.path, "file_name": Path(export.path).name if export.path else "",
        "caption": export.caption, "cutlist_path": export.cutlist_path, "caption_path": export.caption_path,
        "duration": export.duration, "size_bytes": export.size_bytes, "error": export.error,
        "settings": export.settings or {},
        "url": f"/api/exports/{export.id}/file" if export.status == "done" and export.path and Path(export.path).exists() else None,
        "file_removed": export.status == "done" and not (export.path and Path(export.path).exists()),
        "created_at": export.created_at.isoformat() if export.created_at else None,
    }


def _queue(job: Job, task: str, status: str, payload: dict | None = None, message: str = "Queued") -> None:
    if job.task is not None:
        raise HTTPException(409, f"This job is busy ({job.task}). Cancel it or wait for it to finish.")
    job.task = task
    job.task_payload = payload or {}
    job.task_started_at = None
    job.status = status
    job.stage = "queued"
    job.progress = 0.0
    job.message = message
    job.error = None
    touch(job)


def worker_alive() -> bool:
    try:
        return time.time() - get_settings().heartbeat_path.stat().st_mtime < 12
    except OSError:
        return False


# -- health / config ----------------------------------------------------------------------


@router.get("/health")
def health(request: Request) -> dict:
    settings = get_settings()
    auth = {
        "required": settings.auth_required,
        "authenticated": signed_in(request, settings),
        "configured": bool(settings.app_password) or not settings.auth_required,
    }
    if not auth["authenticated"]:
        # enough for the page to show the sign-in screen, and nothing about the machine
        return {"ok": True, "version": __version__, "build": BUILD, "hosted": settings.hosted, "auth": auth}
    try:
        ffmpeg_bin()
        ffmpeg_ok = True
    except FileNotFoundError:
        ffmpeg_ok = False
    claude = ClaudeClient()
    return {
        "ok": True,
        "version": __version__,
        "build": BUILD,
        "hosted": settings.hosted,
        "auth": auth,
        "ffmpeg": ffmpeg_ok,
        "worker": worker_alive(),
        "claude": {
            "configured": claude.available,
            "model": settings.claude_model,
            "budget_per_job_usd": settings.claude_budget_per_job_usd,
            "note": None if claude.available else claude.unavailable_reason,
            "needs_workspace": bool(settings.anthropic_api_key) and not settings.anthropic_workspace_id,
        },
        "sample_fps": settings.ocr_sample_fps,
        "inbox_dir": str(settings.inbox_path),
        "uploads_dir": str(settings.uploads_path),
        "exports_dir": str(settings.exports_path),
    }


@router.get("/sports")
def sports() -> list[dict]:
    out = []
    for item in available_sports():
        adapter = get_adapter(item["key"])
        out.append(
            {
                **item,
                "teams": adapter.teams(),
                "period_label": adapter.period_label,
                "periods": adapter.regulation_periods,
                "period_seconds": adapter.period_seconds,
                "has_clock": adapter.has_clock,
                "fields": [{"name": f.name, "required": f.required, "description": f.description} for f in adapter.bug_fields],
                "options": list(adapter.options),
            }
        )
    return out


# -- file browser ---------------------------------------------------------------------------


@router.get("/fs/browse")
def browse(path: str = "") -> dict:
    _local_only()
    settings = get_settings()
    roots = [str(r) for r in settings.roots]
    if not path:
        return {
            "path": "", "parent": None, "roots": roots,
            "entries": [{"name": r, "path": r, "is_dir": True, "size": None, "modified": None, "is_video": False} for r in roots],
        }
    target = Path(path).expanduser()
    if not is_allowed_path(target):
        raise HTTPException(403, "That folder is outside the allowed roots (ALLOWED_ROOTS in .env).")
    target = target.resolve()
    if not target.is_dir():
        raise HTTPException(404, "Folder not found")
    entries = []
    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise HTTPException(403, "Windows will not let this folder be listed.") from exc
    for child in children:
        if child.name.startswith(".") or child.name.startswith("$"):
            continue
        try:
            is_dir = child.is_dir()
            if not is_dir and not is_video(child):
                continue
            stat = child.stat()
        except OSError:
            continue
        entries.append(
            {"name": child.name, "path": str(child), "is_dir": is_dir,
             "size": None if is_dir else stat.st_size, "modified": stat.st_mtime, "is_video": not is_dir}
        )
    parent = target.parent if is_allowed_path(target.parent) and target.parent != target else None
    return {"path": str(target), "parent": str(parent) if parent else "", "roots": roots, "entries": entries}


@router.get("/inbox")
def inbox() -> dict:
    settings = get_settings()
    files = []
    if settings.inbox_path.is_dir():
        for p in sorted(settings.inbox_path.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                files.append({"name": p.name, "path": str(p.resolve()), "size": p.stat().st_size})
    with session_scope() as s:
        drafts = s.exec(select(Job).where(Job.status == "draft", Job.from_inbox == True)).all()  # noqa: E712
        known = {j.source_path for j in s.exec(select(Job)).all()}
    return {
        "dir": str(settings.inbox_path),
        "files": files,
        "waiting": len(drafts) + sum(1 for f in files if f["path"] not in known),
        "drafts": [j.id for j in drafts],
    }


# -- games ------------------------------------------------------------------------------------


@router.get("/games")
def games(sport: str = "nba", date: str = "", team: str | None = None) -> list[dict]:
    if not date:
        raise HTTPException(400, "Pick a date.")
    try:
        adapter = get_adapter(sport)
        return [g.to_dict() for g in adapter.find_games(date, team or None)]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except PlayByPlayUnavailable as exc:
        raise HTTPException(502, f"Game lookup is unavailable right now: {exc}") from exc
    except Exception as exc:
        raise HTTPException(502, f"Game lookup failed: {type(exc).__name__}: {exc}") from exc


@router.get("/games/run-start")
def run_start(sport: str, game_id: str, team: str, source_path: str | None = None) -> dict:
    """Where "auto: start of biggest run" would begin, shown in the form before analysis."""
    try:
        adapter = get_adapter(sport)
        sidecar = load_sidecar(source_path) if source_path and is_allowed_path(Path(source_path)) else None
        pbp, sides = sidecar if sidecar else (None, {})
        if pbp is None:
            pbp = adapter.fetch_pbp(game_id)
            sides = sides_of(pbp)
    except PlayByPlayUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    side = next((k for k, v in sides.items() if v.upper() == team.upper()), None)
    if side is None:
        return {"available": False, "reason": f"{team} did not play in this game."}
    run = biggest_run_start(pbp, side)
    if run is None:
        return {"available": True, "trailed": False, "label": f"{team} never trailed; the cut starts at the beginning."}
    clock = run.clock or 0.0
    clock_text = f"{int(clock // 60)}:{int(clock % 60):02d}" if clock >= 60 else f"{clock:.1f}"
    own, opp = (run.score_home, run.score_away) if side == "home" else (run.score_away, run.score_home)
    return {
        "available": True, "trailed": True, **run.to_dict(),
        "label": f"Down {run.deficit} ({own}-{opp}) at {adapter.format_period(run.period)} {clock_text}",
        "after": run.description,
    }


# -- jobs -----------------------------------------------------------------------------------------


class JobCreate(BaseModel):
    source_path: str
    sport: str = "nba"
    game_id: str | None = None
    game: dict = Field(default_factory=dict)
    team: str | None = None
    start_spec: dict = Field(default_factory=lambda: {"mode": "start"})
    end_spec: dict = Field(default_factory=lambda: {"mode": "end"})
    options: dict = Field(default_factory=dict)
    calibrate: bool = True


class JobUpdate(BaseModel):
    sport: str | None = None
    game_id: str | None = None
    game: dict | None = None
    team: str | None = None
    start_spec: dict | None = None
    end_spec: dict | None = None
    options: dict | None = None


def _probe_or_400(path: Path, job_id: int | None = None) -> Probe:
    try:
        return probe_file(path, (job_dir(job_id) / "probe.json") if job_id else None)
    except ProbeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/jobs", status_code=201)
def create_job(body: JobCreate) -> dict:
    path = Path(body.source_path).expanduser()
    if not is_allowed_path(path):
        raise HTTPException(403, "That file is outside the allowed roots (ALLOWED_ROOTS in .env).")
    path = path.resolve()
    if not path.is_file():
        raise HTTPException(404, "File not found")
    try:
        get_adapter(body.sport)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    probe = _probe_or_400(path)
    with session_scope() as s:
        job = Job(
            source_path=str(path), source_name=path.name, sport=body.sport, game_id=body.game_id,
            game=body.game, team=body.team, start_spec=body.start_spec, end_spec=body.end_spec,
            options=body.options, probe=probe.to_dict(), status="draft",
        )
        s.add(job)
        s.flush()
        (job_dir(job.id) / "probe.json").write_text(json.dumps(probe.to_dict(), indent=1), encoding="utf-8")
        if body.calibrate:
            _queue(job, "calibrate", "calibrating", message="Waiting for the worker")
        s.add(job)
        s.flush()
        return job_out(job)


@router.get("/jobs")
def list_jobs() -> list[dict]:
    with session_scope() as s:
        jobs = s.exec(select(Job).order_by(Job.created_at.desc())).all()
        return [_with_exports(s, j) for j in jobs]


def _with_exports(s, job: Job) -> dict:
    """``job_out`` plus the newest finished export and how many there are, for the list
    and the job page (a Download button where the owner looks first)."""
    done = s.exec(
        select(Export).where(Export.job_id == job.id, Export.status == "done").order_by(Export.created_at.desc())
    ).all()
    out = job_out(job)
    out["latest_export"] = export_out(done[0]) if done else None
    out["export_count"] = len(done)
    return out


@router.get("/jobs/{job_id}")
def get_job(job_id: int) -> dict:
    with session_scope() as s:
        return _with_exports(s, _job_or_404(s, job_id))


@router.patch("/jobs/{job_id}")
def update_job(job_id: int, body: JobUpdate) -> dict:
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if job.task is not None:
            raise HTTPException(409, "This job is busy. Wait for it to finish or cancel it.")
        data = body.model_dump(exclude_unset=True)
        if "sport" in data and data["sport"]:
            try:
                get_adapter(data["sport"])
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        for key, value in data.items():
            setattr(job, key, value)
        touch(job)
        s.add(job)
        s.flush()
        return job_out(job)


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int, exports: bool = False) -> dict:
    """Removes the job record and its artifacts. Exports stay unless asked (``exports=1``),
    and so does a source file the user pointed at on their own disk. A file the app
    received by upload is the app's own copy and goes with the job, unless another job
    still uses it."""
    uploaded_copy: Path | None = None
    export_files: list[Path] = []
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if job.task is not None and job.task_started_at is not None:
            raise HTTPException(409, "This job is running. Cancel it first.")
        if is_upload(job.source_path):
            shared = s.exec(select(Job).where(Job.source_path == job.source_path, Job.id != job_id)).first()
            if shared is None:
                uploaded_copy = Path(job.source_path)
        for clip in s.exec(select(Clip).where(Clip.job_id == job_id)).all():
            s.delete(clip)
        for export in s.exec(select(Export).where(Export.job_id == job_id)).all():
            if exports and export.path:
                export_files.append(Path(export.path))
            s.delete(export)
        s.delete(job)
    for path in export_files:
        remove_export_files(path)
    settings = get_settings()
    folder = settings.jobs_path / str(job_id)
    shutil.rmtree(folder, ignore_errors=True)
    if settings.scratch_dir:
        shutil.rmtree(settings._resolve(settings.scratch_dir) / str(job_id), ignore_errors=True)
    if uploaded_copy is not None:
        uploaded_copy.unlink(missing_ok=True)
    return {"deleted": job_id}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int) -> dict:
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if job.task is None:
            return job_out(job)
        if job.task_started_at is None:
            # still queued: just take it back out
            job.status = {"calibrate": "draft", "analyze": "ready", "export": "review"}.get(job.task, "draft")
            job.task, job.task_payload, job.message, job.progress = None, {}, "Cancelled", 0.0
        else:
            job.task_payload = {**(job.task_payload or {}), "cancel": True}
            job.message = "Cancelling"
        touch(job)
        s.add(job)
        s.flush()
        return job_out(job)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: int, request: Request, once: bool = False) -> StreamingResponse:
    """Server-sent events: one message whenever the job's status, stage or progress changes.

    The stream ends by itself a few seconds after the job stops being busy (the browser's
    EventSource reconnects if it still cares). ``once=1`` sends the current state and closes.
    """

    def snapshot() -> dict | None:
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is None:
                return None
            return {
                "id": job.id, "status": job.status, "stage": job.stage, "progress": round(job.progress or 0.0, 4),
                "message": job.message, "busy": job.task is not None, "error": job.error,
                **progress_view(job),
            }

    async def stream():
        last = None
        idle = 0.0
        quiet = 0.0
        while True:
            if await request.is_disconnected():
                break
            snap = await asyncio.to_thread(snapshot)
            if snap is None:
                yield "event: gone\ndata: {}\n\n"
                break
            if snap != last:
                last = snap
                idle = 0.0
                yield f"data: {json.dumps(snap)}\n\n"
            else:
                idle += 0.5
                if idle >= 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"
            if once:
                break
            quiet = 0.0 if snap["busy"] else quiet + 0.5
            if quiet >= 5.0:
                yield "event: idle\ndata: {}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# -- calibration ------------------------------------------------------------------------------------


class CalibrateRequest(BaseModel):
    ignore_templates: bool = False


class CalibrationUpdate(BaseModel):
    bug: list[float]
    fields: dict[str, list[float] | None]
    crop: list[float] | None = None
    template_name: str | None = None
    broadcaster: str | None = None
    confirmed: bool = False


class PreviewRequest(BaseModel):
    bug: list[float]
    fields: dict[str, list[float] | None]
    frame: int = 0


def _calibration_or_404(job_id: int) -> Calibration:
    path = job_dir(job_id) / "calibration.json"
    if not path.exists():
        raise HTTPException(404, "This job has not been calibrated yet.")
    return Calibration.load(path)


def _calibration_out(job_id: int, cal: Calibration) -> dict:
    data = cal.to_dict()
    for frame in data["frames"]:
        frame["url"] = f"/api/media/{job_id}/calib/{frame['file']}"
    return data


@router.post("/jobs/{job_id}/calibrate")
def calibrate_job(job_id: int, body: CalibrateRequest | None = None) -> dict:
    """Queue (re)detection of the score bug. The result is read with GET .../calibration."""
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if not Path(job.source_path).is_file():
            raise HTTPException(404, "The source file is no longer where it was.")
        if not job.probe:
            job.probe = _probe_or_400(Path(job.source_path), job.id).to_dict()
        _queue(job, "calibrate", "calibrating", {"ignore_templates": bool(body and body.ignore_templates)})
        s.add(job)
        s.flush()
        return job_out(job)


@router.get("/jobs/{job_id}/calibration")
def get_calibration(job_id: int) -> dict:
    with session_scope() as s:
        _job_or_404(s, job_id)
    return _calibration_out(job_id, _calibration_or_404(job_id))


@router.post("/jobs/{job_id}/calibration/preview")
def preview_calibration(job_id: int, body: PreviewRequest) -> dict:
    """What OCR reads on one sampled frame with the boxes as currently drawn."""
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        adapter = get_adapter(job.sport)
    cal = _calibration_or_404(job_id)
    if not 0 <= body.frame < len(cal.frames):
        raise HTTPException(400, "No such frame")
    path = job_dir(job_id) / "calib" / cal.frames[body.frame]["file"]
    frame = cv2.imread(str(path)) if path.exists() else None
    if frame is None:
        raise HTTPException(404, "That frame was not saved")
    trial = Calibration.from_dict({**cal.to_dict(), "bug": body.bug,
                                   "fields": {k: v for k, v in body.fields.items() if v}})
    reader = make_reader(trial, adapter, frame.shape[1], frame.shape[0], get_engine(4))
    out = read_frame(reader, frame, check_visible=False)
    fw, fh = frame.shape[1], frame.shape[0]
    out["crop"] = crop_to_norm(compute_crop(norm_box(body.bug), fw, fh), fw, fh)
    return out


@router.put("/jobs/{job_id}/calibration")
def save_calibration_edits(job_id: int, body: CalibrationUpdate) -> dict:
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if job.task is not None:
            raise HTTPException(409, "This job is busy.")
        adapter = get_adapter(job.sport)
        probe = Probe.from_dict(job.probe)
    cal = _calibration_or_404(job_id)
    fields = {k: [round(x, 5) for x in norm_box(v)] for k, v in body.fields.items() if v}
    bug = [round(x, 5) for x in norm_box(body.bug)]
    changed = bug != [round(x, 5) for x in cal.bug] or fields != {k: [round(x, 5) for x in v] for k, v in cal.fields.items()}
    fw, fh = probe.display_width, probe.height
    if changed:
        cal.bug, cal.fields = bug, fields
        cal.source = "manual"
        cal.template_id = None  # an edited layout becomes its own template
        cal.template_name = ""
    if body.crop:
        cal.crop = [round(float(v), 6) for v in body.crop]
    elif changed:
        cal.crop = crop_to_norm(compute_crop(tuple(bug), fw, fh), fw, fh)
    if body.broadcaster is not None:
        cal.broadcaster = body.broadcaster.strip()
    if changed:
        cal, ref, mask = recalibrate_manual(cal, probe, job_dir(job_id), adapter, get_engine(4))
    else:
        _, ref, mask = load_calibration(job_id)
    cal.confirmed = body.confirmed or cal.confirmed
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if body.confirmed:
            template = tmpl.save_template(s, cal, adapter, ref, mask, name=(body.template_name or "").strip() or None)
            job.template_id = template.id
        job.calibration = cal.to_dict()
        if job.status in ("draft", "failed"):
            job.status = "ready"
        touch(job)
        s.add(job)
    save_calibration(job_id, cal, ref, mask)
    return _calibration_out(job_id, cal)


@router.get("/jobs/{job_id}/frame")
def job_frame(job_id: int, t: float = Query(0.0, ge=0.0)) -> Response:
    """Any moment of the source as a JPEG (the calibration scrubber beyond the sampled frames)."""
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if not job.probe:
            raise HTTPException(409, "File not probed yet")
        probe = Probe.from_dict(job.probe)
    try:
        frame = extract_frame(probe, t)
    except FrameError as exc:
        raise HTTPException(500, str(exc)) from exc
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "Could not encode the frame")
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# -- analyze / clips ------------------------------------------------------------------------------------


@router.post("/jobs/{job_id}/analyze")
def analyze_job(job_id: int) -> dict:
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        if not (job_dir(job_id) / "calibration.json").exists():
            raise HTTPException(409, "Calibrate this job first.")
        if not Path(job.source_path).is_file():
            raise HTTPException(404, "The source file is no longer where it was.")
        _queue(job, "analyze", "analyzing")
        s.add(job)
        s.flush()
        return job_out(job)


class ClipUpdate(BaseModel):
    enabled: bool | None = None
    src_in: float | None = None
    src_out: float | None = None


@router.get("/jobs/{job_id}/clips")
def list_clips(job_id: int) -> list[dict]:
    with session_scope() as s:
        _job_or_404(s, job_id)
        clips = s.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.order)).all()
        return [clip_out(c) for c in clips]


def _move_edges(clip: Clip, duration: float, src_in: float | None = None, src_out: float | None = None) -> None:
    """Set a clip's in or out point, keeping at least 0.2 s of its first and last segment."""
    clip.segments = move_edges(clip.segments, duration, src_in, src_out)
    clip.src_in, clip.src_out = clip.segments[0][0], clip.segments[-1][1]


@router.patch("/clips/{clip_id}")
def update_clip(clip_id: int, body: ClipUpdate) -> dict:
    with session_scope() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "Clip not found")
        job = s.get(Job, clip.job_id)
        duration = float((job.probe or {}).get("duration") or 1e12)
        if body.enabled is not None:
            clip.enabled = body.enabled
        _move_edges(clip, duration, body.src_in, body.src_out)
        s.add(clip)
        s.flush()
        return clip_out(clip)


class ClipsNudge(BaseModel):
    edge: str = Field(pattern="^(in|out)$")
    delta: float = Field(ge=-5.0, le=5.0)
    only_enabled: bool = True


@router.post("/jobs/{job_id}/clips/nudge")
def nudge_clips(job_id: int, body: ClipsNudge) -> list[dict]:
    """Move the in or out point of every clip in the cut by the same amount."""
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        duration = float((job.probe or {}).get("duration") or 1e12)
        clips = s.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.order)).all()
        for clip in clips:
            if body.only_enabled and not clip.enabled:
                continue
            if body.edge == "in":
                _move_edges(clip, duration, src_in=clip.segments[0][0] + body.delta)
            else:
                _move_edges(clip, duration, src_out=clip.segments[-1][1] + body.delta)
            s.add(clip)
        s.flush()
        return [clip_out(c) for c in clips]


class ClipEdit(BaseModel):
    id: int
    enabled: bool | None = None
    src_in: float | None = None
    src_out: float | None = None


class ClipsBulk(BaseModel):
    updates: list[ClipEdit] = Field(max_length=2000)


@router.post("/jobs/{job_id}/clips/bulk")
def bulk_clips(job_id: int, body: ClipsBulk) -> list[dict]:
    """Several clip edits in one request: what the page's undo, redo and "turn these off"
    send. Every id must belong to the job."""
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        duration = float((job.probe or {}).get("duration") or 1e12)
        clips = s.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.order)).all()
        by_id = {c.id: c for c in clips}
        for edit in body.updates:
            clip = by_id.get(edit.id)
            if clip is None:
                raise HTTPException(404, f"Clip {edit.id} is not part of this job.")
            if edit.enabled is not None:
                clip.enabled = edit.enabled
            _move_edges(clip, duration, edit.src_in, edit.src_out)
            s.add(clip)
        s.flush()
        return [clip_out(c) for c in clips]


@router.post("/jobs/{job_id}/clips/reset")
def reset_clips(job_id: int) -> list[dict]:
    """Every clip back to its detected in and out points. Which clips are on is left alone."""
    with session_scope() as s:
        _job_or_404(s, job_id)
        clips = s.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.order)).all()
        for clip in clips:
            if clip.auto_segments:
                clip.segments = [list(seg) for seg in clip.auto_segments]
                clip.src_in, clip.src_out = clip.segments[0][0], clip.segments[-1][1]
                s.add(clip)
        s.flush()
        return [clip_out(c) for c in clips]


@router.post("/clips/{clip_id}/reset")
def reset_clip(clip_id: int) -> dict:
    with session_scope() as s:
        clip = s.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(404, "Clip not found")
        if clip.auto_segments:
            clip.segments = [list(seg) for seg in clip.auto_segments]
        clip.src_in, clip.src_out = clip.segments[0][0], clip.segments[-1][1]
        clip.enabled = True
        s.add(clip)
        s.flush()
        return clip_out(clip)


# -- export ----------------------------------------------------------------------------------------------


class ExportRequest(BaseModel):
    title: str = ""
    caption: str = ""
    audio_crossfade: bool = True
    options: dict = Field(default_factory=dict)


@router.post("/jobs/{job_id}/export", status_code=201)
def export_job(job_id: int, body: ExportRequest) -> dict:
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        enabled = s.exec(select(Clip).where(Clip.job_id == job_id, Clip.enabled == True)).all()  # noqa: E712
        if not enabled:
            raise HTTPException(400, "No clips are enabled.")
        if not Path(job.source_path).is_file():
            raise HTTPException(404, "The source file is no longer where it was.")
        if job.task is not None:
            raise HTTPException(409, f"This job is busy ({job.task}).")
        export = Export(
            job_id=job_id, title=body.title.strip(), status="pending",
            settings={"caption": body.caption.strip(), "audio_crossfade": body.audio_crossfade, **body.options},
        )
        s.add(export)
        s.flush()
        _queue(job, "export", "exporting", {"export_id": export.id})
        s.add(job)
        s.flush()
        return export_out(export)


@router.get("/jobs/{job_id}/exports")
def list_exports(job_id: int) -> list[dict]:
    with session_scope() as s:
        _job_or_404(s, job_id)
        rows = s.exec(select(Export).where(Export.job_id == job_id).order_by(Export.created_at.desc())).all()
        return [export_out(e) for e in rows]


def _export_or_404(s, export_id: int) -> Export:
    export = s.get(Export, export_id)
    if export is None:
        raise HTTPException(404, "Export not found")
    return export


@router.get("/exports/{export_id}")
def get_export(export_id: int) -> dict:
    with session_scope() as s:
        return export_out(_export_or_404(s, export_id))


@router.get("/exports/{export_id}/file")
def export_file(export_id: int, download: bool = False) -> FileResponse:
    with session_scope() as s:
        export = _export_or_404(s, export_id)
        path = Path(export.path)
    if not export.path or not path.is_file():
        raise HTTPException(404, "The exported file is not there any more.")
    return FileResponse(path, media_type="video/mp4", filename=path.name if download else None)


@router.get("/exports/{export_id}/caption", response_class=PlainTextResponse)
def export_caption(export_id: int) -> str:
    with session_scope() as s:
        return _export_or_404(s, export_id).caption


@router.post("/exports/{export_id}/reveal")
def reveal_export(export_id: int) -> dict:
    """Open the exports folder with the file selected (only when the engine is local)."""
    _local_only()
    with session_scope() as s:
        path = Path(_export_or_404(s, export_id).path)
    if not path.exists():
        raise HTTPException(404, "The exported file is not there any more.")
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError as exc:
        raise HTTPException(500, f"Could not open the folder: {exc}") from exc
    return {"revealed": str(path)}


# -- media -------------------------------------------------------------------------------------------------


@router.get("/media/{job_id}/source")
def media_source(job_id: int) -> FileResponse:
    """The source video (or its preview proxy) with HTTP range support for scrubbing."""
    with session_scope() as s:
        job = _job_or_404(s, job_id)
        source = Path(job.source_path)
        playable = bool((job.probe or {}).get("browser_playable"))
    proxy = _proxy_path(job_id)
    path = proxy if proxy.exists() else source
    if not path.is_file():
        raise HTTPException(404, "The source file is no longer where it was.")
    if path == source and not playable:
        raise HTTPException(415, "This file needs a preview copy, which is made during analysis.")
    media_type = "video/mp4" if path.suffix.lower() in (".mp4", ".m4v", ".mov") else (mimetypes.guess_type(path.name)[0] or "video/mp4")
    return FileResponse(path, media_type=media_type)


@router.get("/media/{job_id}/thumb/{order}")
def media_thumb(job_id: int, order: int) -> FileResponse:
    path = job_dir(job_id) / "thumbs" / f"{order:03d}.jpg"
    if not path.exists():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/media/{job_id}/calib/{name}")
def media_calib(job_id: int, name: str) -> FileResponse:
    if "/" in name or "\\" in name or not name.endswith(".jpg"):
        raise HTTPException(400, "Bad name")
    path = job_dir(job_id) / "calib" / name
    if not path.exists():
        raise HTTPException(404, "No such frame")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


# -- templates -----------------------------------------------------------------------------------------------


def template_out(t: Template) -> dict:
    return {
        "id": t.id, "name": t.name, "sport": t.sport, "broadcaster": t.broadcaster, "bug": t.bug,
        "fields": t.fields, "crop": t.crop, "source": t.source, "use_count": t.use_count,
        "image": f"/api/templates/{t.id}/image" if t.reference_image else None,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "last_used_at": t.last_used_at.isoformat() if t.last_used_at else None,
    }


@router.get("/templates")
def list_templates() -> list[dict]:
    with session_scope() as s:
        return [template_out(t) for t in s.exec(select(Template).order_by(Template.created_at.desc())).all()]


@router.get("/templates/{template_id}/image")
def template_image(template_id: int) -> FileResponse:
    with session_scope() as s:
        t = s.get(Template, template_id)
        path = Path(t.reference_image) if t and t.reference_image else None
    if path is None or not path.exists():
        raise HTTPException(404, "No image")
    return FileResponse(path, media_type="image/png")


@router.delete("/templates/{template_id}")
def delete_template(template_id: int) -> dict:
    with session_scope() as s:
        t = s.get(Template, template_id)
        if t is None:
            raise HTTPException(404, "Template not found")
        for job in s.exec(select(Job).where(Job.template_id == template_id)).all():
            job.template_id = None
            s.add(job)
        s.flush()
        tmpl.delete_template(s, t)
    return {"deleted": template_id}
