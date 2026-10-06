"""What the worker does with a claimed job: calibrate, analyze, or export.

All heavy work (decoding, OCR, rendering, Claude, league APIs) happens here, in the
worker process. Progress is written to the job row so the API can stream it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
from sqlmodel import select

from ..ai.caption import CutFacts, make_caption
from ..ai.claude import ClaudeClient
from ..config import get_settings
from ..db import Clip, Export, Job, session_scope, touch
from ..edits import edit_key, move_edges
from ..pipeline import templates as tmpl
from ..pipeline.analyze import JobSpec, analyze
from ..pipeline.calibration import Calibration, calibrate
from ..pipeline.export import make_proxy, plan_export, render, render_overlay, write_cutlist
from ..pipeline.ocr import get_engine
from ..pipeline.probe import Probe, probe_file
from ..preview import part_path, proxy_path
from ..sports import get_adapter
from .janitor import make_room

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


class JobContext:
    """Throttled progress writes and cancellation checks for one running task."""

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        self._last_write = 0.0
        self._last_check = 0.0
        self._cancelled = False

    def progress(self, fraction: float, stage: str, message: str = "", force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < 0.4:
            return
        self._last_write = now
        with session_scope() as s:
            job = s.get(Job, self.job_id)
            if job is None:
                return
            job.progress = max(0.0, min(1.0, float(fraction)))
            job.stage = stage
            job.message = message
            touch(job)
            s.add(job)

    def should_stop(self) -> bool:
        now = time.time()
        if self._cancelled or now - self._last_check < 1.5:
            return self._cancelled
        self._last_check = now
        with session_scope() as s:
            job = s.get(Job, self.job_id)
            self._cancelled = job is None or bool((job.task_payload or {}).get("cancel"))
        return self._cancelled


def job_dir(job_id: int) -> Path:
    return get_settings().job_dir(job_id)


def load_probe(job: Job) -> Probe:
    if job.probe:
        return Probe.from_dict(job.probe)
    return probe_file(job.source_path, job_dir(job.id) / "probe.json")


def load_calibration(job_id: int) -> tuple[Calibration, object, object]:
    d = job_dir(job_id)
    cal = Calibration.load(d / "calibration.json")
    ref = cv2.imread(str(d / "ref.png")) if (d / "ref.png").exists() else None
    mask = cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE) if (d / "mask.png").exists() else None
    return cal, ref, mask


def save_calibration(job_id: int, cal: Calibration, ref, mask) -> None:
    d = job_dir(job_id)
    cal.save(d / "calibration.json")
    for name, img in (("ref.png", ref), ("mask.png", mask)):
        path = d / name
        if img is not None:
            cv2.imwrite(str(path), img)
        else:
            path.unlink(missing_ok=True)


def _claude_for(job: Job) -> ClaudeClient:
    return ClaudeClient(spent_usd=job.claude_spent_usd or 0.0)


# -- calibrate ------------------------------------------------------------------------


def run_calibrate(job_id: int) -> None:
    ctx = JobContext(job_id)
    with session_scope() as s:
        job = s.get(Job, job_id)
        payload = dict(job.task_payload or {})
        probe = load_probe(job)
        job.probe = probe.to_dict()
        adapter = get_adapter(job.sport)
        templates = [] if payload.get("ignore_templates") else tmpl.templates_for(s, job.sport)
        for t in templates:
            s.expunge(t)
        claude = _claude_for(job)
        game = dict(job.game or {})
        s.add(job)
    ctx.progress(0.02, "calibrating", "Sampling frames", force=True)
    expected = {"away": game.get("away", ""), "home": game.get("home", "")} if game else None
    cal, ref, mask = calibrate(
        probe, job_dir(job_id), adapter, get_engine(4), claude=claude, templates=templates,
        expected_teams=expected, progress=lambda f, m: ctx.progress(f, "calibrating", m),
        should_stop=ctx.should_stop,
    )
    if ctx.should_stop():
        raise Cancelled()
    save_calibration(job_id, cal, ref, mask)
    with session_scope() as s:
        job = s.get(Job, job_id)
        job.calibration = cal.to_dict()
        job.template_id = cal.template_id
        job.claude_spent_usd = claude.usage.cost_usd
        job.status = "ready"
        job.stage = "calibrated"
        job.progress = 1.0
        job.message = (
            f"Matched template {cal.template_name}" if cal.source == "template"
            else "Score bug found" if cal.confidence >= 0.5 else "Needs a manual check"
        )
        job.error = None
        touch(job)
        s.add(job)


# -- analyze ----------------------------------------------------------------------------


def run_analyze(job_id: int) -> None:
    ctx = JobContext(job_id)
    settings = get_settings()
    with session_scope() as s:
        job = s.get(Job, job_id)
        probe = load_probe(job)
        adapter = get_adapter(job.sport)
        spec = JobSpec(
            sport=job.sport, team=job.team, game_id=job.game_id, game=dict(job.game or {}),
            start_spec=dict(job.start_spec or {"mode": "start"}), end_spec=dict(job.end_spec or {"mode": "end"}),
            options=dict(job.options or {}),
        )
        claude = _claude_for(job)
    cal, ref, mask = load_calibration(job_id)
    d = job_dir(job_id)

    result = analyze(
        probe, cal, ref, mask, adapter, spec, d,
        fps=settings.ocr_sample_fps, workers=settings.workers, claude=claude,
        progress=lambda f, stage, m: ctx.progress(0.92 * f, stage, m),
        should_stop=ctx.should_stop,
    )
    if ctx.should_stop():
        raise Cancelled()

    # a source the browser cannot play needs a preview copy for the review screen (the
    # worker's preview thread usually has one under way already; then leave it to that)
    proxy = proxy_path(job_id)
    if not probe.browser_playable and not proxy.exists() and not part_path(job_id).exists():
        ctx.progress(0.93, "proxy", "Preparing a preview copy", force=True)
        try:
            make_proxy(probe, proxy, progress=lambda f, m: ctx.progress(0.93 + 0.06 * f, "proxy", m),
                       should_stop=ctx.should_stop)
        except Exception as exc:
            log.exception("preview proxy failed")
            proxy.unlink(missing_ok=True)
            result.summary["warnings"].append(f"Could not make a preview copy for the browser ({exc}).")

    with session_scope() as s:
        job = s.get(Job, job_id)
        # what the owner changed last time carries over to the same plays: a clip turned
        # off stays off, a moved edge moves the same amount from the new detected edge
        prior: dict[tuple, tuple[bool, float, float]] = {}
        for old in s.exec(select(Clip).where(Clip.job_id == job_id)).all():
            d_in, d_out = old.src_in - old.auto_in, old.src_out - old.auto_out
            if not old.enabled or abs(d_in) > 1e-6 or abs(d_out) > 1e-6:
                prior[edit_key(old.pbp_event_id, old.team, old.period, old.clock, old.kind, old.score_after)] = (old.enabled, d_in, d_out)
            s.delete(old)
        s.flush()
        carried = 0
        for k, c in enumerate(result.clips):
            thumb = d / "thumbs" / f"{k:03d}.jpg"
            segments = [list(seg) for seg in c.segments]
            enabled = True
            kept = prior.pop(edit_key(c.pbp_event_id, c.team, c.period, c.clock, c.kind, c.score_after), None)
            if kept is not None:
                enabled, d_in, d_out = kept
                segments = move_edges(
                    segments, probe.duration,
                    c.src_in + d_in if abs(d_in) > 1e-6 else None, c.src_out + d_out if abs(d_out) > 1e-6 else None,
                )
                carried += 1
            s.add(
                Clip(
                    job_id=job_id, order=k, src_in=segments[0][0], src_out=segments[-1][1],
                    segments=segments,
                    auto_segments=[list(seg) for seg in c.segments], auto_in=c.src_in, auto_out=c.src_out,
                    team=c.team, period=c.period, clock=c.clock, score_before=c.score_before,
                    score_after=c.score_after, score_away=c.score_away, score_home=c.score_home,
                    points=c.points, kind=c.kind, scorer=c.scorer, description=c.description,
                    confidence=c.confidence, enabled=enabled, pbp_event_id=c.pbp_event_id,
                    warnings=list(c.warnings),
                    events=[{"t": ch.t, "points": ch.points, "team": ch.team, "clock": ch.clock, "period": ch.period,
                             "confidence": round(float(getattr(ch, "confidence", 1.0)), 3)}
                            for ch in c.changes],
                    thumbnail=str(thumb) if thumb.exists() else "",
                )
            )
        if carried or prior:
            note = f"Kept your edits on {carried} clip{'s' if carried != 1 else ''} from the previous analysis"
            if prior:
                note += f"; {len(prior)} edited clip{'s' if len(prior) != 1 else ''} from that run {'are' if len(prior) != 1 else 'is'} no longer in the cut"
            result.summary.setdefault("warnings", []).append(note + ".")
        job.summary = json.loads(json.dumps(result.summary, default=str))
        job.claude_spent_usd = claude.usage.cost_usd
        job.status = "review"
        job.stage = "analyzed"
        job.progress = 1.0
        job.message = f"{len(result.clips)} clips ready for review"
        job.error = None
        touch(job)
        s.add(job)


# -- export -----------------------------------------------------------------------------


def _safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 _.-]+", "", text).strip().replace(" ", "_")
    return name[:70] or "possession_cut"


def run_export(job_id: int) -> None:
    ctx = JobContext(job_id)
    settings = get_settings()
    with session_scope() as s:
        job = s.get(Job, job_id)
        export_id = (job.task_payload or {}).get("export_id")
        export = s.get(Export, export_id) if export_id else None
        if export is None:
            raise RuntimeError("Export record not found")
        probe = load_probe(job)
        adapter = get_adapter(job.sport)
        clips = s.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.order)).all()
        enabled = [c for c in clips if c.enabled]
        opts = dict(export.settings or {})
        title, caption_bar = export.title, opts.get("caption", "")
        summary = dict(job.summary or {})
        source_name = job.source_name
        game = dict(job.game or {})
        claude = _claude_for(job)
        export.status = "rendering"
        s.add(export)
        clip_rows = [
            {
                "order": c.order, "team": c.team, "kind": c.kind, "points": c.points, "period": c.period,
                "game_clock": c.clock, "score_before": c.score_before, "score_after": c.score_after,
                "score_away": c.score_away, "score_home": c.score_home, "scorer": c.scorer,
                "description": c.description, "confidence": c.confidence, "warnings": c.warnings,
                "src_in": c.src_in, "src_out": c.src_out, "segments": c.segments, "pbp_event_id": c.pbp_event_id,
            }
            for c in enabled
        ]
    cal, _, _ = load_calibration(job_id)
    for row in clip_rows:
        extra = adapter.export_extend(row["kind"], opts)
        if extra > 0:
            row["segments"] = [list(seg) for seg in row["segments"]]
            row["segments"][-1][1] = min(probe.duration, row["segments"][-1][1] + extra)
            row["src_out"] = row["segments"][-1][1]
    segments = [tuple(seg) for row in clip_rows for seg in row["segments"]]
    plan = plan_export(
        probe, segments, cal.crop,
        crossfade=bool(opts.get("audio_crossfade", True)),
        crossfade_ms=float(opts.get("crossfade_ms", 80)),
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{_safe_name(title or summary.get('suggested_title') or Path(source_name).stem)}_{stamp}"
    out_path = settings.exports_path / f"{base}.mp4"
    work = settings.scratch_path(job_id) / "export_work"
    # about 1.2 MB per second of 1080x1920 at this quality, plus slack; old exports go first
    make_room(int(plan.duration * 1.6e6) + 300_000_000)
    overlay = render_overlay(title, caption_bar, plan.placement, work / "overlay.png")

    ctx.progress(0.01, "rendering", "Rendering", force=True)
    try:
        render(plan, probe, out_path, overlay, work,
               progress=lambda f, m: ctx.progress(0.95 * f, "rendering", m), should_stop=ctx.should_stop)
    except BaseException:
        out_path.unlink(missing_ok=True)  # a half-written file is never a valid video
        raise

    ctx.progress(0.96, "caption", "Writing the caption", force=True)
    facts = CutFacts(**{k: v for k, v in (summary.get("facts") or {}).items() if k in CutFacts.__dataclass_fields__})
    facts.clips = len(clip_rows)
    facts.duration_seconds = round(plan.duration, 1)
    facts.plays = [r["description"] for r in clip_rows if r["description"]][-4:]
    caption_text, caption_source = make_caption(facts, claude)
    caption_path = settings.exports_path / f"{base}.caption.txt"
    caption_path.write_text(caption_text + "\n", encoding="utf-8")

    cutlist_path = settings.exports_path / f"{base}.cutlist.json"
    write_cutlist(
        cutlist_path,
        {
            "title": title, "caption_bar": caption_bar, "source": source_name, "sport": adapter.key,
            "game": game, "team": summary.get("team_abbr"), "start": summary.get("start_label"),
            "end": summary.get("end_label"), "exported_at": datetime.now().isoformat(timespec="seconds"),
            "video": out_path.name, "caption_source": caption_source,
        },
        clip_rows, plan,
    )

    with session_scope() as s:
        export = s.get(Export, export_id)
        export.status = "done"
        export.path = str(out_path)
        export.caption = caption_text
        export.caption_path = str(caption_path)
        export.cutlist_path = str(cutlist_path)
        export.duration = plan.duration
        export.size_bytes = out_path.stat().st_size
        export.settings = {**opts, "caption_source": caption_source, "render": plan.to_dict()}
        s.add(export)
        job = s.get(Job, job_id)
        job.claude_spent_usd = claude.usage.cost_usd
        job.status = "done"
        job.stage = "exported"
        job.progress = 1.0
        job.message = f"Exported {plan.duration:.0f}s to {out_path.name}"
        job.error = None
        touch(job)
        s.add(job)


TASKS = {"calibrate": run_calibrate, "analyze": run_analyze, "export": run_export}
# where a job goes back to if its task fails or is cancelled
PREVIOUS_STATUS = {"calibrate": "draft", "analyze": "ready", "export": "review"}
