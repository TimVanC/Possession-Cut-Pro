"""Does this installation work?

Renders a short scripted game, runs the whole pipeline on it (find the score bug, read
it, cut the clips, export the vertical video) and checks each result against the script.
No network, no Claude, no database; everything happens in a temporary folder.

    python -m possession_cut.selfcheck

Exits 0 when every check passed. A server can run it before each deploy (Railway's
pre-deploy command) so a build whose ffmpeg, OCR models or fonts are broken never goes live.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from .config import ffmpeg_bin, ffprobe_bin
from .pipeline.analyze import JobSpec, analyze
from .pipeline.calibration import calibrate
from .pipeline.export import plan_export, render, render_overlay
from .pipeline.ocr import get_engine
from .pipeline.probe import probe_file
from .sports import get_adapter
from .synth.render import render_video
from .synth.script import AWAY, HOME, GameScript, ScriptBuilder


def check_game() -> GameScript:
    """About two minutes of a scripted third quarter: three home baskets and a trip to the line."""
    b = ScriptBuilder(seed=21, start_scores=(61, 58))
    b.period_start(3, 300.0, ball=AWAY)
    b.possession(AWAY, 7.0, "miss_def")
    b.possession(HOME, 8.0, "make2", anim=True)
    b.possession(AWAY, 8.0, "make3", anim=False)
    b.possession(HOME, 9.0, "make3", anim=False)
    b.possession(AWAY, 6.0, "steal")
    b.possession(HOME, 6.0, "shooting_foul", fts=[True, True], walk=12.0, gap=8.0)
    b.possession(AWAY, 7.0, "miss_def")
    b.possession(HOME, 7.0, "make2", anim=True)
    b.possession(AWAY, 6.0, "miss_def")
    return b.build()


class Report:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, ok: bool, what: str) -> bool:
        print(f"  {'ok  ' if ok else 'FAIL'}  {what}", flush=True)
        if not ok:
            self.failed.append(what)
        return ok


def run(keep: bool = False) -> int:
    report = Report()
    started = time.time()
    work = Path(tempfile.mkdtemp(prefix="possession-cut-selfcheck-"))
    try:
        print("Possession Cut self-check", flush=True)
        try:
            print(f"  ffmpeg:  {ffmpeg_bin()}\n  ffprobe: {ffprobe_bin()}", flush=True)
        except FileNotFoundError as exc:
            report.check(False, str(exc))
            return 1

        script = check_game()
        video = work / "game.mp4"
        t0 = time.time()
        render_video(script, video, width=1280, height=720, fps=30)
        probe = probe_file(video)
        report.check(abs(probe.duration - script.duration) < 0.5 and probe.height == 720,
                     f"rendered a {probe.duration:.0f} s test broadcast ({time.time() - t0:.0f} s)")

        adapter = get_adapter("nba")
        job = work / "job"
        job.mkdir()
        t0 = time.time()
        cal, ref, mask = calibrate(probe, job, adapter, get_engine(4), claude=None, templates=[])
        teams = (cal.teams or {}).get("away", "?"), (cal.teams or {}).get("home", "?")
        report.check(cal.confidence >= 0.8 and {"clock", "away_score", "home_score", "period"} <= set(cal.fields),
                     f"found the score bug ({cal.source}, confidence {cal.confidence:.2f}, {teams[0]} at {teams[1]}; {time.time() - t0:.0f} s)")

        t0 = time.time()
        result = analyze(probe, cal, ref, mask, adapter, JobSpec(sport="nba", team="home"), job, fps=2.0, workers=2, claude=None)
        truth = script.cutlist(HOME)
        clips = result.clips
        report.check([c.score_after for c in clips] == [t["score_after"] for t in truth],
                     f"found every scoring play: {len(clips)} clips for {len(truth)} scripted ({time.time() - t0:.0f} s)")
        if len(clips) == len(truth):
            starts = [abs(c.src_in - t["src_in"]) for c, t in zip(clips, truth, strict=True)]
            ends = [abs(c.src_out - t["src_out"]) for c, t in zip(clips, truth, strict=True)]
            report.check(max(starts) <= 2.5 and max(ends) <= 1.5,
                         f"clip boundaries match the script (starts within {max(starts):.1f} s, ends within {max(ends):.1f} s)")
            report.check([c.kind for c in clips] == [t["kind"] for t in truth], "free throws and baskets told apart")
        pbp = result.summary.get("pbp", {})
        report.check(pbp.get("available") and pbp.get("unmatched_detected") == 0, "plays matched to the play-by-play")

        if clips:
            t0 = time.time()
            segments = [tuple(seg) for c in clips for seg in c.segments]
            plan = plan_export(probe, segments, cal.crop, crossfade=True, crossfade_ms=80)
            overlay = render_overlay("Self-check", "every home bucket", plan.placement, job / "overlay.png")
            out = work / "cut.mp4"
            render(plan, probe, out, overlay, job / "export_work")
            made = probe_file(out)
            report.check((made.width, made.height) == (1080, 1920), f"exported a 1080x1920 video ({time.time() - t0:.0f} s)")
            report.check(made.video_codec == "h264" and made.audio_codec == "aac", "video is H.264 with AAC audio")
            report.check(abs(made.duration - plan.duration) <= 0.25,
                         f"export length matches the cut ({made.duration:.1f} s of {plan.duration:.1f} s)")
    except Exception as exc:  # the point is to report, not to crash
        import traceback

        traceback.print_exc()
        report.check(False, f"{type(exc).__name__}: {exc}")
    finally:
        if keep:
            print(f"  files kept in {work}", flush=True)
        else:
            shutil.rmtree(work, ignore_errors=True)

    took = time.time() - started
    if report.failed:
        print(f"\nSELF-CHECK FAILED ({len(report.failed)} problem(s), {took:.0f} s)", flush=True)
        return 1
    print(f"\nSelf-check passed in {took:.0f} s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run(keep="--keep" in sys.argv))
