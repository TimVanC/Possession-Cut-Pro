#!/usr/bin/env python
"""Benchmark the analysis stage on a long file (PRD target: 2.5 h of 1080p in under 15 min).

Renders the short synthetic game once at the requested resolution, loops it with a
stream copy to the requested length (so no time is spent encoding hours of video),
then runs the real stages on it: probe, calibration, bug sampling + OCR, timeline.

    python tools/benchmark_analysis.py --minutes 150 --height 1080 --workdir D:/tmp/bench

The looped game restarts its score and clock every few minutes, so the cut itself is
meaningless here. What matters is decode + OCR throughput, which is what dominates a
real analysis.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from possession_cut.config import ffmpeg_bin, get_settings  # noqa: E402
from possession_cut.pipeline.calibration import calibrate  # noqa: E402
from possession_cut.pipeline.ocr import get_engine  # noqa: E402
from possession_cut.pipeline.probe import probe_file  # noqa: E402
from possession_cut.pipeline.sampler import sample_bug  # noqa: E402
from possession_cut.pipeline.timeline import build_timeline  # noqa: E402
from possession_cut.sports import get_adapter  # noqa: E402
from possession_cut.synth.render import render_video  # noqa: E402
from possession_cut.synth.script import coverage_game  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=150.0)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0, help="0 = the app default")
    ap.add_argument("--workdir", default=str(Path(tempfile.gettempdir()) / "possession-cut-bench"))
    ap.add_argument("--keep", action="store_true", help="keep the generated files")
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    width = int(round(args.height * 16 / 9 / 2) * 2)
    base = work / f"base_{args.height}p.mp4"
    if not base.exists():
        print(f"Rendering the base game at {width}x{args.height}...")
        t0 = time.time()
        render_video(coverage_game(), base, width=width, height=args.height, fps=args.fps, crf=23)
        print(f"  rendered in {time.time() - t0:.0f}s")
    base_probe = probe_file(base)
    loops = max(0, int(round(args.minutes * 60 / base_probe.duration)) - 1)
    long_path = work / f"long_{args.height}p_{int(args.minutes)}min.mp4"
    if not long_path.exists():
        print(f"Looping it {loops + 1}x with a stream copy...")
        subprocess.run(
            [ffmpeg_bin(), "-y", "-v", "error", "-stream_loop", str(loops), "-i", str(base),
             "-c", "copy", "-movflags", "+faststart", str(long_path)],
            check=True,
        )

    adapter = get_adapter("nba")
    workers = args.workers or get_settings().workers
    result: dict = {"file": str(long_path), "workers": workers}

    t0 = time.time()
    probe = probe_file(long_path)
    result["probe_s"] = round(time.time() - t0, 2)
    result["duration_min"] = round(probe.duration / 60, 1)
    result["resolution"] = f"{probe.width}x{probe.height}@{probe.fps:g}"
    result["size_gb"] = round(probe.size_bytes / 1e9, 2)
    print(f"File: {result['duration_min']} min, {result['resolution']}, {result['size_gb']} GB")

    t0 = time.time()
    cal, ref, mask = calibrate(probe, work / "job", adapter, get_engine(4))
    result["calibrate_s"] = round(time.time() - t0, 1)
    print(f"Calibration: {result['calibrate_s']}s (source {cal.source}, confidence {cal.confidence})")

    last = [0.0]

    def progress(frac: float, message: str) -> None:
        if time.time() - last[0] > 5:
            last[0] = time.time()
            print(f"  {frac * 100:5.1f}%  {message}")

    t0 = time.time()
    raw = sample_bug(probe, cal, adapter, ref, mask, fps=get_settings().ocr_sample_fps, workers=workers, progress=progress)
    result["sample_s"] = round(time.time() - t0, 1)
    result["sampler"] = raw.stats

    t0 = time.time()
    build_timeline(raw, adapter)
    result["timeline_s"] = round(time.time() - t0, 1)

    total = result["probe_s"] + result["calibrate_s"] + result["sample_s"] + result["timeline_s"]
    result["total_s"] = round(total, 1)
    result["total_min"] = round(total / 60, 2)
    result["projected_min_for_150"] = round(total / 60 * 150 / result["duration_min"], 2)
    print(json.dumps(result, indent=1))
    print(f"\nAnalysis of {result['duration_min']} min took {result['total_min']} min "
          f"({probe.duration / total:.0f}x real time). Projected for 2.5 h: {result['projected_min_for_150']} min.")
    (work / "benchmark.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if not args.keep:
        long_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
