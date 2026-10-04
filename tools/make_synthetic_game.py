#!/usr/bin/env python
"""Render a synthetic broadcast with a scripted game and its ground-truth cut list.

    python tools/make_synthetic_game.py --out inbox/synthetic_game.mp4
    python tools/make_synthetic_game.py --preset coverage --out data/coverage.mp4
    python tools/make_synthetic_game.py --height 1080 --periods 4 --period-seconds 720 --out long.mp4

Writes three files: the video, <video>.truth.json (bug layout, live/not-live intervals,
every scripted score, and the target cut list for both teams) and <video>.pbp.json
(scripted play-by-play in the shape the sport adapters return; the app picks it up
automatically when it sits next to the video).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from possession_cut.synth.render import render_video  # noqa: E402
from possession_cut.synth.script import coverage_game, random_game  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="inbox/synthetic_game.mp4")
    ap.add_argument("--preset", choices=["full", "coverage"], default="full",
                    help="full: randomized 4-period game (15-20 min). coverage: short fixed game used by the tests.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--periods", type=int, default=4)
    ap.add_argument("--period-seconds", type=float, default=180.0)
    ap.add_argument("--start-scores", default="78,71", help="away,home score at the start")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bug-offset", type=float, default=0.02, help="bug centre offset as a fraction of width")
    ap.add_argument("--crf", type=int, default=21)
    args = ap.parse_args()

    width = args.width or int(round(args.height * 16 / 9 / 2) * 2)
    if args.preset == "coverage":
        script = coverage_game(args.seed)
    else:
        away, home = (int(x) for x in args.start_scores.split(","))
        script = random_game(
            args.seed,
            periods=tuple(range(1, args.periods + 1)),
            period_seconds=args.period_seconds,
            start_scores=(away, home),
        )

    t0 = time.time()

    def progress(n: int, total: int) -> None:
        elapsed = time.time() - t0
        rate = n / elapsed if elapsed > 0 else 0
        print(f"\r  frame {n}/{total}  {rate:5.0f} fps", end="", flush=True)

    print(f"Rendering {script.duration / 60:.1f} min at {width}x{args.height}@{args.fps} -> {args.out}")
    truth = render_video(script, args.out, width=width, height=args.height, fps=args.fps,
                         bug_offset=args.bug_offset, crf=args.crf, progress=progress)
    print(f"\nDone in {time.time() - t0:.0f}s. {len(truth['score_events'])} scripted scores, "
          f"{len(truth['cutlists']['home'])} home clips, {len(truth['cutlists']['away'])} away clips.")
    print(f"Ground truth: {args.out}.truth.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
