#!/usr/bin/env python
"""Check an exported cut against the PRD's output rules.

    python tools/verify_export.py exports/My_cut.mp4
    python tools/verify_export.py exports/My_cut.mp4 --truth inbox/synthetic_game.mp4.truth.json

Always checked (any export): container and codec settings with ffprobe, canvas size, that
the picture is centred with black bars above and below, the crop geometry recorded in
the cut list (full height, bug about 78% of the crop width, centred on the bug), and
that audio and video are the same length.

With --truth (a synthetic game's ground truth) every exported frame's barcode is read
back, proving the export holds exactly the frames in the cut list, in order, and that
none of them comes from a replay or a commercial.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from possession_cut.config import ffmpeg_bin, ffprobe_bin  # noqa: E402
from possession_cut.synth.render import BARCODE_CELLS, decode_barcode  # noqa: E402

results: list[tuple[bool, str]] = []


def check(ok: bool, text: str) -> bool:
    results.append((bool(ok), text))
    print(("  PASS  " if ok else "  FAIL  ") + text)
    return bool(ok)



def top_level_boxes(path: Path, limit: int = 12) -> list[str]:
    """Names of the MP4's top-level boxes, in file order (a long game's index alone can be
    megabytes, so this walks the box sizes instead of searching the first bytes)."""
    names: list[str] = []
    with path.open("rb") as f:
        pos = 0
        while len(names) < limit:
            head = f.read(8)
            if len(head) < 8:
                break
            size, kind = struct.unpack(">I4s", head)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            names.append(kind.decode("latin1"))
            if size == 0:
                break
            pos += size
            f.seek(pos)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--truth", help="ground truth JSON of the synthetic source, for the frame-by-frame check")
    ap.add_argument("--calibration", help="calibration.json of the job, for the bug-in-crop check on real footage")
    args = ap.parse_args()

    video = Path(args.video)
    cutlist_path = video.with_name(video.stem + ".cutlist.json")
    cutlist = json.loads(cutlist_path.read_text(encoding="utf-8")) if cutlist_path.exists() else None

    info = json.loads(subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, check=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    fps = Fraction(v["avg_frame_rate"])
    print(f"\n{video.name}: {v['width']}x{v['height']} {v['codec_name']} {v.get('profile')} @ {float(fps):.3f} fps, "
          f"{float(info['format']['duration']):.2f}s, {int(info['format']['size']) / 1e6:.1f} MB\n")

    print("Container and codecs (ffprobe)")
    check((v["width"], v["height"]) == (1080, 1920), "canvas is 1080x1920 (9:16)")
    check(v["codec_name"] == "h264" and v.get("profile") == "High", f"video is H.264 High ({v['codec_name']} {v.get('profile')})")
    check(v["pix_fmt"] == "yuv420p", "pixel format yuv420p")
    check(fps <= 60, f"frame rate {float(fps):.3f} is the source rate, at most 60")
    check(a is not None and a["codec_name"] == "aac", "audio is AAC")
    if a:
        check(150_000 <= int(a.get("bit_rate", 0)) <= 230_000, f"audio bitrate about 192 kbps ({int(a.get('bit_rate', 0)) // 1000} kbps)")
        check(abs(float(a["duration"]) - float(v["duration"])) <= 0.08,
              f"audio and video are the same length ({float(a['duration']):.3f}s vs {float(v['duration']):.3f}s)")
    order = top_level_boxes(video)
    check("moov" in order and "mdat" in order and order.index("moov") < order.index("mdat"),
          f"+faststart: the index comes before the media ({' '.join(order)})")

    print("\nCanvas")
    mid = float(info["format"]["duration"]) / 2
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", f"{mid:.2f}", "-i", str(video), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(1920, 1080)
    rows = frame.mean(axis=1)
    lit = np.flatnonzero(rows > 24)
    # the picture is the longest run of bright rows; title text above it is short
    runs, start = [], lit[0]
    for prev, cur in zip(lit, lit[1:], strict=False):
        if cur - prev > 12:
            runs.append((start, prev))
            start = cur
    runs.append((start, lit[-1]))
    top, bottom = max(runs, key=lambda r: r[1] - r[0])
    pic_h = bottom - top + 1
    check(abs(top - (1920 - pic_h) / 2) <= 6, f"picture is centred vertically (rows {top}..{bottom}, {pic_h} px tall)")
    check(frame[:30].max() < 30 and frame[-30:].max() < 30, "black canvas at the very top and bottom")
    check(frame[top + 10 : bottom - 10, :8].mean() > 20 and frame[top + 10 : bottom - 10, -8:].mean() > 20, "picture spans the full 1080 width")
    print(f"        crop aspect on canvas: {1080 / pic_h:.3f}:1 (the reference is about 1.18:1)")

    if cutlist:
        print("\nCut list (" + cutlist_path.name + ")")
        render = cutlist["render"]
        cx, cy, cw, ch = render["crop"]
        total = sum(c["out_end"] - c["out_start"] for c in cutlist["clips"])
        check(abs(total - float(v["duration"])) <= 0.1, f"video length equals the cut list total ({total:.2f}s)")
        check(60 <= render["audio_crossfade_ms"] <= 100 or render["audio_crossfade_ms"] == 0,
              f"audio crossfade {render['audio_crossfade_ms']} ms (60 to 100 ms)")
        on_canvas = render["video_on_canvas"]
        check(abs(on_canvas["h"] - pic_h) <= 4 and abs(on_canvas["y"] - top) <= 4, "picture sits where the cut list says")
        check(cy == 0, "crop starts at the top of the source (full source height)")
        bug = None
        src_w = src_h = None
        if args.truth:
            truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
            bug, src_w, src_h = truth["layout"]["bug_px"], truth["width"], truth["height"]
        elif args.calibration:
            cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
            src_w, src_h = cal["frame_width"], cal["frame_height"]
            bug = [cal["bug"][0] * src_w, cal["bug"][1] * src_h, cal["bug"][2] * src_w, cal["bug"][3] * src_h]
        if bug:
            check(ch >= src_h - 1, f"crop height is the full source height ({ch} of {src_h})")
            fill = (bug[2] - bug[0]) / cw
            clamped = cx == 0 or cx + cw >= src_w - 1
            check(abs(fill - 0.78) <= 0.015 or cw >= src_w - 2, f"score bug fills {fill * 100:.1f}% of the crop width (rule: 78%)")
            off = abs((cx + cw / 2) - (bug[0] + bug[2]) / 2)
            check(off <= 2 or clamped, f"crop is centred on the bug (off by {off:.1f} px{', clamped to the frame edge' if clamped and off > 2 else ''})")
            check(cx <= bug[0] and bug[2] <= cx + cw and bug[3] <= cy + ch, "the whole bug is inside the crop")

    if args.truth and cutlist:
        print("\nFrame by frame (barcodes from the synthetic source)")
        truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
        bc = truth["layout"]["barcode"]
        render = cutlist["render"]
        cx, cy, cw, ch = render["crop"]
        sw, sh, pad_y = render["video_on_canvas"]["w"], render["video_on_canvas"]["h"], render["video_on_canvas"]["y"]
        sx, sy = sw / cw, sh / ch
        c = bc["cell"]
        x0, y0 = (bc["x0"] - cx) * sx, pad_y + (bc["y0"] - cy) * sy
        w, h = BARCODE_CELLS * c * sx, c * sy
        ex0, ey0 = int(x0) - int(x0) % 2, int(y0) - int(y0) % 2
        ew, eh = int(w) + 4 + int(w) % 2, int(h) + 4 + int(h) % 2
        raw = subprocess.run(
            [ffmpeg_bin(), "-v", "error", "-i", str(video), "-vf", f"crop={ew}:{eh}:{ex0}:{ey0},format=gray", "-f", "rawvideo", "-"],
            capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, eh, ew)
        got = []
        for f in frames:
            cells = [float(f[int(y0 - ey0 + 0.5 * c * sy) - 2 : int(y0 - ey0 + 0.5 * c * sy) + 3,
                             int(x0 - ex0 + (i + 0.5) * c * sx) - 2 : int(x0 - ex0 + (i + 0.5) * c * sx) + 3].mean())
                     for i in range(BARCODE_CELLS)]
            got.append(decode_barcode(cells))
        src_fps = Fraction(render["fps"])
        want = []
        for seg in render["segments"]:
            first = int(round(seg["src_in"] * src_fps))
            want += [first + k for k in range(seg["frames"])]
        check(None not in got, f"all {len(got)} exported frames carry a readable barcode (crop and scale are exact)")
        check(got == want, "the export holds exactly the frames of the cut list, in order")
        not_live = truth["intervals"]["not_live"]
        bad = [n for n in got if n is not None and any(a <= n / truth["fps"] < b for a, b in not_live)]
        check(not bad, f"no frame comes from a replay or commercial ({len(bad)} offending frames)")
        # every enabled clip must be one the script actually scored
        scored = {(c["score_before"], c["score_after"]) for c in truth["cutlists"]["home"] + truth["cutlists"]["away"]}
        clips_ok = all((c["score_before"], c["score_after"]) in scored for c in cutlist["clips"])
        check(clips_ok, f"all {len(cutlist['clips'])} clips are scripted scores")

    failed = [t for ok, t in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed." + ("" if not failed else "  FAILED: " + "; ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
