"""Export: planning, the filter graph, and a real render checked frame by frame.

Every frame of the synthetic source carries its own index as a barcode. Reading those
back out of the exported file proves which source frames are in the cut, in what order,
and that the crop and canvas placement are exactly where the PRD puts them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from possession_cut.config import ffmpeg_bin, ffprobe_bin
from possession_cut.pipeline import export as ex
from possession_cut.pipeline.clips import build_clips
from possession_cut.pipeline.export import (
    build_batch,
    make_proxy,
    plan_export,
    render,
    render_overlay,
    thumbnail,
    write_cutlist,
)
from possession_cut.pipeline.geometry import OUTPUT_H, OUTPUT_W
from possession_cut.pipeline.probe import Probe, parse_probe, probe_file
from possession_cut.sports import get_adapter
from possession_cut.synth.render import BARCODE_CELLS, decode_barcode

from .conftest import BACKEND, _cache_dir, _calibration_version
from .test_probe_frames import _raw


def fake_probe(rate="30/1", duration="9000.0", **kw) -> Probe:
    return parse_probe("game.mp4", _raw(rate=rate, duration=duration, **kw))


CROP = [0.1875, 0.0, 0.665625, 1.0]


# -- planning ----------------------------------------------------------------------


def test_plan_snaps_to_frames_and_merges_touching_segments():
    plan = plan_export(fake_probe(), [(10.01, 20.49), (20.5, 25.0), (100.0, 104.02)], CROP)
    assert [(s.first_frame, s.n_frames) for s in plan.segments] == [(300, 450), (3000, 121)]
    assert plan.segments[0].src_in == pytest.approx(10.0)
    assert plan.duration == pytest.approx((450 + 121) / 30)
    assert plan.crop[2] % 2 == 0 and plan.crop[3] == 1080
    assert plan.placement[0] == OUTPUT_W and plan.placement[2] == (OUTPUT_H - plan.placement[1]) // 2


def test_plan_crossfade_is_whole_frames_between_60_and_100_ms():
    for rate in ("30/1", "30000/1001", "25/1", "24000/1001", "60/1", "60000/1001", "50/1"):
        plan = plan_export(fake_probe(rate=rate), [(1, 5), (10, 14)], CROP)
        assert 0.060 <= plan.crossfade <= 0.100, rate
        frames = plan.half * plan.fps
        assert abs(frames - round(frames)) < 1e-9, "a whole number of frames, so video and audio stay aligned"
    assert plan_export(fake_probe(), [(1, 5), (10, 14)], CROP, crossfade=False).half == 0
    assert plan_export(fake_probe(), [(1, 5)], CROP).half == 0, "one segment has no cut to fade"


def test_plan_is_exact_at_ntsc_rates_and_caps_at_60():
    plan = plan_export(fake_probe(rate="30000/1001"), [(100.0, 110.0)], CROP)
    assert plan.fps == Fraction(30000, 1001)
    seg = plan.segments[0]
    assert seg.first_frame == 2997 and seg.n_frames == 300
    assert seg.src_in == pytest.approx(2997 * 1001 / 30000)
    assert plan_export(fake_probe(rate="120/1"), [(1, 5)], CROP).out_fps == 60


def test_plan_rejects_an_empty_cut():
    with pytest.raises(ex.ExportError):
        plan_export(fake_probe(), [], CROP)


def test_filter_graph_keeps_audio_and_video_the_same_length():
    probe = fake_probe()
    plan = plan_export(probe, [(10, 20), (30, 34), (50, 62.5)], CROP)
    inputs, graph, out_args = build_batch(plan, probe, 0, 2, Path("overlay.png"), final=True)
    assert inputs.count("-ss") == 3 and inputs.count("-i") == 4, "three seeked inputs plus the overlay"
    assert graph.count("trim=start=") == 6 and "concat=n=3:v=1:a=0" in graph
    assert graph.count("acrossfade=") == 2
    x, y, w, h = plan.crop
    assert f"crop={w}:{h}:{x}:{y}" in graph and "pad=1080:1920:0:" in graph and "overlay=0:0" in graph
    import re

    audio = [float(m) for m in re.findall(r"atrim=start=[\d.]+:duration=([\d.]+)", graph)]
    assert sum(audio) - 2 * plan.crossfade == pytest.approx(plan.duration, abs=1e-4), "each crossfade removes what the padding added"
    assert out_args[out_args.index("-crf") + 1] == "18" and "+faststart" in out_args
    assert out_args[out_args.index("-profile:v") + 1] == "high" and out_args[out_args.index("-b:a") + 1] == "192k"


def test_filter_graph_for_awkward_sources():
    interlaced = fake_probe(width=1440, sar="4:3", field_order="tt", acodec=None)
    plan = plan_export(interlaced, [(10, 20), (30, 40)], CROP)
    inputs, graph, _ = build_batch(plan, interlaced, 0, 1, None, final=True)
    assert "bwdif" in graph, "interlaced sources are deinterlaced"
    assert "scale=1920:1080" in graph, "anamorphic sources are squared before cropping"
    assert "anullsrc" in " ".join(inputs) and "acrossfade" not in graph, "no source audio: a silent track is added"


# -- overlay -------------------------------------------------------------------------


def test_overlay_text_stays_off_the_video(tmp_path):
    placement = (1080, 912, 504)
    path = render_overlay("Knicks 29-point comeback vs Spurs", "2026 NBA Finals, Game 4", placement, tmp_path / "o.png")
    alpha = np.asarray(Image.open(path))[:, :, 3]
    assert alpha.shape == (OUTPUT_H, OUTPUT_W)
    assert alpha[:504].max() == 255, "title in the top bar"
    assert alpha[504 + 912 :].max() == 255, "caption in the bottom bar"
    assert alpha[504 : 504 + 912].max() == 0, "nothing is drawn over the video itself"
    rows = np.flatnonzero(alpha[:504].max(axis=1))
    assert rows.max() < 504 - 20 and rows.min() > 100, "inside the bar, clear of the phone UI at the very top"
    cols = np.flatnonzero(alpha[:504].max(axis=0))
    assert abs((cols.min() + cols.max()) / 2 - 540) < 12, "centred"
    assert render_overlay("", "", placement, tmp_path / "none.png") is None


def test_overlay_wraps_and_shrinks_long_titles(tmp_path):
    long_title = "New York Knicks erase a 29-point third quarter deficit against the San Antonio Spurs in Game 4"
    path = render_overlay(long_title, "", (1080, 912, 504), tmp_path / "long.png")
    alpha = np.asarray(Image.open(path))[:, :, 3]
    assert alpha[504:].max() == 0
    cols = np.flatnonzero(alpha.max(axis=0))
    assert cols.min() >= 40 and cols.max() <= 1040, "kept inside the side margins"


# -- a real render ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def coverage_export(coverage_analysis, coverage_calibration, coverage_probe, coverage_video):
    """The home cut of the coverage game, rendered once and cached."""
    tl, events = coverage_analysis
    cal, *_ = coverage_calibration
    clips = build_clips(tl, events, get_adapter("nba"), "home")
    segments = [s for c in clips for s in c.segments]
    plan = plan_export(coverage_probe, segments, cal.crop)
    h = hashlib.sha256(_calibration_version().encode())
    for name in ("export.py", "clips.py", "timeline.py", "events.py", "sampler.py"):
        h.update((BACKEND / "possession_cut" / "pipeline" / name).read_bytes())
    h.update((BACKEND / "possession_cut" / "sports" / "nba.py").read_bytes())
    out = _cache_dir() / f"export_{h.hexdigest()[:12]}.mp4"
    work = _cache_dir() / "export_work"
    seen: list[float] = []
    if not out.exists():
        for stale in _cache_dir().glob("export_*.mp4"):
            stale.unlink(missing_ok=True)
        overlay = render_overlay("Knicks comeback vs Spurs", "Synthetic test game", plan.placement, work / "overlay.png")
        render(plan, coverage_probe, out, overlay, work, progress=lambda f, m: seen.append(f))
        assert seen and seen[-1] == 1.0 and seen == sorted(seen), "progress only moves forward"
    return out, plan, clips


def ffprobe_json(path: Path) -> dict:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def exported_barcodes(path: Path, plan, truth: dict) -> list[int | None]:
    """Source frame index carried by each exported frame, read at the mapped barcode position."""
    bc = truth["layout"]["barcode"]
    cx, cy, cw, ch = plan.crop
    sw, sh, pad_y = plan.placement
    sx, sy = sw / cw, sh / ch
    c = bc["cell"]
    x0 = (bc["x0"] - cx) * sx
    y0 = pad_y + (bc["y0"] - cy) * sy
    w, h = BARCODE_CELLS * c * sx, c * sy
    ex0, ey0 = int(x0) - int(x0) % 2, int(y0) - int(y0) % 2
    ew, eh = int(w) + 4 + int(w) % 2, int(h) + 4 + int(h) % 2
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-i", str(path), "-vf", f"crop={ew}:{eh}:{ex0}:{ey0},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, eh, ew)
    out = []
    for frame in frames:
        cells = []
        for i in range(BARCODE_CELLS):
            px = x0 - ex0 + (i + 0.5) * c * sx
            py = y0 - ey0 + 0.5 * c * sy
            cells.append(float(frame[int(py) - 2 : int(py) + 3, int(px) - 2 : int(px) + 3].mean()))
        out.append(decode_barcode(cells))
    return out


@pytest.mark.video
def test_export_passes_ffprobe_checks(coverage_export):
    path, plan, _ = coverage_export
    info = ffprobe_json(path)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert (video["width"], video["height"]) == (1080, 1920), "9:16 canvas"
    assert video["codec_name"] == "h264" and video["profile"] == "High" and video["pix_fmt"] == "yuv420p"
    assert Fraction(video["avg_frame_rate"]) == plan.out_fps == 30, "source frame rate"
    assert audio["codec_name"] == "aac" and audio["sample_rate"] == "48000" and audio["channels"] == 2
    assert 150_000 <= int(audio["bit_rate"]) <= 230_000, "AAC at about 192 kbps"
    assert float(info["format"]["duration"]) == pytest.approx(plan.duration, abs=0.06)
    assert float(audio["duration"]) == pytest.approx(float(video["duration"]), abs=0.06), "audio as long as video"
    head = path.read_bytes()[:200_000]
    assert 0 < head.find(b"moov") < head.find(b"mdat"), "+faststart: index before media"


@pytest.mark.video
def test_export_contains_exactly_the_planned_source_frames(coverage_export, coverage_video):
    path, plan, _ = coverage_export
    _, truth = coverage_video
    got = exported_barcodes(path, plan, truth)
    want = [seg.first_frame + k for seg in plan.segments for k in range(seg.n_frames)]
    assert len(got) == len(want)
    assert None not in got, "every exported frame carries a readable barcode: crop and placement are right"
    assert got == want, "hard cuts on exactly the planned frames, in order"


@pytest.mark.video
def test_no_replay_or_commercial_frame_is_in_the_export(coverage_export, coverage_video):
    path, plan, _ = coverage_export
    _, truth = coverage_video
    fps = truth["fps"]
    not_live = truth["intervals"]["not_live"]
    for n in exported_barcodes(path, plan, truth):
        t = n / fps
        assert not any(a <= t < b for a, b in not_live), f"source frame {n} (t={t:.2f}) is not live"


@pytest.mark.video
def test_export_canvas_bars_title_and_crop(coverage_export, coverage_calibration, coverage_video):
    path, plan, _ = coverage_export
    cal, *_ = coverage_calibration
    _, truth = coverage_video
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", "3", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True,
    ).stdout
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(OUTPUT_H, OUTPUT_W, 3)
    sw, sh, pad_y = plan.placement
    assert sw == 1080 and abs(pad_y - (1920 - sh) / 2) <= 1, "crop scaled to 1080 wide, centred vertically"
    assert 1.15 <= sw / sh <= 1.21, "near-square crop, about 1.18:1"
    assert frame[:60].max() < 20 and frame[-60:].max() < 20, "black canvas above and below"
    top_bar = frame[: pad_y - 4]
    assert top_bar.max() > 200, "white title text in the top bar"
    assert frame[pad_y + 20 : pad_y + sh - 20].mean() > 40, "the picture sits in the middle"
    # the score bug spans the bottom of the crop, about 78% of its width, fully visible
    bx0, by0, bx1, by1 = truth["layout"]["bug_px"]
    cx, cy, cw, ch = plan.crop
    assert (bx1 - bx0) / cw == pytest.approx(0.78, abs=0.01)
    assert cx <= bx0 and bx1 <= cx + cw and by1 <= cy + ch
    left_margin, right_margin = bx0 - cx, (cx + cw) - bx1
    assert abs(left_margin - right_margin) <= 2, "even margins either side of the bug"


@pytest.mark.video
def test_audio_stays_in_sync_through_the_cuts(coverage_export, coverage_video):
    """The synthetic source beeps at 880 Hz on every made basket. Each beep must land in
    the export where its picture does."""
    path, plan, clips = coverage_export
    _, truth = coverage_video
    pcm = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768

    def tone(t0: float, freq: float = 880.0, dur: float = 0.12) -> float:
        i0 = int(t0 * 48000)
        chunk = audio[i0 : i0 + int(dur * 48000)]
        tt = np.arange(len(chunk)) / 48000
        return float(np.hypot((chunk * np.sin(2 * np.pi * freq * tt)).mean(), (chunk * np.cos(2 * np.pi * freq * tt)).mean()))

    checked = 0
    for ev in truth["score_events"]:
        if ev["team"] != "home" or ev["kind"] != "fg":
            continue
        out_t = 0.0
        for seg in plan.segments:
            seg_out = seg.src_in + seg.duration(plan.fps)
            if seg.src_in <= ev["make_time"] < seg_out - 0.2:
                t = out_t + (ev["make_time"] - seg.src_in)
                assert tone(t + 0.01) > 8 * tone(t - 0.5), f"beep for the make at {ev['make_time']:.1f}s is where it should be"
                assert tone(t + 0.01) > 4 * tone(t + 0.4)
                checked += 1
                break
            out_t += seg.duration(plan.fps)
    assert checked == 8, "all eight home baskets are in the cut with their sound"
    assert np.abs(audio).max() < 0.99, "no clipping at the crossfades"


@pytest.mark.video
def test_long_cuts_render_in_batches_with_the_same_result(coverage_probe, coverage_calibration, coverage_video, tmp_path, monkeypatch):
    cal, *_ = coverage_calibration
    _, truth = coverage_video
    segments = [(5.0, 6.5), (20.0, 21.0), (40.2, 41.4), (300.0, 301.5), (330.0, 331.0)]
    plan = plan_export(coverage_probe, segments, cal.crop)
    monkeypatch.setattr(ex, "MAX_INPUTS", 2)
    out = render(plan, coverage_probe, tmp_path / "batched.mp4", None, tmp_path / "work")
    got = exported_barcodes(out, plan, truth)
    want = [seg.first_frame + k for seg in plan.segments for k in range(seg.n_frames)]
    assert got == want
    info = ffprobe_json(out)
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert audio["codec_name"] == "aac" and float(audio["duration"]) == pytest.approx(plan.duration, abs=0.08)
    assert not list((tmp_path / "work").glob("export_part_*")), "intermediate parts are cleaned up"


@pytest.mark.video
def test_cutlist_sidecar(coverage_export, tmp_path):
    _, plan, clips = coverage_export
    path = tmp_path / "cutlist.json"
    write_cutlist(path, {"title": "Knicks comeback vs Spurs", "source": "coverage.mp4"}, [c.to_dict() for c in clips], plan)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["title"] and doc["render"]["canvas"] == [1080, 1920] and len(doc["clips"]) == 9
    first = doc["clips"][0]
    for key in ("src_in", "src_out", "period", "clock", "score_before", "score_after", "confidence", "segments", "out_start", "out_end"):
        assert key in first
    assert doc["clips"][-1]["out_end"] == pytest.approx(plan.duration, abs=0.2)
    assert 60 <= doc["render"]["audio_crossfade_ms"] <= 100


@pytest.mark.video
def test_thumbnail_and_preview_proxy(coverage_probe, coverage_calibration, tmp_path):
    cal, *_ = coverage_calibration
    thumb = tmp_path / "thumbs" / "clip.jpg"
    thumbnail(coverage_probe, 12.0, cal.crop, thumb)
    img = Image.open(thumb)
    assert img.width == 240 and 1.15 <= img.width / img.height <= 1.21

    # an MKV with AC-3 audio is not something a browser will play
    mkv = tmp_path / "game.mkv"
    subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-ss", "0", "-t", "6", "-i", coverage_probe.path, "-c:v", "copy", "-c:a", "ac3", str(mkv)],
        check=True,
    )
    src = probe_file(mkv)
    assert not src.browser_playable
    proxy = probe_file(make_proxy(src, tmp_path / "proxy.mp4"))
    assert proxy.browser_playable and proxy.audio_codec == "aac" and proxy.video_codec == "h264"
    assert proxy.duration == pytest.approx(src.duration, abs=0.15) and (proxy.width, proxy.height) == (1280, 720)
