from __future__ import annotations

import subprocess

import numpy as np
import pytest

from possession_cut.config import ffmpeg_bin
from possession_cut.pipeline.frames import extract_frame, stream_roi
from possession_cut.pipeline.probe import ProbeError, parse_probe, probe_file
from possession_cut.synth.render import BARCODE_CELLS, bug_layout, decode_barcode, read_barcode


def _raw(width=1920, height=1080, sar="1:1", rate="30000/1001", field_order="progressive", codec="h264",
         acodec="aac", fmt="mov,mp4,m4a,3gp,3g2,mj2", duration="9000.5"):
    streams = [{
        "codec_type": "video", "codec_name": codec, "width": width, "height": height,
        "sample_aspect_ratio": sar, "avg_frame_rate": rate, "r_frame_rate": rate,
        "pix_fmt": "yuv420p", "field_order": field_order,
    }]
    if acodec:
        streams.append({"codec_type": "audio", "codec_name": acodec, "channels": 2, "sample_rate": "48000"})
    return {"streams": streams, "format": {"format_name": fmt, "duration": duration, "size": "123", "bit_rate": "8000000"}}


def test_parse_probe_basic():
    p = parse_probe("game.mp4", _raw())
    assert (p.width, p.height, p.display_width) == (1920, 1080, 1920)
    assert (p.fps_num, p.fps_den) == (30000, 1001)
    assert p.fps == pytest.approx(29.97, abs=0.001)
    assert p.browser_playable and p.has_audio and not p.interlaced and not p.anamorphic


def test_parse_probe_broadcast_ts_is_not_browser_playable():
    p = parse_probe("game.ts", _raw(width=1440, sar="4:3", field_order="tt", codec="mpeg2video", acodec="ac3", fmt="mpegts"))
    assert p.display_width == 1920 and p.anamorphic, "anamorphic HD is stretched to square pixels"
    assert p.interlaced
    assert not p.browser_playable


def test_export_fps_caps_at_60():
    assert float(parse_probe("a.mp4", _raw(rate="120/1")).export_fps) == 60.0
    assert float(parse_probe("a.mp4", _raw(rate="60000/1001")).export_fps) == pytest.approx(59.94, abs=0.01)


def test_rejects_under_480p_and_missing_video():
    with pytest.raises(ProbeError, match="480p"):
        parse_probe("small.mp4", _raw(width=640, height=360))
    with pytest.raises(ProbeError, match="No video"):
        parse_probe("audio.mp4", {"streams": [{"codec_type": "audio"}], "format": {"duration": "10"}})
    with pytest.raises(ProbeError, match="not found"):
        probe_file("does-not-exist.mp4")


@pytest.mark.video
def test_rejects_real_small_file(tmp_path):
    small = tmp_path / "small.mp4"
    subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(small)],
        check=True,
    )
    with pytest.raises(ProbeError, match="480p"):
        probe_file(small)


@pytest.mark.video
def test_probe_synthetic(coverage_video, tmp_path):
    path, truth = coverage_video
    p = probe_file(path, tmp_path / "probe.json")
    assert (p.width, p.height) == (1280, 720)
    assert p.fps == 30.0
    assert p.duration == pytest.approx(truth["duration"], abs=0.2)
    assert p.video_codec == "h264" and p.audio_codec == "aac" and p.browser_playable
    assert (tmp_path / "probe.json").exists()


@pytest.mark.video
def test_extract_frame_is_time_accurate(coverage_video):
    path, truth = coverage_video
    p = probe_file(path)
    lay = bug_layout(1280, 720, 0.02)
    for t in (0.0, 61.5, 200.0):
        frame = extract_frame(p, t)
        assert frame.shape == (720, 1280, 3)
        assert read_barcode(frame, lay) == round(t * 30)
    small = extract_frame(p, 10.0, max_height=360)
    assert small.shape == (360, 640, 3)


@pytest.mark.video
@pytest.mark.parametrize("start", [0.0, 100.0, 200.5])
def test_roi_stream_samples_land_on_their_timestamps(coverage_video, start):
    """Sample k must be the frame at start + k/fps, not a neighbour. (ffmpeg's fps filter
    picks a frame a quarter second late with its default rounding.)"""
    path, truth = coverage_video
    p = probe_file(path)
    bc = truth["layout"]["barcode"]
    c = bc["cell"]
    roi = (bc["x0"], bc["y0"], c * BARCODE_CELLS, c)
    samples = list(stream_roi(p, roi, 2.0, start, 10.0))
    assert len(samples) == 20
    for t, crop in samples:
        cells = [float(crop[c // 4 : c - c // 4, i * c + c // 4 : (i + 1) * c - c // 4].mean()) for i in range(BARCODE_CELLS)]
        n = decode_barcode(cells)
        assert n is not None
        assert abs(n / 30.0 - t) <= 1 / 30 + 1e-6


@pytest.mark.video
@pytest.mark.parametrize("roi", [(333, 641, 665, 45), (101, 77, 51, 33), (0, 0, 64, 32), (1215, 687, 65, 33)])
def test_roi_stream_returns_exactly_the_pixels_asked_for(coverage_video, roi):
    """Odd offsets and sizes: ffmpeg rounds those for subsampled video unless we handle it."""
    path, _ = coverage_video
    p = probe_file(path)
    x, y, w, h = roi
    full = extract_frame(p, 20.0)
    ((t, crop),) = list(stream_roi(p, roi, 2.0, 20.0, 0.5))
    assert t == 20.0 and crop.shape == (h, w, 3)
    diff = np.abs(crop.astype(int) - full[y : y + h, x : x + w].astype(int))
    assert diff.mean() < 2.0, "same pixels as the full frame at that position"


@pytest.mark.video
def test_roi_stream_stops_at_end_of_file(coverage_video):
    path, _ = coverage_video
    p = probe_file(path)
    samples = list(stream_roi(p, (0, 0, 64, 32), 2.0, p.duration - 3.0, 10.0))
    assert 4 <= len(samples) <= 7
    assert all(isinstance(c, np.ndarray) and c.shape == (32, 64, 3) for _, c in samples)
