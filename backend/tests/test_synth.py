"""The synthetic game is the test harness for everything else, so it gets its own tests."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from possession_cut.config import ffmpeg_bin
from possession_cut.synth.render import (
    BARCODE_CELLS,
    SyntheticRenderer,
    _barcode_bits,
    bug_layout,
    decode_barcode,
    read_barcode,
)
from possession_cut.synth.script import (
    AWAY,
    HOME,
    format_clock,
    format_shot_clock,
    random_game,
)


def test_clock_formats():
    assert format_clock(150.0) == "2:30"
    assert format_clock(149.99) == "2:29"
    assert format_clock(60.0) == "1:00"
    assert format_clock(59.95) == "59.9"
    assert format_clock(4.32) == "4.3"
    assert format_clock(0.0) == "0.0"
    assert format_shot_clock(24.0) == "24"
    assert format_shot_clock(23.2) == "24"
    assert format_shot_clock(0.4) == "1"
    assert format_shot_clock(None) == ""


def test_coverage_game_has_every_situation(coverage_script):
    g = coverage_script
    kinds = {k for _, _, k in g.hidden}
    assert kinds == {"commercial", "replay_hidden"}
    assert len(g.replays) == 2, "two replays re-air an earlier clock and score"
    assert any(e.kind == "ft" and e.and_one_of is not None for e in g.events)
    assert any(e.kind == "ft" and e.and_one_of is None for e in g.events)
    assert {e.start_cause for e in g.events if e.kind == "fg"} >= {
        "clock_start", "def_rebound", "steal", "off_rebound", "after_make",
    }
    assert len(g.anims) >= 4
    assert max(e.score_away for e in g.events) >= 100, "a score crosses into three digits"
    assert {e.period for e in g.events} == {3, 4}


def test_scores_never_decrease_and_clock_only_runs_down(coverage_script):
    g = coverage_script
    prev = None
    for i in range(int(g.duration * 10)):
        s = g._base_state(i / 10)
        if prev is not None:
            assert s.away_score >= prev.away_score and s.home_score >= prev.home_score
            if s.period == prev.period:
                assert s.clock <= prev.clock + 1e-6
        prev = s


def test_replay_shows_an_earlier_game_state(coverage_script):
    g = coverage_script
    r0, _r1, _src0 = g.replays[1]  # the and-one replay
    during = g.state_at(r0 + 0.5)
    before = g.state_at(r0 - 0.1)
    assert during.scene == "replay_bug" and during.visible
    assert during.clock > before.clock, "clock jumps backward (shows more time) in a replay"
    assert during.home_score < before.home_score, "and the score from before the basket"


def test_cutlist_follows_prd_rules(coverage_script):
    g = coverage_script
    clips = g.cutlist(HOME)
    assert [c["kind"] for c in clips] == [
        "field_goal", "field_goal", "free_throws", "field_goal", "field_goal",
        "field_goal", "field_goal", "field_goal", "field_goal",
    ]
    assert sum(c["points"] for c in clips) == 109 - 88
    not_live = g.not_live_intervals()
    for c in clips:
        for a, b in c["segments"]:
            assert b > a
            for h0, h1 in not_live:
                # cut list values are rounded to the millisecond
                assert b <= h0 + 2e-3 or a >= h1 - 2e-3, "no clip segment overlaps a replay or commercial"
        if c["kind"] == "field_goal":
            first = c["segments"][0]
            assert first[0] >= c["possession_start"] - 1.0 - 0.01
            assert 3.0 <= first[1] - first[0] <= 30.0
    ft = clips[2]
    assert ft["points"] == 2 and len(ft["segments"]) == 2
    # each free throw: 2 s before the ball drops to about 2 s after, the score showing by the end
    made = [e for e in g.events if e.team == HOME and e.kind == "ft" and e.and_one_of is None][:2]
    for (a, b), ev in zip(ft["segments"], made, strict=True):
        assert a == pytest.approx(ev.make_time - 2.0, abs=0.01)
        assert ev.visible_time + 0.5 - 0.01 <= b <= ev.visible_time + 1.0 + 0.01
        assert 3.5 <= b - a <= 4.5
    and_one = clips[3]
    assert and_one["points"] == 3 and len(and_one["segments"]) == 2, "and-one FT rides on its basket clip"

    without_ft = g.cutlist(HOME, include_free_throws=False, include_and_one_ft=False)
    assert all(c["kind"] == "field_goal" and len(c["segments"]) == 1 for c in without_ft)
    assert len(without_ft) == 8


@pytest.mark.parametrize("seed", range(1, 26))
def test_random_games_are_valid(seed):
    g = random_game(seed)
    assert 500 < g.duration < 2400
    for team in (AWAY, HOME):
        clips = g.cutlist(team)
        own = [e for e in g.events if e.team == team]
        assert sum(c["points"] for c in clips) == sum(e.points for e in own)
        for a, b in zip(clips, clips[1:], strict=False):
            assert a["segments"][-1][1] < b["segments"][0][0], "clips are ordered and do not overlap"
    for ev in g.events:
        assert 0.4 <= ev.visible_time - ev.make_time <= 2.0, "bug lags the make by 0.5 to 2 s"
    assert len(g.play_by_play()) == len(g.events)


def test_barcode_roundtrip():
    for n in (0, 1, 150, 11544, 2**20 - 1):
        bits = _barcode_bits(n)
        assert len(bits) == BARCODE_CELLS
        assert decode_barcode([255.0 * b for b in bits]) == n
    bad = [255.0 * b for b in _barcode_bits(1234)]
    bad[5] = 255.0 - bad[5]
    assert decode_barcode(bad) is None, "a flipped bit fails the checksum"


def test_layout_crop_matches_prd_rule():
    lay = bug_layout(1280, 720, 0.02)
    x0, _y0, x1, _y1 = lay["bug"]
    cx, cy, cw, ch = lay["crop"]
    assert ch == 720 and cy == 0
    assert (x1 - x0) / cw == pytest.approx(0.78, abs=0.005)
    assert cx + cw / 2 == pytest.approx((x0 + x1) / 2, abs=1.0), "crop is centred on the bug, not the frame"
    assert cw / ch == pytest.approx(1.18, abs=0.02)


def test_bug_is_static_while_background_moves(coverage_script):
    r = SyntheticRenderer(coverage_script)
    x0, y0, x1, y1 = r.layout["blocks"]["away_logo"]
    a, b = r.frame(150), r.frame(240)
    assert np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])
    assert np.abs(a[300:400].astype(int) - b[300:400].astype(int)).mean() > 5


@pytest.mark.video
def test_rendered_video_matches_truth(coverage_video):
    path, truth = coverage_video
    assert truth["width"] == 1280 and truth["height"] == 720
    assert len(truth["cutlists"]["home"]) == 9
    lay = bug_layout(1280, 720, 0.02)
    # decode three frames straight from the file and read their barcodes back
    for t in (5.0, 135.0, 300.0):
        raw = subprocess.run(
            [ffmpeg_bin(), "-v", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, check=True,
        ).stdout
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(720, 1280, 3)
        assert read_barcode(frame, lay) == round(t * truth["fps"])
