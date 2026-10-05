"""Keeping clip edges on the game camera: crowd shots and close-ups the bug cannot see."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from possession_cut.pipeline import camera
from possession_cut.pipeline.camera import (
    CAMERA_FPS,
    FRAME_H,
    FRAME_W,
    cutaway_flags,
    learn_surface,
    trim_edges,
)
from possession_cut.pipeline.clips import ClipDraft
from possession_cut.sports.base import ScoreChange
from possession_cut.synth.render import SyntheticRenderer
from possession_cut.synth.script import AWAY, HOME, ScriptBuilder


def small(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)


def flags(text: str) -> np.ndarray:
    """'xx....x' -> cutaway flags, one character per sample."""
    return np.array([c == "x" for c in text])


def cutaway_game():
    """Two home baskets: one whose possession starts under a crowd shot and ends in a
    close-up of the scorer, one shown on the game camera throughout."""
    b = ScriptBuilder(seed=11, start_scores=(50, 50))
    b.period_start(2, 400.0, ball=AWAY)
    b.possession(AWAY, 6.0, "dead_turnover", dead=6.0)
    b.possession(HOME, 9.0, "make2", anim=False, cutaway_lead=3.0, closeup_after=(1.8, 4.0), inbound=6.5)
    b.possession(AWAY, 7.0, "miss_def")
    b.possession(HOME, 8.0, "make3", anim=False)
    b.possession(AWAY, 6.0, "miss_def")
    return b.build()


# -- the picture ---------------------------------------------------------------------------


def test_surface_colour_separates_the_game_camera_from_cutaways():
    script = cutaway_game()
    r = SyntheticRenderer(script, width=640, height=360, fps=10)
    (c0, c1), _ = script.cutaways
    game_times = [t / 2 for t in range(0, int(script.duration * 2)) if not script.in_cutaway(t / 2) and script.is_live(t / 2)]
    game = np.stack([small(r.frame(int(t * r.fps))) for t in game_times])
    away = np.stack([small(r.frame(int(t * r.fps))) for t in np.arange(c0 + 0.2, c1 - 0.2, 0.5)])
    model = learn_surface(np.concatenate([game, away]))
    assert model is not None and model.typical > 0.2
    assert not cutaway_flags(game, model).any(), "every game-camera frame shows the floor"
    assert cutaway_flags(away, model).all(), "a crowd shot shows none of it"

    # a camera flash whites out the frame; a frame in near darkness shows nothing either
    flash = np.full_like(game[:1], 250)
    assert cutaway_flags(flash, model).all()


def test_no_surface_colour_means_no_judgement():
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (40, FRAME_H, FRAME_W, 3), dtype=np.uint8)
    dark = np.full((40, FRAME_H, FRAME_W, 3), 20, dtype=np.uint8)
    assert learn_surface(dark) is None, "nothing but shadow"
    assert learn_surface(noise[:4]) is None, "too few frames"
    model = learn_surface(noise)
    assert model is None or not cutaway_flags(noise, model).any()


def test_ice_counts_as_a_surface():
    rng = np.random.default_rng(5)
    rink = np.full((30, FRAME_H, FRAME_W, 3), 235, dtype=np.uint8)
    rink[:, : FRAME_H // 4] = rng.integers(20, 90, (30, FRAME_H // 4, FRAME_W, 3))  # the crowd
    bench = rng.integers(20, 120, (4, FRAME_H, FRAME_W, 3), dtype=np.uint8)
    model = learn_surface(np.concatenate([rink, bench]))
    assert model is not None
    assert not cutaway_flags(rink, model).any() and cutaway_flags(bench, model).all()


# -- what to trim ------------------------------------------------------------------------


def test_leading_cutaway_is_trimmed_to_the_first_game_frame():
    assert trim_edges(flags("xxxxxx" + "." * 30), 18.0, 3.0) == (3.0, 0.0)
    # the clip can open on a moment of game camera before the director cuts away
    assert trim_edges(flags("..xxxxxxxx" + "." * 30), 20.0, 3.0) == (5.0, 0.0)
    # a cutaway that starts well into the possession is part of the play: leave it
    assert trim_edges(flags("." * 8 + "xxxx" + "." * 20), 16.0, 3.0) == (0.0, 0.0)
    # one odd frame is a camera flash, not a cutaway
    assert trim_edges(flags("x" + "." * 20), 10.5, 3.0) == (0.0, 0.0)
    assert trim_edges(flags("." * 10 + "x" + "." * 10), 10.5, 3.0) == (0.0, 0.0)


def test_trailing_cutaway_is_trimmed_back_to_the_last_game_frame():
    # 20 samples = 10 s; the last two are a close-up. Last game frame at 8.5 s.
    lead, tail = trim_edges(flags("." * 18 + "xx"), 10.0, 3.0)
    assert lead == 0.0 and tail == pytest.approx(10.0 - 8.6)
    # a single close-up sample at the very end counts: there is nothing after it to disprove it
    assert trim_edges(flags("." * 19 + "x"), 10.0, 3.0)[1] == pytest.approx(0.9)
    # the play itself ended on another camera (a long run): do not guess
    assert trim_edges(flags("." * 10 + "x" * 10), 10.0, 3.0) == (0.0, 0.0)


def test_trimming_never_guts_a_clip():
    assert trim_edges(flags("x" * 20), 10.0, 3.0) == (0.0, 0.0), "all on another camera: left alone"
    assert trim_edges(flags(""), 0.0, 3.0) == (0.0, 0.0)
    # trimming the start would leave under the minimum length: keep the start
    assert trim_edges(flags("xxxxxxxx" + ".."), 5.0, 3.0) == (0.0, 0.0)
    # both ends flagged but only the end fits
    lead, tail = trim_edges(flags("xxxxxx" + "..." + "x"), 5.0, 3.0)
    assert lead == 0.0 and tail > 0


# -- on video ---------------------------------------------------------------------------------


@pytest.mark.video
def test_clips_are_trimmed_to_the_game_camera_on_video(tmp_path):
    from possession_cut.pipeline.probe import probe_file
    from possession_cut.synth.render import render_video

    script = cutaway_game()
    video = tmp_path / "cutaway.mp4"
    render_video(script, video, width=1280, height=720, fps=30)
    probe = probe_file(video)
    (lead0, lead1), (close0, close1) = script.cutaways
    truth = script.cutlist(HOME)
    assert len(truth) == 2

    def draft(tc: dict) -> ClipDraft:
        seen = tc["visible_times"][0]
        change = ScoreChange(
            index=0, team=HOME, points=tc["points"], t=seen, t_prev=seen - 0.5, period=2, clock=0.0, clock_before=0.0,
            score_before=tc["score_before"], score_after=tc["score_after"], score_away=50, score_home=tc["score_after"],
        )
        return ClipDraft(
            team=HOME, kind=tc["kind"], points=tc["points"], period=2, clock=0.0, score_before=tc["score_before"],
            score_after=tc["score_after"], score_away=50, score_home=tc["score_after"],
            segments=[list(s) for s in tc["segments"]], changes=[change], start_cause="clock_start",
        )

    clips = [draft(tc) for tc in truth]
    untouched = [list(s) for s in clips[1].segments]
    assert clips[0].src_in < lead1 and clips[0].src_out > close0, "the scripted clip does run into both cutaways"

    stats = camera.refine_clips(probe, clips, min_keep=3.0)
    assert stats["checked"] == 2 and stats["trimmed"] == 1 and stats["seconds_removed"] > 2.0

    first = clips[0]
    assert lead1 - 0.05 <= first.src_in <= lead1 + 1.0 / CAMERA_FPS + 0.05, "starts on the first game-camera frame"
    assert first.src_out <= close0 + 0.05, "ends before the close-up"
    make = next(e.make_time for e in script.events if e.team == HOME)
    assert first.src_out >= make + 1.0, "and still shows the ball going in"
    assert "camera" in first.start_cause
    assert clips[1].segments == untouched, "a clip on the game camera throughout is not touched"

    # free throws are short windows from their own camera angle: never judged
    ft = draft(truth[0])
    ft.kind = "free_throws"
    before = [list(s) for s in ft.segments]
    camera.refine_clips(probe, [ft], min_keep=3.0)
    assert ft.segments == before
