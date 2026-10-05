"""What a real broadcast taught the NBA rules (2026 Finals Game 4 on ESPN/ABC).

Its bug shows a score 2.2 to 3.8 s after the make, where the PRD assumed 0.5 to 2 s. The
make's own shot clock reset was being taken for the start of the possession, baskets
scored through a foul were read as free throws, and the graphic sometimes showed a reset
as 23 rather than 24. These tests hold the fixes in place without the footage.
"""

from __future__ import annotations

import statistics

import pytest

from possession_cut.pipeline.clips import build_clips
from possession_cut.pipeline.events import detect_score_events
from possession_cut.pipeline.timeline import build_timeline
from possession_cut.sports import get_adapter
from possession_cut.sports.base import ScoreChange
from possession_cut.synth.script import AWAY, HOME, coverage_game, random_game

from .helpers import grade, run_logic
from .test_timeline import make_raw, running

NBA = get_adapter("nba")
SLOW = (1.6, 2.2)  # seconds added to the bug's lag: 2.2 to 4 s in all, like ESPN's


def possession(start_clock: float, seconds: float, home: int, shot_from: int = 24) -> list[dict]:
    """Live play: clock and shot clock running."""
    rows = running(start_clock, int(seconds * 2), home=home)
    for i, r in enumerate(rows):
        r["shot"] = shot_from - i // 2
    return rows


# -- whole games with a slow bug ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(1, 9))
def test_slow_bug_random_games_meet_the_acceptance_criteria(seed):
    g = random_game(seed, extra_lag=SLOW)
    assert all(2.0 <= e.visible_time - e.make_time <= 4.1 for e in g.events)
    for team in (HOME, AWAY):
        _, events, clips = run_logic(g, team)
        assert len(events) == len(g.events)
        result = grade(clips, g.cutlist(team), g.not_live_intervals())
        assert result.ok, (seed, team, result.problems, result.missed, result.extra, result.leaks)


def test_slow_bug_coverage_game():
    g = coverage_game(extra_lag=SLOW)
    for team in (HOME, AWAY):
        _, _, clips = run_logic(g, team)
        result = grade(clips, g.cutlist(team), g.not_live_intervals())
        assert result.ok, (team, result.problems, result.missed, result.extra, result.leaks)
    _, _, home = run_logic(g, HOME)
    assert [c.kind for c in home] == ["field_goal", "field_goal", "free_throws"] + ["field_goal"] * 6
    assert len(home[3].segments) == 2 and home[3].points == 3, "the and-one still rides on its basket"
    assert min(c.duration for c in home if c.kind == "field_goal") > 5.0, "no clip starts at its own make"


def test_bug_lag_is_measured_from_the_games_own_baskets():
    for extra, low, high in (((0.0, 0.0), 0.8, 2.0), (SLOW, 2.6, 3.9)):
        g = random_game(2, extra_lag=extra)
        tl, _, _ = run_logic(g)
        truth = statistics.median(e.visible_time - e.make_time for e in g.events if e.kind == "fg")
        lag = NBA.bug_lag(tl)
        assert lag is not None and low <= lag <= high
        assert abs(lag - truth) <= 0.6, "within a sample of the scripted lag"
    assert NBA.bug_lag(build_timeline(make_raw(running(300, 40)), NBA)) is None, "no baskets, nothing to measure"


# -- one possession at a time ---------------------------------------------------------------


def one_basket(lag_samples: int) -> tuple[ScoreChange, list]:
    rows = [{"clock": "5:00", "home": 70, "shot": 24}] * 6  # dead ball, 0 to 3 s
    rows += possession(299.5, 10.0, home=70)  # 3 s to 13 s
    after = running(289.5, 30, home=70)  # the make at 13 s: shot clock back to 24, play goes on
    for i, r in enumerate(after):
        r["shot"] = 24 if i < 8 else 24 - (i - 8) // 2
        if i >= lag_samples:
            r["home"] = 72
    tl = build_timeline(make_raw(rows + after), NBA)
    (event,) = detect_score_events(tl, NBA)
    return event, build_clips(tl, [event], NBA, HOME)


def test_the_makes_own_reset_is_not_the_possession_start():
    event, (clip,) = one_basket(lag_samples=6)  # the score shows 3 s after the ball drops
    assert event.t == pytest.approx(16.0)
    assert clip.start_cause == "clock_start"
    assert clip.src_in == pytest.approx(2.0, abs=0.6), "the inbound at 3 s, less the pre-roll"
    assert clip.duration > 12.0, "the whole possession, not the three seconds after the make"


def test_clip_ends_a_beat_after_the_make_once_the_score_has_shown():
    slow, (clip,) = one_basket(lag_samples=6)
    assert clip.src_out == pytest.approx(slow.t + 0.5), "slow bug: the new score has just shown"
    fast, (clip,) = one_basket(lag_samples=2)
    assert fast.t == pytest.approx(14.0)
    assert clip.src_out == pytest.approx(12.75 + 2.5), "fast bug: 2.5 s after the ball dropped"
    assert clip.src_out <= fast.t + 1.5, "never later than the PRD's score + 1.5 s"


def test_reset_the_graphic_shows_as_23_still_counts():
    shots = [14, 13, 13, 12, 12, 23, 23, 22, 22, 21, 21, 20]
    rows = [{**r, "shot": s} for r, s in zip(running(300, len(shots)), shots, strict=True)]
    (reset,) = build_timeline(make_raw(rows), NBA).shot_resets
    assert reset.value == 24.0 and reset.t == pytest.approx(2.25), "between the 12 and the 23"
    assert reset.t_start == pytest.approx(reset.t), "already counting when first seen"


def test_reset_corrected_by_the_operator_takes_the_corrected_value():
    shots = [11, 10, 10, 24, 14, 14, 13, 13, 12, 12]  # 24 shown for a moment, then 14: offensive rebound
    rows = [{**r, "shot": s} for r, s in zip(running(300, len(shots)), shots, strict=True)]
    (reset,) = build_timeline(make_raw(rows), NBA).shot_resets
    assert reset.value == 14.0


def test_misread_shot_clock_is_not_a_reset():
    shots = [10, 10, 9, 9, 23, 8, 8, 7, 7, 6, 6, 5]  # one bad read
    rows = [{**r, "shot": s} for r, s in zip(running(300, len(shots)), shots, strict=True)]
    assert build_timeline(make_raw(rows), NBA).shot_resets == []


def test_basket_through_a_foul_is_a_basket_with_its_free_throw():
    rows = [{"clock": "5:00", "home": 60, "shot": 24}] * 6
    rows += possession(299.5, 9.0, home=60)  # 3 s to 12 s
    whistle = {"clock": "4:51", "shot": 24}  # clock stops on the foul, the ball drops
    rows += [{**whistle, "home": 60}] * 6  # the score shows 3 s later
    rows += [{**whistle, "home": 62}] * 30  # the walk to the line
    rows += [{**whistle, "home": 63}] * 6
    tl = build_timeline(make_raw(rows), NBA)
    basket, free_throw = detect_score_events(tl, NBA)
    assert basket.clock_stopped and basket.stopped_before < 4.0, "stopped, but only since the whistle"
    assert NBA.classify(basket) == "field_goal" and NBA.classify(free_throw) == "free_throws"
    (clip,) = build_clips(tl, [basket, free_throw], NBA, HOME)
    assert clip.kind == "field_goal" and clip.points == 3 and len(clip.segments) == 2
    assert clip.src_in < 3.0, "from the start of the possession"


def test_play_by_play_settles_what_kind_of_score_it_was():
    def change(points: int, stopped_before: float, kinds: list[str]) -> ScoreChange:
        return ScoreChange(
            index=0, team=HOME, points=points, t=100.0, t_prev=99.0, period=2, clock=200.0, clock_before=200.0,
            score_before=50, score_after=50 + points, score_away=40, score_home=50 + points,
            clock_stopped=stopped_before > 3.5, stopped_for=stopped_before, stopped_before=stopped_before,
            extra={"pbp": [{"kind": k} for k in kinds]},
        )

    assert NBA.classify(change(2, 30.0, [])) == "free_throws", "no play-by-play: a long stoppage means the line"
    assert NBA.classify(change(2, 3.0, [])) == "field_goal"
    assert NBA.classify(change(2, 30.0, ["field_goal"])) == "field_goal", "a basket seen late after a long break"
    assert NBA.classify(change(2, 3.0, ["free_throw", "free_throw"])) == "free_throws"
    assert NBA.classify(change(3, 30.0, ["field_goal", "free_throw"])) == "field_goal"
    assert NBA.classify(change(1, 0.0, [])) == "free_throws"


def test_possession_already_under_way_when_the_bug_comes_back():
    rows = [{"clock": "3:00", "home": 68, "shot": 24}] * 6  # timeout
    rows += [{"visible": False}] * 24  # 12 s away; play resumes out of sight
    rows += possession(175.0, 8.0, home=68, shot_from=19)  # back at 15 s, five seconds in
    after = running(167.0, 16, home=68)
    for i, r in enumerate(after):
        r["shot"] = 24
        if i >= 5:
            r["home"] = 70
    tl = build_timeline(make_raw(rows + after), NBA)
    (event,) = detect_score_events(tl, NBA)
    (clip,) = build_clips(tl, [event], NBA, HOME)
    assert clip.start_cause == "bug_returned"
    assert clip.src_in == pytest.approx(15.0, abs=0.01), "from the first frame the bug is back"
    assert not clip.warnings and clip.confidence > 0.9, "not flagged as an over-long possession"
