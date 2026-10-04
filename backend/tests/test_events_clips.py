"""Score events and NBA clip boundaries against scripted ground truth (no video)."""

from __future__ import annotations

import pytest

from possession_cut.pipeline.clips import build_clips
from possession_cut.pipeline.events import detect_score_events
from possession_cut.pipeline.timeline import build_timeline
from possession_cut.sports import get_adapter
from possession_cut.synth.script import AWAY, HOME, ScriptBuilder, random_game

from .helpers import add_ocr_noise, grade, ideal_samples, run_logic
from .test_timeline import make_raw, running

NBA = get_adapter("nba")


def test_every_scripted_score_becomes_one_event(coverage_script):
    tl, events, _ = run_logic(coverage_script)
    truth = coverage_script.events
    assert len(events) == len(truth), "every score detected, none invented"
    for ev, tr in zip(events, truth, strict=True):
        assert ev.team == tr.team and ev.points == tr.points
        assert (ev.score_away, ev.score_home) == (tr.score_away, tr.score_home)
        assert 0 <= ev.t - tr.visible_time <= 0.5, "stamped at the first sample showing the new score"
        assert ev.period == tr.period
        assert ev.clock == pytest.approx(tr.clock, abs=2.5), "bug clock at appearance, a beat after the make"
    kinds = [NBA.classify(e) for e in events]
    assert kinds.count("free_throws") == sum(1 for t in truth if t.kind == "ft")


def test_coverage_game_meets_the_acceptance_criteria(coverage_script):
    for team in (HOME, AWAY):
        _, _, clips = run_logic(coverage_script, team)
        g = grade(clips, coverage_script.cutlist(team), coverage_script.not_live_intervals())
        assert g.ok, (g.problems, g.missed, g.extra, g.leaks)
        assert all(-1.5 <= e <= 1.5 for e in g.start_errors), "starts within 1.5 s of the possession start"
        assert all(abs(e) <= 1.0 for e in g.end_errors), "ends within 1 s of target"


def test_clip_details_on_the_coverage_game(coverage_script):
    _, _, clips = run_logic(coverage_script, HOME)
    assert [c.kind for c in clips] == ["field_goal", "field_goal", "free_throws"] + ["field_goal"] * 6
    assert [c.points for c in clips] == [2, 3, 2, 3, 2, 3, 2, 2, 2]
    ft = clips[2]
    assert len(ft.segments) == 2, "two made free throws, the walk between them cut out"
    assert all(b - a == pytest.approx(4.0, abs=0.01) for a, b in ft.segments), "3 s before to 1 s after each make"
    and_one = clips[3]
    assert len(and_one.segments) == 2 and and_one.points == 3, "basket, then its free throw as a trailing segment"
    assert and_one.segments[1][1] - and_one.segments[1][0] == pytest.approx(4.0, abs=0.01)
    assert clips[0].start_cause == "clock_start" and clips[1].start_cause == "shot_clock"
    for c in clips:
        assert 3.0 <= c.segments[0][1] - c.segments[0][0] <= 30.0
        assert c.confidence > 0.9 and not c.warnings


def test_options_free_throws_and_one_and_opponent(coverage_script):
    g = coverage_script
    _, _, no_ft = run_logic(g, HOME, include_free_throws=False)
    assert [c.kind for c in no_ft].count("free_throws") == 0 and len(no_ft) == 8
    assert len(no_ft[2].segments) == 2, "the and-one free throw stays with its basket"

    _, _, no_and1 = run_logic(g, HOME, include_and_one_ft=False)
    assert len(no_and1) == 9 and len(no_and1[3].segments) == 1 and no_and1[3].points == 2

    _, _, neither = run_logic(g, HOME, include_free_throws=False, include_and_one_ft=False)
    assert grade(neither, g.cutlist(HOME, include_free_throws=False, include_and_one_ft=False), g.not_live_intervals()).ok

    _, _, both = run_logic(g, HOME, include_opponent=True)
    assert {c.team for c in both} == {HOME, AWAY}
    assert len(both) == len(g.cutlist(HOME)) + len(g.cutlist(AWAY))
    assert all(a.src_out <= b.src_in + 1e-6 for a, b in zip(both, both[1:], strict=False)), "ordered, no overlap"


def test_time_window_limits_the_cut(coverage_script):
    tl, events, _ = run_logic(coverage_script)
    late = build_clips(tl, events, NBA, HOME, t_min=150.0)
    assert [c.score_after for c in late] == [100, 103, 105, 107, 109]
    middle = build_clips(tl, events, NBA, HOME, t_min=150.0, t_max=250.0)
    assert [c.score_after for c in middle] == [100, 103, 105]
    # an and-one whose basket is before the window does not leave an orphan free throw
    after_basket = build_clips(tl, events, NBA, HOME, t_min=105.0)
    assert after_basket[0].score_after == 100


@pytest.mark.parametrize("seed", range(1, 13))
def test_random_games_meet_the_acceptance_criteria(seed):
    g = random_game(seed)
    for team in (HOME, AWAY):
        _, events, clips = run_logic(g, team)
        assert len(events) == len(g.events)
        result = grade(clips, g.cutlist(team), g.not_live_intervals())
        assert result.ok, (seed, team, result.problems, result.missed, result.extra, result.leaks)


@pytest.mark.parametrize("phase", [0.13, 0.31, 0.44])
def test_result_does_not_depend_on_where_the_sample_grid_falls(phase):
    g = random_game(3)
    _, _, clips = run_logic(g, HOME, raw=ideal_samples(g, phase=phase))
    assert grade(clips, g.cutlist(HOME), g.not_live_intervals()).ok


@pytest.mark.parametrize(("seed", "rate"), [(1, 0.01), (2, 0.01), (3, 0.02), (5, 0.03), (8, 0.03), (13, 0.03)])
def test_ocr_noise_does_not_break_the_cut(seed, rate):
    """A few percent of corrupted reads: every score still found, nothing invented, nothing
    not-live leaks in, and no possession is cut short. Boundaries may slip a little."""
    g = random_game(seed)
    for team in (HOME, AWAY):
        raw = add_ocr_noise(ideal_samples(g), seed * 7 + (team == HOME), rate)
        _, _, clips = run_logic(g, team, raw=raw)
        result = grade(clips, g.cutlist(team), g.not_live_intervals(), start_tol=2.5, end_tol=2.0)
        assert result.ok, (seed, team, result.problems, result.missed, result.extra, result.leaks)


def test_long_possession_is_trimmed_to_the_last_30_seconds():
    # no shot clock in this bug, clock running for 50 s, then a basket
    rows = [{"clock": "5:00", "home": 60}] * 8 + running(300, 100, home=60)
    rows += [{"clock": "4:10", "home": 62}] * 8
    tl = build_timeline(make_raw(rows), NBA)
    events = detect_score_events(tl, NBA)
    (clip,) = build_clips(tl, events, NBA, HOME)
    assert clip.src_out - clip.src_in == pytest.approx(30.0, abs=0.01)
    assert clip.src_out == pytest.approx(events[0].t + 1.5)
    assert any("trimmed" in w for w in clip.warnings) and clip.confidence < 0.9


def test_quick_score_after_own_basket_merges_into_one_clip():
    b = ScriptBuilder(seed=3, start_scores=(40, 40))
    b.period_start(2, 300.0, ball=HOME)
    b.possession(HOME, 8.0, "make2", anim=False, inbound=2.0)
    b.possession(AWAY, 0.6, "steal")  # stolen on the inbound
    b.possession(HOME, 2.0, "make2", anim=False)
    b.possession(AWAY, 9.0, "miss_def")
    g = b.build()
    _, _, clips = run_logic(g, HOME)
    truth = g.cutlist(HOME)
    assert len(truth) == 1 and truth[0]["points"] == 4, "overlapping clips merge"
    assert len(clips) == 1 and clips[0].points == 4 and len(clips[0].changes) == 2
    assert grade(clips, truth, g.not_live_intervals()).ok


def test_score_first_seen_after_a_break_ends_the_clip_at_the_break():
    # the basket drops, the broadcast cuts to a replay before the bug updates
    rows = [{"clock": "5:00", "home": 70, "shot": 24}] * 6
    rows += [{"clock": r["clock"], "home": 70, "shot": 24 - i // 2} for i, r in enumerate(running(299.5, 16))]
    rows += [{"visible": False}] * 10
    rows += [{"clock": "4:51", "home": 72, "shot": 24}] * 10
    tl = build_timeline(make_raw(rows), NBA)
    events = detect_score_events(tl, NBA)
    assert len(events) == 1 and any("after" in n for n in events[0].notes)
    (clip,) = build_clips(tl, events, NBA, HOME)
    assert clip.src_out <= 11.0 + 1e-6, "ends where the broadcast cut away, not after the replay"
    assert any("break" in w for w in clip.warnings)


def test_unseen_first_free_throw_is_still_a_free_throw_clip():
    # bug hidden between the two free throws: 80 -> 82 with the clock stopped throughout
    rows = running(300, 10, home=80) + [{"clock": "4:55", "home": 80}] * 16
    rows += [{"visible": False}] * 12 + [{"clock": "4:55", "home": 82}] * 10
    tl = build_timeline(make_raw(rows), NBA)
    (event,) = detect_score_events(tl, NBA)
    assert event.points == 2 and event.clock_stopped and NBA.classify(event) == "free_throws"
    (clip,) = build_clips(tl, [event], NBA, HOME)
    assert clip.kind == "free_throws" and any("free throws read as one" in w for w in clip.warnings)


def test_basket_with_clock_stopped_after_the_make_is_not_a_free_throw():
    # last two minutes: clock stops on the make, the score shows a second later
    rows = running(110, 20, home=90, shot=20) + [{"clock": "1:40", "home": 90, "shot": 24}] * 2
    rows += [{"clock": "1:40", "home": 92, "shot": 24}] * 8
    tl = build_timeline(make_raw(rows), NBA)
    (event,) = detect_score_events(tl, NBA)
    assert not event.clock_stopped and NBA.classify(event) == "field_goal"
