"""Timeline cleaning rules, on simulated bug reads (no video)."""

from __future__ import annotations

import numpy as np
import pytest

from possession_cut.pipeline.sampler import RawSamples
from possession_cut.pipeline.timeline import (
    STATE_HIDDEN,
    STATE_LIVE,
    STATE_REPLAY,
    Timeline,
    _drop_shot_outliers,
    build_timeline,
    lnds_mask,
    weighted_lnds,
)
from possession_cut.sports import get_adapter
from possession_cut.synth.script import format_clock

from .helpers import add_ocr_noise, ideal_samples

NBA = get_adapter("nba")


def make_raw(rows: list[dict], fps: float = 2.0) -> RawSamples:
    """Rows of {clock, shot, away, home, period, visible} at successive samples."""
    n = len(rows)
    names = ("away_score", "home_score", "period", "clock", "shot_clock")
    raw = RawSamples(
        fps=fps, t=np.arange(n) / fps, visible=np.ones(n, dtype=bool), similarity=np.ones(n, dtype=np.float32),
        texts={k: [""] * n for k in names}, confs={k: np.zeros(n, dtype=np.float32) for k in names},
    )
    for i, row in enumerate(rows):
        raw.visible[i] = row.get("visible", True)
        values = {
            "away_score": row.get("away", 10), "home_score": row.get("home", 10),
            "period": row.get("period", "2nd"), "clock": row.get("clock", "5:00"), "shot_clock": row.get("shot", ""),
        }
        for k, v in values.items():
            if v is None or not raw.visible[i]:
                continue
            raw.texts[k][i] = str(v)
            raw.confs[k][i] = 1.0 if str(v) else 0.0
    return raw


def running(start_clock: float, n: int, fps: float = 2.0, **extra) -> list[dict]:
    return [{"clock": format_clock(start_clock - i / fps), **extra} for i in range(n)]


def test_lnds_keeps_the_longest_consistent_chain():
    values = np.array([1, 2, 9, 3, 4, 4, 2, 5], dtype=float)
    assert lnds_mask(values).tolist() == [True, True, False, True, True, True, False, True]
    assert lnds_mask(np.array([])).tolist() == []
    assert lnds_mask(np.array([3.0, 3.0, 3.0])).all(), "ties are allowed: a stopped clock repeats"


def test_weighted_lnds_prefers_heavy_runs():
    # a long true run beats a short misread that would otherwise extend the chain
    assert weighted_lnds([10, 18, 10, 12], [30, 2, 29, 40]) == [True, False, True, True]
    assert weighted_lnds([], []) == []


def test_scores_never_decrease_and_misreads_are_filled():
    rows = running(300, 40)
    for r in rows:
        r["home"] = 50
    rows[12]["home"] = 80  # one-sample misread upward
    rows[20]["home"] = 5  # one-sample misread downward
    rows[25]["home"] = ""  # unreadable
    tl = build_timeline(make_raw(rows), NBA)
    assert (tl.score_home == 50).all()
    assert not tl.home_read[12] and not tl.home_read[20] and not tl.home_read[25]


def test_new_score_needs_two_samples():
    rows = running(300, 30)
    for i, r in enumerate(rows):
        r["home"] = 50 if i < 15 else 52
    rows[8]["home"] = 52  # a lone early "52" must not count
    tl = build_timeline(make_raw(rows), NBA)
    first_52 = int(np.flatnonzero(tl.score_home == 52)[0])
    assert first_52 == 15, "the event is where the new score starts to hold"


def test_score_holds_through_an_unreadable_animation():
    rows = running(300, 30)
    for i, r in enumerate(rows):
        r["home"] = 50 if i < 14 else ("" if i < 17 else 53)
    tl = build_timeline(make_raw(rows), NBA)
    assert tl.score_home[16] == 50 and tl.score_home[17] == 53
    assert tl.live[14:17].all(), "an unreadable score does not make the sample not live"


def test_clock_misread_is_rejected_and_interpolated():
    rows = running(300, 30)
    rows[10]["clock"] = "0:55"  # dropped digit: far in the future
    rows[18]["clock"] = "7:51"  # misread: in the past
    tl = build_timeline(make_raw(rows), NBA)
    assert tl.live.all(), "isolated misreads are not replays"
    assert tl.clock[10] == pytest.approx(295.0, abs=0.6) and tl.clock[18] == pytest.approx(291.0, abs=0.6)
    assert not tl.clock_read[10] and not tl.clock_read[18]
    live_clock = tl.clock[tl.live]
    assert (np.diff(live_clock) <= 1e-6).all(), "clock only runs down within a period"


def test_bug_hidden_and_backward_clock_are_not_live():
    rows = running(300, 20)  # live
    rows += [{"visible": False}] * 8  # commercial
    stopped = [{"clock": "4:50"}] * 6  # back, clock stopped
    replay = running(298, 10)  # replay re-airs 4:58..4:53 with the bug
    rows += stopped + replay + [{"clock": "4:50"}] * 6
    tl = build_timeline(make_raw(rows), NBA)
    assert (tl.state[20:28] == STATE_HIDDEN).all()
    assert (tl.state[34:44] == STATE_REPLAY).all(), "clock jumped backward for a sustained stretch"
    assert (tl.state[:20] == STATE_LIVE).all() and (tl.state[44:] == STATE_LIVE).all()
    # nothing not-live may be inside a clip: intervals are widened to the neighbouring live samples
    spans = tl.not_live_intervals
    assert spans[0] == (9.5, 14.0) and spans[1] == (16.5, 22.0)


def test_replay_showing_an_older_score_is_flagged_and_does_not_create_an_event():
    live_a = [{"clock": "4:50", "home": 50}] * 6
    new_score = [{"clock": "4:50", "home": 52}] * 2  # the basket registers...
    replay = [{"clock": "4:50", "home": 50}] * 6  # ...then a replay shows the old score, same clock
    back = [{"clock": "4:50", "home": 52}] * 8
    tl = build_timeline(make_raw(running(296, 12, home=50) + live_a + new_score + replay + back), NBA)
    first_52 = int(np.flatnonzero(tl.score_home == 52)[0])
    assert first_52 == 18, "first showing of 52 stands"
    assert (np.diff(tl.score_home) >= 0).all(), "the cleaned score never goes back down"
    assert (tl.state[20:26] == STATE_REPLAY).all()


def test_lone_misread_equal_to_the_next_score_is_not_taken_as_the_basket():
    rows = [{"clock": "4:50", "home": 15}] * 30
    rows[5] = {"clock": "4:50", "home": 16}  # misread
    rows[18:] = [{"clock": "4:50", "home": 16}] * 12  # the real free throw
    tl = build_timeline(make_raw(rows), NBA)
    assert int(np.flatnonzero(tl.score_home == 16)[0]) == 18
    assert tl.live.all()


def test_clock_running_flag():
    rows = running(300, 12) + [{"clock": "4:54"}] * 12 + running(293.5, 10)
    tl = build_timeline(make_raw(rows), NBA)
    assert tl.clock_running[2:10].all()
    assert not tl.clock_running[15:21].any()
    assert tl.clock_running[26:32].all()


def test_clock_start_after_stoppage_is_located():
    rows = running(300, 10) + [{"clock": "4:55", "shot": 24}] * 12
    rows += [{"clock": format_clock(294.9 - i * 0.5), "shot": 24 if i < 2 else 23} for i in range(10)]
    tl = build_timeline(make_raw(rows), NBA)
    starts = tl.clock_starts
    assert len(starts) == 1
    # clock stopped through sample 21 (t=10.5); first lower value at t=11.0
    assert 10.0 <= starts[0].t <= 11.0 and starts[0].stopped_for >= 5.0


def test_one_repeated_clock_read_is_not_a_stoppage():
    rows = running(300, 30)
    rows[10]["clock"] = rows[9]["clock"]  # the same value three samples in a row
    tl = build_timeline(make_raw(rows), NBA)
    assert tl.clock_starts == []


def test_shot_clock_reset_and_when_it_starts_running():
    shots = [8, 8, 7, 7, 6, 6] + [24] * 7 + [23, 23, 22, 22, 21, 21]
    rows = [{"clock": format_clock(300 - i * 0.5), "shot": v} for i, v in enumerate(shots)]
    tl = build_timeline(make_raw(rows), NBA)
    (reset,) = tl.shot_resets
    assert reset.value == 24 and reset.t == pytest.approx(2.75)
    # 24 was held for 3.5 s; it began counting one second before the first 23 (t=6.5)
    assert reset.t_start == pytest.approx(5.25, abs=0.3)


def test_shot_clock_dip_before_reset_to_14_is_kept():
    shots = [15, 15, 14, 14, 13, 14, 14, 13, 13, 12, 12]
    rows = [{"clock": format_clock(300 - i * 0.5), "shot": v} for i, v in enumerate(shots)]
    tl = build_timeline(make_raw(rows), NBA)
    assert [r.value for r in tl.shot_resets] == [14.0]


@pytest.mark.parametrize("bad", [[5, 5], [1], [75, 1], [2, 2, 2]])
def test_shot_clock_misreads_do_not_fake_a_reset(bad):
    shots = [18, 18, 17, 17, 16, 16] + bad + [14, 14, 13, 13, 12, 12]
    rows = [{"clock": format_clock(300 - i * 0.5), "shot": v} for i, v in enumerate(shots)]
    tl = build_timeline(make_raw(rows), NBA)
    assert tl.shot_resets == [], "a drop of ten seconds between samples cannot be real"


def test_shot_outlier_pass_adopts_a_new_level_that_persists():
    t = np.arange(14) / 2
    shot = np.array([20, 20, 19, 19, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5], dtype=float)
    _drop_shot_outliers(shot, t, (24.0, 14.0))
    assert not np.isnan(shot[10:]).any(), "four agreeing reads over 1.5 s win over the old chain"


def test_shot_clock_switching_off_marks_a_possession():
    # the other team gets the ball with 23 s left in the period: no shot clock for them
    rows = [{"clock": format_clock(26 - i * 0.5), "shot": v} for i, v in enumerate([6, 6, 5, 5, 4, 4])]
    rows += [{"clock": format_clock(23 - i * 0.5), "shot": ""} for i in range(10)]
    tl = build_timeline(make_raw(rows), NBA)
    assert len(tl.shot_resets) == 1 and np.isnan(tl.shot_resets[0].value)
    assert tl.shot_resets[0].t == pytest.approx(2.75)


def test_period_is_cleaned_and_inferred_when_missing():
    rows = running(20, 20, period="3rd") + [{"visible": False}] * 6 + running(720, 20, period="4th")
    rows[5]["period"] = "1st"  # misread
    rows[30]["period"] = ""
    tl = build_timeline(make_raw(rows), NBA)
    assert (tl.period[:20] == 3).all() and (tl.period[26:] == 4).all()

    no_period = [{**r, "period": ""} for r in rows]
    tl2 = build_timeline(make_raw(no_period), NBA)
    assert tl2.notes.get("period_inferred") and tl2.period[0] == 1 and tl2.period[-1] == 2
    assert tl2.live[26:].all(), "the clock going back up to 12:00 is a new period, not a replay"


def test_timeline_matches_the_script_and_survives_noise(coverage_script):
    g = coverage_script
    for label, raw in (("clean", ideal_samples(g)), ("noisy", add_ocr_noise(ideal_samples(g), seed=11, rate=0.03))):
        tl = build_timeline(raw, NBA)
        wrong_score = wrong_clock = 0
        for i in range(len(tl)):
            s = g.state_at(float(tl.t[i]))
            if s.scene == "live":
                assert tl.state[i] == STATE_LIVE, label
                if s.anim_team is None and (tl.score_away[i], tl.score_home[i]) != (s.away_score, s.home_score):
                    wrong_score += 1
                if not np.isnan(tl.clock[i]) and abs(tl.clock[i] - float(_clock_value(s.clock_text))) > 1.01:
                    wrong_clock += 1
            elif s.scene in ("commercial", "replay_hidden"):
                assert tl.state[i] == STATE_HIDDEN, label
        # a corrupted read at a transition can delay the new score by a sample or two
        assert wrong_score <= (0 if label == "clean" else 6), label
        assert wrong_clock == 0, label


def _clock_value(text: str) -> float:
    from possession_cut.pipeline.ocr import parse_clock

    return parse_clock(text)


def test_timeline_parquet_roundtrip(coverage_script, tmp_path):
    tl = build_timeline(ideal_samples(coverage_script), NBA)
    path = tmp_path / "timeline.parquet"
    tl.save(path)
    back = Timeline.load(path, tl.fps)
    for name in ("t", "visible", "state", "clock_running", "clock_read"):
        assert np.array_equal(getattr(tl, name), getattr(back, name))
    for name in ("period", "clock", "shot_clock", "score_away", "score_home"):
        assert np.allclose(getattr(tl, name), getattr(back, name), equal_nan=True)
    import polars as pl

    cols = pl.read_parquet(path).columns
    for required in ("t_video", "bug_visible", "period", "clock", "shot_clock", "score_home", "score_away", "confidence"):
        assert required in cols, "PRD timeline columns"
