"""The real thing: decode the synthetic broadcast, OCR the bug, build the cut, grade it.

These are the PRD acceptance criteria that depend on video:
every scripted score by the followed team is detected, zero false clips, clip starts
within 1.5 s of the scripted possession start, ends within 1 s of target, and replays
and commercial gaps never appear in a clip.
"""

from __future__ import annotations

import numpy as np
import pytest

from possession_cut.pipeline.clips import build_clips
from possession_cut.pipeline.sampler import RawSamples, chunk_plan, sample_bug
from possession_cut.pipeline.timeline import STATE_HIDDEN, STATE_LIVE
from possession_cut.sports import get_adapter

from .helpers import grade


def test_chunk_plan_covers_the_grid_exactly():
    chunks = chunk_plan(0.0, 384.83, 2.0, chunk_seconds=150.0)
    assert chunks == [(0.0, 150.0), (150.0, 150.0), (300.0, 85.0)]
    assert sum(int(round(d * 2)) for _, d in chunks) == 770
    assert chunk_plan(10.2, 12.0, 2.0) == [(10.5, 1.5)], "starts on the next grid point"
    assert chunk_plan(5.0, 5.0, 2.0) == []


pytestmark_video = pytest.mark.video


@pytest.mark.video
def test_sampler_reads_the_whole_file(coverage_raw, coverage_script, coverage_probe):
    raw = coverage_raw
    assert len(raw) == int(coverage_probe.duration * 2) + 1, "one sample every half second from t=0"
    assert np.allclose(np.diff(raw.t), 0.5)
    wrong = 0
    for i in range(len(raw)):
        s = coverage_script.state_at(float(raw.t[i]))
        assert bool(raw.visible[i]) == s.visible, f"visibility at {raw.t[i]}"
        if not s.visible:
            continue
        near = [coverage_script.state_at(max(0.0, float(raw.t[i]) + d)) for d in (0.0, -1 / 30, 1 / 30)]
        if raw.texts["clock"][i] not in {n.clock_text for n in near}:
            wrong += 1
        if s.anim_team is None and (raw.texts["away_score"][i], raw.texts["home_score"][i]) != (str(s.away_score), str(s.home_score)):
            if not any(n.anim_team for n in near):
                wrong += 1
    assert wrong <= 2, "OCR on the synthetic bug is essentially exact"


@pytest.mark.video
def test_sampler_is_deterministic_across_worker_counts(coverage_calibration, coverage_probe, coverage_raw):
    cal, ref, mask, _ = coverage_calibration
    seen = []
    part = sample_bug(
        coverage_probe, cal, get_adapter("nba"), ref, mask, fps=2.0, workers=2, start=60.0, end=120.0,
        progress=lambda f, m: seen.append(f),
    )
    lo, hi = 120, 240
    assert len(part) == hi - lo and part.t[0] == 60.0
    assert part.texts["clock"] == coverage_raw.texts["clock"][lo:hi]
    assert part.texts["home_score"] == coverage_raw.texts["home_score"][lo:hi]
    assert np.array_equal(part.visible, coverage_raw.visible[lo:hi])
    assert seen and seen[-1] == 1.0
    assert part.stats["cache_hits"] > part.stats["ocr_calls"], "unchanged fields are not re-read"


@pytest.mark.video
def test_raw_samples_parquet_roundtrip(coverage_raw, tmp_path):
    path = tmp_path / "raw.parquet"
    coverage_raw.save(path)
    back = RawSamples.load(path, 2.0)
    assert np.array_equal(back.t, coverage_raw.t) and np.array_equal(back.visible, coverage_raw.visible)
    assert back.texts == coverage_raw.texts


@pytest.mark.video
def test_timeline_from_video_matches_the_script(coverage_analysis, coverage_script, coverage_video):
    tl, events = coverage_analysis
    _, truth = coverage_video
    assert len(events) == len(truth["score_events"]), "every scripted score, no extras"
    for ev, tr in zip(events, truth["score_events"], strict=True):
        assert (ev.team, ev.points, ev.score_away, ev.score_home) == (tr["team"], tr["points"], tr["score_away"], tr["score_home"])
        assert 0 <= ev.t - tr["visible_time"] <= 0.55
    for i in range(len(tl)):
        s = coverage_script.state_at(float(tl.t[i]))
        if s.scene in ("commercial", "replay_hidden"):
            assert tl.state[i] == STATE_HIDDEN
        elif s.scene == "live":
            assert tl.state[i] == STATE_LIVE
    assert tl.notes["replay_samples"] >= 8, "both bug-carrying replays were caught going backward"


@pytest.mark.video
@pytest.mark.parametrize("team", ["home", "away"])
def test_cut_from_video_meets_the_acceptance_criteria(coverage_analysis, coverage_video, team):
    tl, events = coverage_analysis
    _, truth = coverage_video
    clips = build_clips(tl, events, get_adapter("nba"), team)
    g = grade(clips, truth["cutlists"][team], truth["intervals"]["not_live"])
    assert not g.missed, "every scripted score by the followed team is detected"
    assert not g.extra, "zero false clips"
    assert not g.leaks, "replays and commercial gaps never appear in a clip"
    assert not g.problems, g.problems
    assert g.start_errors and all(abs(e) <= 1.5 for e in g.start_errors), g.start_errors
    assert all(abs(e) <= 1.0 for e in g.end_errors), g.end_errors
