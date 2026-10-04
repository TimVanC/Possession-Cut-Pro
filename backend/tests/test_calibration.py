"""Calibration against the synthetic broadcast, whose bug geometry is known exactly."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

from possession_cut.pipeline import calibration as calib
from possession_cut.pipeline import templates as tmpl
from possession_cut.pipeline.bugreader import VISIBLE_THRESHOLD
from possession_cut.pipeline.calib_local import size_field_boxes
from possession_cut.pipeline.calibration import calibrate, make_reader, read_frame, sample_times
from possession_cut.pipeline.frames import extract_frame
from possession_cut.pipeline.geometry import crop_from_norm, expand, iou
from possession_cut.sports import get_adapter

pytestmark = pytest.mark.video


def test_sample_times_are_spread_and_inside_the_file():
    times = sample_times(9000.0)
    assert len(times) == 12
    assert times == sorted(times) and times[0] > 300 and times[-1] < 8700


def test_local_calibration_finds_the_bug_without_adjustment(coverage_calibration, coverage_video):
    cal, _ref, _mask, _ = coverage_calibration
    _, truth = coverage_video
    assert cal.source == "local"
    assert iou(tuple(cal.bug), tuple(truth["layout"]["bug"])) >= 0.93
    assert set(cal.fields) == set(truth["layout"]["fields"]), "every NBA field located"
    assert cal.teams == {"away": "SA", "home": "NY"}
    assert cal.confidence >= 0.9 and not cal.warnings
    assert cal.checks["static_inside"] > 0.6 > cal.checks["static_outside"]


def test_crop_matches_the_prd_rule(coverage_calibration, coverage_video):
    cal, *_ = coverage_calibration
    _, truth = coverage_video
    got = crop_from_norm(cal.crop, 1280, 720)
    want = tuple(truth["layout"]["crop_px"])
    assert got[1] == 0 and got[3] == 720
    assert abs(got[0] - want[0]) <= 2 and abs(got[2] - want[2]) <= 2
    bug_w = (cal.bug[2] - cal.bug[0]) * 1280
    assert bug_w / got[2] == pytest.approx(0.78, abs=0.01)


def _states_near(script, t: float):
    """The decoded frame for time t can be the one just before or after it."""
    return [script.state_at(max(0.0, t + d)) for d in (0.0, -1 / 30, 1 / 30)]


def test_every_sampled_frame_reads_exactly_what_the_script_shows(coverage_calibration, coverage_script):
    cal, *_ = coverage_calibration
    assert len(cal.frames) == 12
    for fr in cal.frames:
        states = _states_near(coverage_script, fr["time"])
        state = states[0]
        assert fr["visible"] == state.visible
        if not state.visible:
            continue
        reads = fr["reads"]
        assert reads["clock"]["text"] in {s.clock_text for s in states}
        assert reads["period"]["value"] == state.period
        assert reads["shot_clock"]["text"] in {s.shot_text for s in states}
        if state.anim_team != "away":
            assert reads["away_score"]["value"] == state.away_score
        if state.anim_team != "home":
            assert reads["home_score"]["value"] == state.home_score


def test_bug_visibility_by_scene(coverage_calibration, coverage_probe, coverage_script, ocr_engine):
    cal, ref, mask, _ = coverage_calibration
    reader = make_reader(cal, get_adapter("nba"), 1280, 720, ocr_engine, ref, mask)
    g = coverage_script
    commercial = next(h for h in g.hidden if h[2] == "commercial")
    replay_hidden = next(h for h in g.hidden if h[2] == "replay_hidden")
    replay_bug = g.replays[0]
    anim = g.anims[0]
    cases = [
        ("live", 5.0, True),
        ("commercial with decoy numbers where the bug sits", (commercial[0] + commercial[1]) / 2, False),
        ("replay with the bug hidden", (replay_hidden[0] + replay_hidden[1]) / 2, False),
        ("replay that re-airs the bug", (replay_bug[0] + replay_bug[1]) / 2, True),
        ("score animation covering one team block", (anim[0] + anim[1]) / 2, True),
    ]
    for label, t, want in cases:
        out = read_frame(reader, extract_frame(coverage_probe, t))
        assert out["visible"] is want, f"{label}: similarity {out['similarity']} vs threshold {VISIBLE_THRESHOLD}"


def test_template_is_saved_and_reused(coverage_calibration, coverage_probe, ocr_engine, settings, tmp_path):
    from possession_cut.db import session_scope

    cal, ref, mask, _ = coverage_calibration
    adapter = get_adapter("nba")
    with session_scope() as session:
        cal_copy = calib.Calibration.from_dict(cal.to_dict())
        cal_copy.broadcaster = "ESPN"
        template = tmpl.save_template(session, cal_copy, adapter, ref, mask)
        assert template.name.startswith("ESPN NBA ")
        assert cal_copy.template_id == template.id
        second = tmpl.save_template(session, calib.Calibration.from_dict(cal.to_dict()), adapter, ref, mask, name=template.name)
        assert second.name == f"{template.name} (2)", "names stay unique"
        tmpl.delete_template(session, second)
        templates = tmpl.templates_for(session, "nba")
        assert [t.id for t in templates] == [template.id]

    # a new job on the same broadcast: matched by image similarity, no detection needed
    def boom(*_a, **_k):
        raise AssertionError("detection should not run when a template matches")

    original = calib.LocalDetector
    calib.LocalDetector = boom  # type: ignore[assignment]
    try:
        reused, ref2, _ = calibrate(coverage_probe, tmp_path / "job2", adapter, ocr_engine, templates=templates)
    finally:
        calib.LocalDetector = original
    assert reused.source == "template" and reused.template_id == template.id
    assert reused.confidence >= 0.9 and reused.bug == cal_copy.bug
    assert ref2 is not None


def test_template_does_not_match_unrelated_footage(coverage_calibration, settings):
    from possession_cut.db import session_scope

    cal, ref, mask, _ = coverage_calibration
    adapter = get_adapter("nba")
    rng = np.random.default_rng(3)
    other = [cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8), (9, 9), 0) for _ in range(6)]
    with session_scope() as session:
        template = tmpl.save_template(session, calib.Calibration.from_dict(cal.to_dict()), adapter, ref, mask)
        assert tmpl.match_template(template, other, adapter) is None
        four_by_three = [f[:, :960] for f in other]
        assert tmpl.match_template(template, four_by_three, adapter) is None, "aspect ratio must match"


class FakeClaude:
    """Stands in for Claude vision: returns the true boxes with the jitter a vision model has."""

    available = True
    unavailable_reason = None

    def __init__(self, truth: dict, seed: int = 5) -> None:
        self.truth = truth
        self.rng = random.Random(seed)
        self.calls: list[str] = []
        b = truth["layout"]["bug"]
        j = lambda: self.rng.uniform(-0.008, 0.008)  # noqa: E731
        self.bug = (b[0] + j(), b[1] + j() / 2, b[2] + j(), b[3] + j() / 2)

    @staticmethod
    def _k(v: float) -> int:
        return int(round(v * 1000))

    def json(self, *, purpose, system, content, schema, **_kw):
        self.calls.append(purpose)
        assert content[0]["type"] == "image" and content[1]["type"] == "text"
        if purpose == "calibration:locate":
            x0, y0, x1, y1 = (self._k(v) for v in self.bug)
            return {"score_bug_visible": True, "bug_box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "away_team": "SA", "home_team": "NY", "broadcaster": "ESPN"}
        assert purpose == "calibration:fields"
        region = expand(self.bug, 0.015, (self.bug[3] - self.bug[1]) * 0.35)
        rw, rh = region[2] - region[0], region[3] - region[1]
        out = {}
        for name in schema["properties"]:
            f = self.truth["layout"]["fields"][name]
            jx = lambda: self.rng.uniform(-0.012, 0.012)  # noqa: E731
            out[name] = {
                "x0": self._k((f[0] - region[0]) / rw + jx()), "y0": self._k((f[1] - region[1]) / rh + jx()),
                "x1": self._k((f[2] - region[0]) / rw + jx()), "y1": self._k((f[3] - region[1]) / rh + jx()),
            }
        return out


def test_claude_route_snaps_approximate_boxes_to_real_text(coverage_probe, coverage_video, coverage_script, ocr_engine, tmp_path):
    _, truth = coverage_video
    fake = FakeClaude(truth)
    cal, ref, mask = calibrate(coverage_probe, tmp_path / "job", get_adapter("nba"), ocr_engine, claude=fake)
    assert cal.source == "claude" and cal.broadcaster == "ESPN"
    assert fake.calls.count("calibration:locate") == 12 and fake.calls.count("calibration:fields") == 3
    assert iou(tuple(cal.bug), tuple(truth["layout"]["bug"])) >= 0.93, "snapped to the static region"
    assert cal.confidence >= 0.9, cal.warnings
    for fr in cal.frames:
        states = _states_near(coverage_script, fr["time"])
        state = states[0]
        assert fr["reads"]["clock"]["text"] in {s.clock_text for s in states}
        if state.anim_team is None:
            assert fr["reads"]["away_score"]["value"] == state.away_score
            assert fr["reads"]["home_score"]["value"] == state.home_score


def test_claude_unavailable_falls_back_to_local(coverage_probe, ocr_engine, tmp_path):
    class NoClaude:
        available = False
        unavailable_reason = "The API key is not scoped to a workspace."

    cal, *_ = calibrate(coverage_probe, tmp_path / "job", get_adapter("nba"), ocr_engine, claude=NoClaude())
    assert cal.source == "local" and cal.confidence >= 0.9
    assert any("not scoped to a workspace" in w for w in cal.warnings)


def test_no_bug_yields_a_manual_starting_point(coverage_probe, ocr_engine, tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    noise = [cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8), (15, 15), 0) for _ in range(12)]
    monkeypatch.setattr(calib, "extract_frames", lambda probe, times: noise)
    cal, ref, mask = calibrate(coverage_probe, tmp_path / "job", get_adapter("nba"), ocr_engine)
    assert cal.source == "manual" and cal.confidence == 0.0 and ref is None
    assert any("by hand" in w for w in cal.warnings)
    assert len(cal.frames) == 12 and cal.bug and cal.crop


def test_field_padding_stops_at_artwork():
    # a score "88" with a logo 6 px to its right: padding must not take in the logo
    bug_img = np.full((46, 400, 3), 30, np.uint8)
    cv2.circle(bug_img, (150, 23), 14, (0, 140, 255), -1)
    bug = (0.25, 0.80, 0.25 + 400 / 1280, 0.80 + 46 / 720)
    text = (0.25 + 80 / 1280, 0.80 + 8 / 720, 0.25 + 130 / 1280, 0.80 + 38 / 720)
    padded = size_field_boxes({"home_score": text}, bug, 1280, 720, bug_img)["home_score"]
    right_px = padded[2] * 1280 - 0.25 * 1280
    left_px = padded[0] * 1280 - 0.25 * 1280
    assert right_px <= 137, "stops before the logo edge at x=136"
    assert left_px <= 62, "but still grows on the open side for a third digit"
