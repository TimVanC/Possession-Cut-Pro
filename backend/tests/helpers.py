"""Test helpers: simulated bug reads from a scripted game, and grading against ground truth."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from possession_cut.pipeline.clips import ClipDraft, build_clips
from possession_cut.pipeline.events import detect_score_events
from possession_cut.pipeline.sampler import RawSamples
from possession_cut.pipeline.timeline import Timeline, build_timeline
from possession_cut.sports import get_adapter
from possession_cut.synth.script import GameScript

FIELDS = ("away_label", "away_score", "home_label", "home_score", "period", "clock", "shot_clock")


def ideal_samples(script: GameScript, fps: float = 2.0, phase: float = 0.0) -> RawSamples:
    """What a perfect OCR pass would read off the bug, sampled at ``fps``."""
    n = int((script.duration - phase) * fps)
    t = phase + np.arange(n) / fps
    raw = RawSamples(
        fps=fps, t=t, visible=np.zeros(n, dtype=bool), similarity=np.zeros(n, dtype=np.float32),
        texts={f: [""] * n for f in FIELDS}, confs={f: np.zeros(n, dtype=np.float32) for f in FIELDS},
    )
    for i, ti in enumerate(t):
        s = script.state_at(float(ti))
        raw.visible[i] = s.visible
        raw.similarity[i] = 1.0 if s.visible else 0.2
        if not s.visible:
            continue
        values = {
            "away_label": script.teams["away"]["abbr"],
            "home_label": script.teams["home"]["abbr"],
            "away_score": "" if s.anim_team == "away" else str(s.away_score),
            "home_score": "" if s.anim_team == "home" else str(s.home_score),
            "period": s.period_text,
            "clock": s.clock_text,
            "shot_clock": s.shot_text,
        }
        for name, text in values.items():
            raw.texts[name][i] = text
            raw.confs[name][i] = 1.0 if text else 0.0
    return raw


def add_ocr_noise(raw: RawSamples, seed: int, rate: float = 0.02) -> RawSamples:
    """Corrupt a share of reads the way OCR does: wrong digit, dropped character, nothing."""
    rng = random.Random(seed)
    swaps = {"0": "8", "8": "0", "1": "7", "7": "1", "3": "8", "5": "6", "6": "5", "9": "0", "2": "7", "4": "1"}
    for name in ("away_score", "home_score", "clock", "shot_clock", "period"):
        for i, text in enumerate(raw.texts[name]):
            if not text or rng.random() > rate:
                continue
            roll = rng.random()
            if roll < 0.4:
                k = rng.randrange(len(text))
                raw.texts[name][i] = text[:k] + swaps.get(text[k], text[k]) + text[k + 1 :]
            elif roll < 0.7 and len(text) > 1:
                k = rng.randrange(len(text))
                raw.texts[name][i] = text[:k] + text[k + 1 :]
            else:
                raw.texts[name][i] = ""
                raw.confs[name][i] = 0.0
    return raw


def run_logic(script: GameScript, team: str = "home", fps: float = 2.0, raw: RawSamples | None = None, **options):
    """Timeline -> events -> clips for a scripted game, with no video involved."""
    adapter = get_adapter("nba")
    raw = raw if raw is not None else ideal_samples(script, fps)
    tl = build_timeline(raw, adapter)
    events = detect_score_events(tl, adapter)
    clips = build_clips(tl, events, adapter, team, options)
    return tl, events, clips


@dataclass
class Grade:
    matched: int = 0
    missed: list[dict] = field(default_factory=list)
    extra: list[ClipDraft] = field(default_factory=list)
    start_errors: list[float] = field(default_factory=list)  # clip start minus scripted possession start
    end_errors: list[float] = field(default_factory=list)  # clip end minus target end
    leaks: list[tuple[float, float]] = field(default_factory=list)  # clip time that overlaps not-live footage
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missed or self.extra or self.leaks or self.problems)


def grade(clips: list[ClipDraft], truth_clips: list[dict], not_live: list, start_tol: float = 1.5, end_tol: float = 1.0) -> Grade:
    """Acceptance criteria from the PRD, applied clip by clip.

    Every scripted score detected, no false clips, starts within 1.5 s of the scripted
    possession start, ends within 1 s of target, nothing not-live inside a clip.

    Where the bug gives no signal for a possession change (``start_observable`` False in
    the truth), the start cannot be located by any bug-reading rule. Those clips must still
    exist, end on target and contain the whole possession; they may start early and merge
    into the previous clip of the same team.
    """
    g = Grade()
    covered: set[int] = set()
    for c in clips:
        mine = [
            (k, tc) for k, tc in enumerate(truth_clips)
            if tc["team"] == c.team and c.score_before < tc["score_after"] <= c.score_after
        ]
        if not mine:
            g.extra.append(c)
            continue
        covered.update(k for k, _ in mine)
        g.matched += len(mine)
        first, last = mine[0][1], mine[-1][1]
        label = f"clip to {last['score_after']}"
        want_points = sum(tc["points"] for _, tc in mine)
        if c.points != want_points or c.score_before != first["score_before"]:
            g.problems.append(f"{label}: points {c.points} vs {want_points}")
        for _, tc in mine[1:]:
            if tc.get("start_observable", True):
                g.problems.append(f"{label}: merged with the clip to {first['score_after']} although its start was visible")
        end_err = c.src_out - last["src_out"]
        g.end_errors.append(end_err)
        if abs(end_err) > end_tol:
            g.problems.append(f"{label}: ends {end_err:+.2f}s from target")
        if first["kind"] == "field_goal":
            start_err = c.src_in - first["possession_start"]
            if first.get("start_observable", True):
                g.start_errors.append(start_err)
                # a target start pushed later by a cut-out break is graded against the target
                pushed = abs(first["src_in"] - (first["possession_start"] - 1.0)) > 0.01
                off = c.src_in - first["src_in"] if pushed else start_err
                if abs(off) > start_tol:
                    g.problems.append(f"{label} ({first['start_cause']}): starts {start_err:+.2f}s from possession start")
            elif start_err > start_tol:
                g.problems.append(f"{label}: starts {start_err:+.2f}s into a possession")
        else:
            start_err = c.src_in - first["src_in"]
            if abs(start_err) > end_tol:
                g.problems.append(f"free throws to {last['score_after']}: start {start_err:+.2f}s from target")
        if len(mine) == 1 and first.get("start_observable", True):
            if c.kind != first["kind"]:
                g.problems.append(f"{label}: kind {c.kind} vs {first['kind']}")
            if len(c.segments) != len(first["segments"]):
                g.problems.append(f"{label}: {len(c.segments)} segments vs {len(first['segments'])}")
    g.missed = [tc for k, tc in enumerate(truth_clips) if k not in covered]
    for c in clips:
        for a, b in c.segments:
            for h0, h1 in not_live:
                lo, hi = max(a, h0), min(b, h1)
                if hi - lo > 1e-3:
                    g.leaks.append((round(lo, 3), round(hi, 3)))
    return g


def timeline_summary(tl: Timeline) -> str:
    return f"{len(tl)} samples, live {int(tl.live.sum())}, notes {tl.notes}"
