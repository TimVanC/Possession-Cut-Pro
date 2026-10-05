"""Stage 6: clip boundaries.

Sport-agnostic assembly around sport-specific rules: the adapter says where a scoring
play starts and ends and what kind of play it is; this module applies the guards that
hold for every sport (length limits, cutting out not-live stretches, merging overlaps).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from ..sports.base import ScoreChange, SportAdapter
from . import intervals as iv
from .timeline import Timeline

MIN_SEGMENT = 0.4
SAME_STOPPAGE_SECONDS = 75.0
STOPPED_BEFORE_FT = 3.0  # the clock has been stopped at least this long before a free throw scores
AND_ONE_SECONDS = 120.0

DEFAULT_OPTIONS = {
    "include_free_throws": True,
    "include_and_one_ft": True,
    "include_opponent": False,
    # trim crowd shots and close-ups off clip edges (pipeline/camera.py; needs the video)
    "trim_cutaways": True,
}


@dataclass
class ClipDraft:
    team: str  # "away" | "home"
    kind: str
    points: int
    period: int | None
    clock: float | None
    score_before: int
    score_after: int
    score_away: int
    score_home: int
    segments: list[list[float]]
    changes: list[ScoreChange] = field(default_factory=list)
    start_cause: str = ""
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    scorer: str = ""
    description: str = ""
    pbp_event_id: str | None = None

    @property
    def src_in(self) -> float:
        return self.segments[0][0]

    @property
    def src_out(self) -> float:
        return self.segments[-1][1]

    @property
    def duration(self) -> float:
        return iv.total(self.segments)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["src_in"], d["src_out"], d["duration"] = self.src_in, self.src_out, round(self.duration, 3)
        return d


def _same_stoppage(a: ScoreChange, b: ScoreChange, tol: float = 0.05) -> bool:
    """Two scores during one dead ball: same period, and both came out of a stoppage at
    the same clock value. (A slow bug can show the last free throw after play has resumed,
    so the clock at the score itself is no guide.)"""
    clock_a = a.clock_held if a.clock_held is not None else a.clock
    clock_b = b.clock_held if b.clock_held is not None else b.clock
    if a.period != b.period or clock_a is None or clock_b is None:
        return False
    return abs(clock_a - clock_b) <= tol and 0 <= b.t - a.t <= SAME_STOPPAGE_SECONDS


def _gap_before(tl: Timeline, change: ScoreChange) -> tuple[float, float] | None:
    """The not-live stretch the score appeared straight after, if there was one."""
    for a, b in tl.not_live_intervals:
        if a >= change.t_prev - tl.dt and b <= change.t + tl.dt and b - a >= 2 * tl.dt:
            return a, b
    return None


def build_clips(
    tl: Timeline,
    events: list[ScoreChange],
    adapter: SportAdapter,
    follow: str,
    options: dict | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
) -> list[ClipDraft]:
    """Clips for the followed side (and the opponent if asked) between two video times."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    sides = {follow} | ({"home" if follow == "away" else "away"} if opts["include_opponent"] else set())
    pre_roll, _post_roll = adapter.default_rolls
    min_len, max_len = adapter.min_clip_seconds, adapter.max_clip_seconds
    duration = float(tl.t[-1] + tl.dt) if len(tl) else 0.0

    drafts: list[ClipDraft] = []
    clip_of: dict[int, ClipDraft] = {}  # event index -> the clip it ended up in
    kinds = {e.index: adapter.classify(e) for e in events}

    def selected(e: ScoreChange) -> bool:
        return e.team in sides and (t_min is None or e.t >= t_min) and (t_max is None or e.t <= t_max)

    def window(e: ScoreChange) -> list[float]:
        before, after = adapter.score_window(tl, e)
        return [e.t - before, e.t + after]

    for pos, e in enumerate(events):
        if not selected(e):
            continue
        kind = kinds[e.index]
        warnings = list(e.notes)

        # a follow-up the sport hangs on the previous score's clip (football's extra point)
        prev_any = events[pos - 1] if pos > 0 else None
        if prev_any is not None and prev_any.team == e.team and adapter.tail_of(prev_any, e):
            parent_clip = clip_of.get(prev_any.index)
            if parent_clip is not None and opts.get("include_tail", True):
                parent_clip.segments.append(window(e))
                parent_clip.changes.append(e)
                clip_of[e.index] = parent_clip
            continue

        if kind == "free_throws":
            seg = window(e)
            prev = events[pos - 1] if pos > 0 else None
            parent = None
            if prev is not None and prev.team == e.team and e.t - prev.t <= AND_ONE_SECONDS:
                if kinds[prev.index] != "free_throws":
                    # And-one: a basket, then a free throw with the clock where the whistle
                    # left it. The basket's score can show a beat before the clock stops,
                    # so allow a second of slack there.
                    if (e.clock_stopped or e.stopped_before >= STOPPED_BEFORE_FT) and _same_stoppage(prev, e, tol=1.05):
                        parent = clip_of.get(prev.index) if prev.index in clip_of else False
                elif _same_stoppage(prev, e) and prev.index in clip_of and clip_of[prev.index].kind != "free_throws":
                    parent = clip_of[prev.index]  # second free throw riding on a basket clip
            if parent is False:
                continue  # and-one whose basket is outside the selected range
            if parent is not None:
                if opts["include_and_one_ft"]:
                    parent.segments.append(seg)
                    parent.changes.append(e)
                    clip_of[e.index] = parent
                continue
            if not opts["include_free_throws"]:
                continue
            last = drafts[-1] if drafts else None
            if last is not None and last.kind == "free_throws" and last.team == e.team and _same_stoppage(last.changes[-1], e):
                last.segments.append(seg)
                last.changes.append(e)
                last.warnings.extend(w for w in warnings if w not in last.warnings)
                clip_of[e.index] = last
                continue
            if e.points > 1:
                warnings.append(f"{e.points} free throws read as one score change; only the last is clipped")
            draft = ClipDraft(
                team=e.team, kind="free_throws", points=0, period=e.period, clock=e.clock,
                score_before=e.score_before, score_after=e.score_after, score_away=e.score_away,
                score_home=e.score_home, segments=[seg], changes=[e], start_cause="free_throw", warnings=warnings,
            )
            drafts.append(draft)
            clip_of[e.index] = draft
            continue

        # a scoring play: possession start -> just after the score shows
        anchor = e
        gap = _gap_before(tl, e)
        if gap is not None:
            # The make happened before the broadcast cut away; the clip ends at the cutaway.
            anchor = replace(e, t=gap[0], extra={**e.extra, "anchored": True})
            end = gap[0]
            warnings.append("score appeared after a break; clip ends where the broadcast cut away")
        else:
            end = adapter.clip_end(tl, e, opts)
        start_t, cause = adapter.possession_start(tl, anchor)
        start = start_t - pre_roll
        if start > end - 1.0:
            start, cause = end - min_len, "min_length"
        if end - start > max_len:
            start = end - max_len
            warnings.append(f"possession longer than {max_len:.0f}s; trimmed to the last {max_len:.0f}s")
            if cause != "max_length":
                cause += "+trimmed"
        if end - start < min_len:
            start = end - min_len
        draft = ClipDraft(
            team=e.team, kind=kind, points=0, period=e.period, clock=e.clock,
            score_before=e.score_before, score_after=e.score_after, score_away=e.score_away,
            score_home=e.score_home, segments=[[start, end]], changes=[e], start_cause=cause, warnings=warnings,
        )
        drafts.append(draft)
        clip_of[e.index] = draft

    # guards that apply to every clip
    not_live = tl.not_live_intervals
    kept: list[ClipDraft] = []
    for d in drafts:
        segs = iv.merge([(a, b) for a, b in d.segments])
        segs = iv.subtract(segs, not_live)
        segs = [s for s in iv.clamp(segs, 0.0, duration) if s[1] - s[0] >= MIN_SEGMENT]
        if not segs:
            continue
        d.segments = [[round(a, 3), round(b, 3)] for a, b in segs]
        kept.append(d)

    kept.sort(key=lambda d: d.src_in)
    merged: list[ClipDraft] = []
    for d in kept:
        if merged and d.src_in <= merged[-1].src_out and d.team == merged[-1].team:
            m = merged[-1]
            m.changes.extend(d.changes)
            m.segments = [[a, b] for a, b in iv.merge([tuple(s) for s in m.segments + d.segments])]
            m.warnings.extend(w for w in d.warnings if w not in m.warnings)
            if d.kind != "free_throws":
                m.kind = d.kind
        elif merged and d.src_in <= merged[-1].src_out:
            # overlapping clips of different teams: keep both, cut at the boundary
            d.segments[0][0] = merged[-1].src_out
            if d.segments[0][1] - d.segments[0][0] >= MIN_SEGMENT:
                merged.append(d)
        else:
            merged.append(d)

    for d in merged:
        d.changes.sort(key=lambda c: c.t)
        first, last = d.changes[0], d.changes[-1]
        d.points = sum(c.points for c in d.changes)
        d.period, d.clock = first.period, first.clock
        d.score_before, d.score_after = first.score_before, last.score_after
        d.score_away, d.score_home = last.score_away, last.score_home
        conf = min(c.confidence for c in d.changes)
        conf = 0.5 + 0.5 * min(1.0, max(0.0, conf))
        if "max_length" in d.start_cause or "trimmed" in d.start_cause:
            conf *= 0.8
        if d.warnings:
            conf *= 0.8
        d.confidence = round(conf, 3)
    return merged
