"""Start and end points of a cut, including "auto: start of biggest run".

A start/end spec is one of:
    {"mode": "start"}                          start of game (or "end" for the end spec)
    {"mode": "game_time", "period": 3, "clock": 146.0}
    {"mode": "auto_run"}                       the moment the followed team trailed by the most

``auto_run`` is resolved from play-by-play when it is available (so the form can show it
before analysis) and from the bug's own score track when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sports.base import ScoreChange, ScoringEvent, SportAdapter
from .matching import MatchResult
from .timeline import Timeline


@dataclass
class RunStart:
    period: int
    clock: float | None
    deficit: int
    score_away: int
    score_home: int
    event_id: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def biggest_run_start(pbp: list[ScoringEvent], follow_side: str) -> RunStart | None:
    """The opponent score that put the followed team down by its largest deficit.

    If that deficit was reached more than once, the last time counts: the run starts
    from the final low point. None when the team never trailed.
    """
    best: ScoringEvent | None = None
    best_deficit = 0
    for ev in pbp:
        if ev.score_home is None or ev.score_away is None:
            continue
        own, opp = (ev.score_home, ev.score_away) if follow_side == "home" else (ev.score_away, ev.score_home)
        deficit = opp - own
        if deficit > 0 and deficit >= best_deficit and ev.extra.get("side", None) != follow_side:
            best, best_deficit = ev, deficit
    if best is None:
        return None
    return RunStart(
        period=best.period, clock=best.clock, deficit=best_deficit,
        score_away=best.score_away, score_home=best.score_home,
        event_id=best.event_id, description=best.description,
    )


def run_start_from_timeline(tl: Timeline, events: list[ScoreChange], follow_side: str) -> ScoreChange | None:
    """Same thing read off the bug: the opponent score change that made the deficit largest."""
    best: ScoreChange | None = None
    best_deficit = 0
    for ev in events:
        if ev.team == follow_side:
            continue
        own, opp = (ev.score_home, ev.score_away) if follow_side == "home" else (ev.score_away, ev.score_home)
        deficit = opp - own
        if deficit > 0 and deficit >= best_deficit:
            best, best_deficit = ev, deficit
    return best


def time_at_game_time(tl: Timeline, adapter: SportAdapter, period: int, clock: float | None) -> float | None:
    """Video time at which the live bug first reaches (period, clock)."""
    target = adapter.elapsed(period, clock if clock is not None else adapter.period_length(period))
    if target is None:
        return None
    for i in np.flatnonzero(tl.live & ~np.isnan(tl.clock) & ~np.isnan(tl.period)):
        elapsed = adapter.elapsed(int(tl.period[i]), float(tl.clock[i]))
        if elapsed is not None and elapsed >= target - 1e-6:
            return float(tl.t[i])
    return None


@dataclass
class Window:
    t_min: float | None = None
    t_max: float | None = None
    start_label: str = "Start of game"
    end_label: str = "End of game"
    notes: list[str] | None = None
    run_start: dict | None = None


def resolve_window(
    tl: Timeline,
    events: list[ScoreChange],
    adapter: SportAdapter,
    follow_side: str,
    start_spec: dict | None,
    end_spec: dict | None,
    pbp: list[ScoringEvent] | None = None,
    matches: MatchResult | None = None,
) -> Window:
    """Turn the job's start/end spec into video times."""
    w = Window(notes=[])
    start = start_spec or {"mode": "start"}
    end = end_spec or {"mode": "end"}

    if start.get("mode") == "game_time":
        t = time_at_game_time(tl, adapter, int(start["period"]), start.get("clock"))
        if t is None:
            w.notes.append("The start point was not found on the bug; starting from the beginning.")
        else:
            w.t_min = t
            w.start_label = f"{adapter.format_period(int(start['period']))} {_clock_text(start.get('clock'))}"
    elif start.get("mode") == "auto_run":
        anchor_t: float | None = None
        run = biggest_run_start(pbp, follow_side) if pbp else None
        if run is not None:
            w.run_start = run.to_dict()
            if matches is not None:
                for m in matches.matches.values():
                    if any(e.event_id == run.event_id for e in m.events):
                        anchor_t = m.change.t
                        break
            if anchor_t is None:
                # the play is in the play-by-play but was not seen on the bug: go by game time
                anchor_t = time_at_game_time(tl, adapter, run.period, run.clock)
        if anchor_t is None:
            change = run_start_from_timeline(tl, events, follow_side)
            if change is not None:
                anchor_t = change.t
                own, opp = (change.score_home, change.score_away) if follow_side == "home" else (change.score_away, change.score_home)
                w.run_start = {
                    "period": change.period, "clock": change.clock, "deficit": opp - own,
                    "score_away": change.score_away, "score_home": change.score_home,
                    "event_id": None, "description": "read from the score bug",
                }
                if pbp:
                    w.notes.append("Biggest-run start taken from the bug; the play-by-play play was not located.")
        if anchor_t is None:
            w.notes.append("The followed team never trailed; starting from the beginning.")
        else:
            # strictly after the opponent basket that set the low point
            w.t_min = anchor_t + tl.dt / 2
            rs = w.run_start or {}
            if rs.get("period") is not None:
                w.start_label = (
                    f"Biggest run: down {rs.get('deficit')} at "
                    f"{adapter.format_period(rs['period'])} {_clock_text(rs.get('clock'))}"
                )

    if end.get("mode") == "game_time":
        t = time_at_game_time(tl, adapter, int(end["period"]), end.get("clock"))
        if t is None:
            w.notes.append("The end point was not found on the bug; running to the end.")
        else:
            # a make just before the end point shows on the bug a beat later
            w.t_max = t + 2.5
            w.end_label = f"{adapter.format_period(int(end['period']))} {_clock_text(end.get('clock'))}"
    return w


def _clock_text(clock: float | None) -> str:
    if clock is None:
        return ""
    if clock >= 60:
        return f"{int(clock // 60)}:{int(clock % 60):02d}"
    return f"{clock:.1f}"
