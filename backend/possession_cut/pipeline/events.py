"""Stage 5: score events. Each stable change in a team's score is one event."""

from __future__ import annotations

import numpy as np

from ..sports.base import ScoreChange, SportAdapter
from .timeline import Timeline

STOPPED_LOOKBACK = 3.5
LAG_REACH = 4.5  # how far before a score showing to look for the stoppage it came out of
HELD_SECONDS = 1.5


def _stoppage_before(tl: Timeline, j: int) -> tuple[float, float | None]:
    """The longest the game clock had been stopped at any sample in the LAG_REACH seconds
    up to sample j, and the value it was stopped at (None if it was running)."""
    best, value = 0.0, None
    k = j
    while k >= 0 and tl.t[j] - tl.t[k] <= LAG_REACH:
        if tl.live[k] and not np.isnan(tl.clock[k]):
            held = tl.stopped_duration(k)
            if held > best:
                best, value = held, float(tl.clock[k])
        k -= 1
    return best, (value if best >= HELD_SECONDS else None)


def detect_score_events(tl: Timeline, adapter: SportAdapter) -> list[ScoreChange]:
    """Every score increase for both teams, with the moment the new score first appeared."""
    events: list[ScoreChange] = []
    n = len(tl)
    for side in ("away", "home"):
        other = "home" if side == "away" else "away"
        score, read = tl.score(side), tl.score_read(side)
        other_score = tl.score(other)
        prev = np.nan
        last_read = -1
        for i in range(n):
            v = score[i]
            if np.isnan(v):
                continue
            if not np.isnan(prev) and v > prev:
                j = last_read if last_read >= 0 else max(0, i - 1)
                clock = tl.value_near(tl.clock, i, 3)
                clock_before = tl.value_near(tl.clock, j, 3)
                stopped = (
                    tl.stopped_since(i, STOPPED_LOOKBACK)
                    and not np.isnan(clock)
                    and not np.isnan(clock_before)
                    and abs(clock - clock_before) < 0.05
                )
                stopped_before, clock_held = _stoppage_before(tl, j)
                notes: list[str] = []
                points = int(v - prev)
                hidden_between = int((~tl.live[j + 1 : i]).sum())
                if hidden_between >= 2:
                    notes.append(f"score appeared after {tl.t[i] - tl.t[j]:.0f}s without a readable live bug")
                if points > adapter.max_points_per_event and not stopped:
                    notes.append(f"score jumped by {points}; more than one play may be merged")
                o = other_score[i]
                conf = float(tl.confidence[i : min(n, i + 2)].min()) if i < n else 0.0
                p = tl.period[i]
                events.append(
                    ScoreChange(
                        index=0,
                        team=side,
                        points=points,
                        t=float(tl.t[i]),
                        t_prev=float(tl.t[j]),
                        period=None if np.isnan(p) else int(p),
                        clock=None if np.isnan(clock) else float(clock),
                        clock_before=None if np.isnan(clock_before) else float(clock_before),
                        score_before=int(prev),
                        score_after=int(v),
                        score_away=int(v if side == "away" else (0 if np.isnan(o) else o)),
                        score_home=int(v if side == "home" else (0 if np.isnan(o) else o)),
                        clock_stopped=bool(stopped),
                        stopped_for=tl.stopped_duration(i),
                        stopped_before=stopped_before,
                        clock_held=clock_held,
                        confidence=conf,
                        notes=notes,
                    )
                )
            if read[i]:
                last_read = i
            prev = v
    events.sort(key=lambda e: (e.t, e.team))
    for k, e in enumerate(events):
        e.index = k
    return events
