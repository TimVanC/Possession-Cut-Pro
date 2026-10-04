"""Stage 7: cross-check detected score events against league play-by-play.

Play-by-play labels and cross-checks; it never moves a cut. A detected event is matched
by period + clock within 3 s + team + points (the PRD rule). The running score is used
to make that unambiguous: a team reaches each score exactly once in a game, so "NYK went
to 78 on a 3" pins the play even when two threes fall seconds apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..sports.base import ScoreChange, ScoringEvent

CLOCK_TOLERANCE = 3.0
# A score that shows up late (after a replay or break) still belongs to its play.
LATE_SCORE_TOLERANCE = 30.0


@dataclass
class Match:
    change: ScoreChange
    events: list[ScoringEvent]
    how: str  # "clock" (full agreement) | "score" (running score only) | "merged" (several plays)
    clock_diff: float | None = None


@dataclass
class MatchResult:
    matches: dict[int, Match] = field(default_factory=dict)  # by ScoreChange.index
    unmatched_changes: list[ScoreChange] = field(default_factory=list)
    unmatched_pbp: list[ScoringEvent] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "matched": len(self.matches),
            "by_clock": sum(1 for m in self.matches.values() if m.how == "clock"),
            "unmatched_detected": len(self.unmatched_changes),
            "unmatched_pbp": len(self.unmatched_pbp),
        }


def _own_after(ev: ScoringEvent, side: str) -> int | None:
    return ev.score_home if side == "home" else ev.score_away


def sides_of(pbp: list[ScoringEvent]) -> dict[str, str]:
    """{'away': 'SAS', 'home': 'NYK'} from play-by-play. Falls back to score movement."""
    out: dict[str, str] = {}
    prev_home = prev_away = 0
    for ev in pbp:
        side = ev.extra.get("side") if ev.extra else None
        if side is None and ev.score_home is not None and ev.score_away is not None:
            side = "home" if ev.score_home > prev_home else ("away" if ev.score_away > prev_away else None)
        if ev.score_home is not None:
            prev_home, prev_away = ev.score_home, ev.score_away or 0
        if side and ev.team and side not in out:
            out[side] = ev.team
        if len(out) == 2:
            break
    return out


def match_events(
    changes: list[ScoreChange],
    pbp: list[ScoringEvent],
    sides: dict[str, str] | None = None,
    clock_tolerance: float = CLOCK_TOLERANCE,
) -> MatchResult:
    """Pair every detected score change with its play-by-play event(s).

    Three passes, strictest first, so a loose match can never take a play that a strict
    match would have claimed:

    1. running score, period and clock (within 3 s) all agree;
    2. the PRD rule on its own: period + clock within 3 s + team + points (covers a
       misread score on the bug);
    3. running score agrees but the clock does not (the score showed up late, after a
       replay or a break). Same period, within 30 s.
    """
    result = MatchResult()
    sides = sides or sides_of(pbp)
    used: set[str] = set()

    def chain_for(ch: ScoreChange, by_score: dict) -> list[ScoringEvent] | None:
        chain = [by_score[s] for s in range(ch.score_before + 1, ch.score_after + 1) if s in by_score]
        if not chain or sum(e.points for e in chain) != ch.points or any(e.event_id in used for e in chain):
            return None
        return chain

    def clock_gap(ch: ScoreChange, ev: ScoringEvent) -> float | None:
        if ch.clock is None or ev.clock is None:
            return None
        return abs(ch.clock - ev.clock)

    def same_period(ch: ScoreChange, ev: ScoringEvent) -> bool:
        return ch.period is None or ch.period == ev.period

    for side in ("away", "home"):
        abbr = (sides.get(side) or "").upper()
        team_pbp = [e for e in pbp if e.team.upper() == abbr] if abbr else []
        by_score = {_own_after(e, side): e for e in team_pbp if _own_after(e, side) is not None}
        mine = [c for c in changes if c.team == side]

        for ch in mine:
            chain = chain_for(ch, by_score)
            if chain is None:
                continue
            gap = clock_gap(ch, chain[-1])
            if gap is not None and gap <= clock_tolerance and same_period(ch, chain[-1]):
                result.matches[ch.index] = Match(ch, chain, "merged" if len(chain) > 1 else "clock", gap)
                used.update(e.event_id for e in chain)

        for ch in (c for c in mine if c.index not in result.matches):
            best, best_gap = None, clock_tolerance + 1e-6
            for e in team_pbp:
                if e.event_id in used or e.points != ch.points or not same_period(ch, e):
                    continue
                gap = clock_gap(ch, e)
                if gap is not None and gap <= best_gap:
                    best, best_gap = e, gap
            if best is not None:
                result.matches[ch.index] = Match(ch, [best], "clock", best_gap)
                used.add(best.event_id)

        for ch in (c for c in mine if c.index not in result.matches):
            chain = chain_for(ch, by_score)
            if chain is None or not same_period(ch, chain[-1]):
                continue
            gap = clock_gap(ch, chain[-1])
            if gap is None or gap <= LATE_SCORE_TOLERANCE:
                result.matches[ch.index] = Match(ch, chain, "merged" if len(chain) > 1 else "score", gap)
                used.update(e.event_id for e in chain)

    result.unmatched_changes = [c for c in changes if c.index not in result.matches]
    result.unmatched_pbp = [e for e in pbp if e.event_id not in used]
    return result


def label_clips(clips: list, result: MatchResult | None) -> None:
    """Put scorer and description on each clip and flag what play-by-play disagrees with."""
    if result is None:
        return
    for clip in clips:
        found = [result.matches.get(ch.index) for ch in clip.changes]
        events = [e for m in found if m for e in m.events]
        if events:
            scorers: list[str] = []
            for e in events:
                if e.scorer and e.scorer not in scorers:
                    scorers.append(e.scorer)
            clip.scorer = " / ".join(scorers)
            clip.description = " + ".join(e.description for e in events if e.description)
            clip.pbp_event_id = events[0].event_id
        if any(m is None for m in found):
            clip.warnings.append("no play-by-play match for this score")
            clip.confidence = round(clip.confidence * 0.75, 3)
        else:
            off = [m for m in found if m and m.how == "score" and m.clock_diff is not None]
            if off:
                worst = max(m.clock_diff for m in off)
                clip.warnings.append(f"bug clock differs from play-by-play by {worst:.0f}s")
                clip.confidence = round(clip.confidence * 0.9, 3)


# -- local play-by-play file ------------------------------------------------------


def sidecar_path(source: str | Path) -> Path:
    return Path(str(source) + ".pbp.json")


def load_sidecar(source: str | Path) -> tuple[list[ScoringEvent], dict[str, str]] | None:
    """Play-by-play from ``<video>.pbp.json`` if it exists.

    Lets a game be labelled with no network (very old games, private recordings) and is
    how the synthetic broadcast carries its scripted play-by-play.
    """
    path = sidecar_path(source)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        events = [ScoringEvent.from_dict(e) for e in data.get("events", [])]
    except (OSError, ValueError, TypeError):
        return None
    sides = {k: data[k] for k in ("away", "home") if data.get(k)}
    # sidecars need not carry the side on each event: derive it
    for ev in events:
        if not ev.extra.get("side"):
            for side, abbr in sides.items():
                if ev.team.upper() == abbr.upper():
                    ev.extra["side"] = side
    return events, sides or sides_of(events)
