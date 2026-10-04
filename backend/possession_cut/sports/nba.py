"""NBA adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import FieldSpec, Game, PlayByPlayUnavailable, ScoreChange, ScoringEvent, SportAdapter

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.timeline import Timeline


class NBAAdapter(SportAdapter):
    key = "nba"
    name = "NBA"
    bug_fields = (
        FieldSpec("away_label", "text", False, "the away (first-listed) team's abbreviation or name"),
        FieldSpec("away_score", "score", True, "the away team's score (number only, room for 3 digits)"),
        FieldSpec("home_label", "text", False, "the home (second-listed) team's abbreviation or name"),
        FieldSpec("home_score", "score", True, "the home team's score (number only, room for 3 digits)"),
        FieldSpec("period", "period", True, "the period indicator, e.g. 1st, 2nd, 3rd, 4th, OT, Q4"),
        FieldSpec("clock", "clock", True, "the game clock, e.g. 10:42 or 38.4"),
        FieldSpec("shot_clock", "shot_clock", False, "the shot clock, a small 1-2 digit number (24 down to 0)"),
    )
    default_rolls = (1.0, 1.5)
    min_clip_seconds = 3.0
    max_clip_seconds = 30.0
    regulation_periods = 4
    period_seconds = 720.0
    overtime_seconds = 300.0
    max_points_per_event = 3
    period_label = "Q"
    shot_clock_max = 24.0
    shot_clock_resets = (24.0, 14.0)
    # The bug shows a make 0.5-2 s after it happens, and the shot clock resets on the make.
    # Signals this close to the score appearing belong to the make, not to the possession.
    make_lag_guard = 2.5

    def find_games(self, date: str, team: str | None = None) -> list[Game]:
        raise PlayByPlayUnavailable("NBA game lookup is not wired up yet")

    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        raise PlayByPlayUnavailable("NBA play-by-play is not wired up yet")

    def classify(self, change: ScoreChange) -> str:
        # +1 is always a free throw. A bigger jump with the clock stopped throughout is a
        # trip to the line whose first make was not seen (the bug was hidden between shots).
        if change.points == 1 or change.clock_stopped:
            return "free_throws"
        return "field_goal"

    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        """Latest of: shot clock reset to 24/14, game clock starting after a stoppage,
        or the opponent scoring. Falls back to the maximum clip length."""
        limit = change.t - self.make_lag_guard
        candidates: list[tuple[float, str]] = []
        for reset in reversed(timeline.shot_resets):
            if reset.t <= limit and reset.t_start <= change.t - 1.0:
                candidates.append((reset.t_start, "shot_clock"))
                break
        for start in reversed(timeline.clock_starts):
            if start.t <= limit:
                candidates.append((start.t, "clock_start"))
                break
        opponent = "home" if change.team == "away" else "away"
        for t_opp in reversed(timeline.score_change_times(opponent)):
            if t_opp <= limit:
                # By the time the bug shows the opponent's basket it is already through the
                # net, so the clip starts there with no pre-roll (it would show their make).
                candidates.append((t_opp + self.default_rolls[0], "opponent_score"))
                break
        for t_own in reversed(timeline.score_change_times(change.team)):
            if t_own <= limit:
                # A possession cannot begin before the team's own previous basket. When this
                # is the best signal the two clips overlap and merge into one.
                candidates.append((t_own + self.default_rolls[0], "own_score"))
                break
        if not candidates:
            return change.t - self.max_clip_seconds, "max_length"
        t_start, cause = max(candidates)
        if change.t - t_start > self.max_clip_seconds + 10:
            return change.t - self.max_clip_seconds, "max_length"
        return t_start, cause
