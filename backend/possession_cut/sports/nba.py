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

    def find_games(self, date: str, team: str | None = None) -> list[Game]:
        raise PlayByPlayUnavailable("NBA game lookup is not wired up yet")

    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        raise PlayByPlayUnavailable("NBA play-by-play is not wired up yet")

    def classify(self, change: ScoreChange) -> str:
        return "free_throws" if change.points == 1 else "field_goal"

    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        raise NotImplementedError
