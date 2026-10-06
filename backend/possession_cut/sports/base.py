"""The sport adapter interface. The pipeline core only ever talks to this.

One module per sport implements it: where play-by-play comes from, which score bug
fields to read, how a clip's start is found, and the sport's timing constants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.timeline import Timeline


@dataclass(frozen=True)
class FieldSpec:
    """One sub-region of the score bug."""

    name: str
    charset: str  # key into pipeline.ocr.CHARSETS
    required: bool = True
    description: str = ""


@dataclass
class Game:
    game_id: str
    date: str  # YYYY-MM-DD
    away: str  # abbreviation
    home: str
    away_name: str = ""
    home_name: str = ""
    away_score: int | None = None
    home_score: int | None = None
    status: str = ""
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoringEvent:
    """A scoring play from play-by-play, normalized across sports."""

    event_id: str
    period: int
    clock: float | None  # seconds remaining in the period; None where the sport has no clock
    team: str  # abbreviation of the scoring team
    points: int
    scorer: str = ""
    description: str = ""
    score_away: int | None = None
    score_home: int | None = None
    kind: str = "score"  # field_goal, free_throw, touchdown, field_goal_kick, goal, run, ...
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ScoringEvent:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


def same_team_code(a: str, b: str) -> bool:
    """'NY' ~ 'NYK', 'SA' ~ 'SAS', 'GS' ~ 'GSW'; 'KNICKS' ~ 'NYK' is not attempted."""
    a, b = a.strip().upper(), b.strip().upper()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a) or (len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2])


@dataclass
class ScoreChange:
    """A score change read off the bug (pipeline output, sport-agnostic)."""

    index: int
    team: str  # "away" | "home"
    points: int
    t: float  # video time the new score first appeared
    t_prev: float  # last sample that still showed the old score
    period: int | None
    clock: float | None  # bug clock when the new score appeared
    clock_before: float | None  # bug clock at the last sample with the old score
    score_before: int
    score_after: int
    score_away: int
    score_home: int
    clock_stopped: bool = False
    # seconds the game clock had already shown its current value when the score appeared
    stopped_for: float = 0.0
    # The stoppage the score came out of: the longest the clock had been stopped in the
    # few seconds before the score showed, and the value it was stopped at. A slow bug
    # shows a free throw after play has resumed, so the clock at the score is no guide.
    stopped_before: float = 0.0
    clock_held: float | None = None
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)
    # Matched play-by-play events as dicts under "pbp" (set before clip building), plus
    # anything else an adapter wants to carry between its own hooks.
    extra: dict[str, Any] = field(default_factory=dict)


class PlayByPlayUnavailable(RuntimeError):
    """The league source could not provide data (old game, API down, no network)."""


class SportAdapter(ABC):
    key: str = ""
    name: str = ""
    # Sub-ROIs calibration must locate, in the order they are usually laid out.
    bug_fields: tuple[FieldSpec, ...] = ()
    # (pre_roll, post_roll) seconds around a scoring clip.
    default_rolls: tuple[float, float] = (1.0, 1.5)
    min_clip_seconds: float = 3.0
    max_clip_seconds: float = 30.0
    regulation_periods: int = 4
    period_seconds: float = 720.0
    overtime_seconds: float = 300.0
    has_clock: bool = True
    # Largest single scoring play, used to sanity-check score jumps.
    max_points_per_event: int = 3
    period_label: str = "Q"
    # Shot/play clock: its largest value and the values it is reset to. Empty = any.
    shot_clock_max: float = 24.0
    shot_clock_resets: tuple[float, ...] = ()
    # How far the bug clock (when the score shows) may sit from the play-by-play clock and
    # still count as full agreement. Sports whose play-by-play stamps the start of the play
    # rather than the score need more.
    pbp_clock_tolerance: float = 3.0
    # Sport-specific toggles: {"key", "label", "hint", "default", "where": "setup" | "export"}
    options: tuple[dict[str, Any], ...] = ()

    # -- league data -----------------------------------------------------
    @abstractmethod
    def find_games(self, date: str, team: str | None = None) -> list[Game]:
        """Games on ``date`` (YYYY-MM-DD), optionally filtered to one team."""

    @abstractmethod
    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        """Normalized scoring events for one game, in game order."""

    def fetch_defense(self, game_id: str) -> list[ScoringEvent]:
        """Blocks and steals from play-by-play, for the "defensive plays" option: ``kind``
        "block" or "steal", ``points`` 0, ``team`` the defending side, ``scorer`` the
        player. Sports whose records have no such plays return nothing."""
        return []

    def teams(self) -> list[dict[str, str]]:
        """Known teams: [{'abbr': 'NYK', 'name': 'New York Knicks'}, ...]."""
        return []

    # -- timing ------------------------------------------------------------
    def period_length(self, period: int) -> float:
        return self.period_seconds if period <= self.regulation_periods else self.overtime_seconds

    def elapsed(self, period: int | None, clock: float | None) -> float | None:
        """Game seconds played at (period, clock). Monotonic over a game; None if unknown."""
        if period is None or clock is None:
            return None
        done = sum(self.period_length(p) for p in range(1, period))
        return done + max(0.0, self.period_length(period) - clock)

    def format_period(self, period: int | None) -> str:
        if period is None:
            return "?"
        if period <= self.regulation_periods:
            return f"{self.period_label}{period}"
        n = period - self.regulation_periods
        return "OT" if n == 1 else f"{n}OT"

    # -- reading the bug -------------------------------------------------------
    def parse_field(self, name: str, text: str):
        """Typed value of one bug field read (None when the text is not valid for it).

        "period" must come back as a number that never decreases over a game (baseball
        encodes inning and half into one). Fields the timeline has no column for are kept
        as text in ``Timeline.extra``.
        """
        from ..pipeline.calibration import parse_field

        return parse_field(name, text)

    # -- clip rules ----------------------------------------------------------
    def classify(self, change: ScoreChange) -> str:
        """Clip kind for a score change (field_goal, free_throws, touchdown, goal, run...)."""
        return "score"

    @abstractmethod
    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        """Video time the scoring possession/play began, and which signal decided it."""

    def clip_end(self, timeline: Timeline, change: ScoreChange, options: dict) -> float:
        return change.t + self.default_rolls[1]

    def score_window(self, timeline: Timeline, change: ScoreChange) -> tuple[float, float]:
        """Seconds to show before and after the score appears, for scores clipped as a
        short window rather than a whole possession (free throws, extra points)."""
        return 3.0, 1.0

    def tail_of(self, previous: ScoreChange, change: ScoreChange) -> bool:
        """Is ``change`` a short follow-up to ``previous`` (same team, the score right
        before it) that should ride on its clip rather than stand alone? Basketball's
        and-one is handled by the core; football's extra point is the other case."""
        return False

    def export_extend(self, clip_kind: str, options: dict) -> float:
        """Extra seconds to add to the end of a clip of this kind at export time."""
        return 0.0

    def describe_points(self, points: int, kind: str) -> str:
        return f"+{points}"
