"""NFL adapter.

Schedules and play-by-play come from nflverse through nflreadpy (every season since
1999). nflverse ships one file per season, so a game's scoring plays are cut out of it
and cached per game ID; the season file is only downloaded for a game not seen before.

nflverse stamps each play with the clock at the snap. That is what locates the start of
a scoring play on the bug, and why the bug clock (read when the score shows, after the
play) is allowed to sit well away from the play-by-play clock when the two are matched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from . import http
from .base import FieldSpec, Game, PlayByPlayUnavailable, ScoreChange, ScoringEvent, SportAdapter

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.timeline import Timeline

TEAMS = (
    ("ARI", "Arizona Cardinals"), ("ATL", "Atlanta Falcons"), ("BAL", "Baltimore Ravens"),
    ("BUF", "Buffalo Bills"), ("CAR", "Carolina Panthers"), ("CHI", "Chicago Bears"),
    ("CIN", "Cincinnati Bengals"), ("CLE", "Cleveland Browns"), ("DAL", "Dallas Cowboys"),
    ("DEN", "Denver Broncos"), ("DET", "Detroit Lions"), ("GB", "Green Bay Packers"),
    ("HOU", "Houston Texans"), ("IND", "Indianapolis Colts"), ("JAX", "Jacksonville Jaguars"),
    ("KC", "Kansas City Chiefs"), ("LV", "Las Vegas Raiders"), ("LAC", "Los Angeles Chargers"),
    ("LA", "Los Angeles Rams"), ("MIA", "Miami Dolphins"), ("MIN", "Minnesota Vikings"),
    ("NE", "New England Patriots"), ("NO", "New Orleans Saints"), ("NYG", "New York Giants"),
    ("NYJ", "New York Jets"), ("PHI", "Philadelphia Eagles"), ("PIT", "Pittsburgh Steelers"),
    ("SF", "San Francisco 49ers"), ("SEA", "Seattle Seahawks"), ("TB", "Tampa Bay Buccaneers"),
    ("TEN", "Tennessee Titans"), ("WAS", "Washington Commanders"),
)
# nflverse schedules keep the abbreviation a franchise had at the time.
FORMER_TEAMS = {"OAK": "Oakland Raiders", "SD": "San Diego Chargers", "STL": "St. Louis Rams"}
TEAM_NAMES = {**dict(TEAMS), **FORMER_TEAMS}
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "WSH": "WAS"}
ROUNDS = {"WC": "Wild Card", "DIV": "Divisional Round", "CON": "Conference Championship"}

PBP_COLUMNS = (
    "play_id", "qtr", "quarter_seconds_remaining", "play_type", "total_home_score", "total_away_score",
    "td_player_name", "kicker_player_name", "passer_player_name", "receiver_player_name",
    "rusher_player_name", "safety", "safety_player_name", "two_point_attempt", "desc",
)
SCORER_COLUMNS = {
    "touchdown": ("td_player_name",),
    "field_goal": ("kicker_player_name",),
    "extra_point": ("kicker_player_name",),
    "two_point": ("receiver_player_name", "rusher_player_name", "passer_player_name"),
    "safety": ("safety_player_name",),
}


def _load_schedule(season: int) -> pl.DataFrame:
    import nflreadpy

    return nflreadpy.load_schedules(seasons=[season])


def _load_pbp(season: int) -> pl.DataFrame:
    import nflreadpy

    return nflreadpy.load_pbp(seasons=[season])


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def season_of(date: str) -> int:
    """The year the season started: January and February games belong to the year before."""
    year, month = int(date[:4]), int(date[5:7])
    return year - 1 if month <= 2 else year


def _franchise(abbr: str) -> str:
    abbr = abbr.upper()
    return FRANCHISE.get(abbr, abbr)


def _roman(number: int) -> str:
    out = ""
    for value, numeral in ((50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while number >= value:
            out += numeral
            number -= value
    return out


def _label(row: dict) -> str:
    kind = row.get("game_type") or ""
    if kind == "SB":
        number = (_int(row.get("season")) or 0) - 1965
        return f"Super Bowl {_roman(number)}" if 0 < number < 90 else "Super Bowl"
    if kind == "REG":
        return f"Week {row['week']}" if row.get("week") is not None else ""
    return ROUNDS.get(kind, kind)


def _game_row(row: dict, date: str) -> Game:
    away, home = row.get("away_team") or "", row.get("home_team") or ""
    away_score, home_score = _int(row.get("away_score")), _int(row.get("home_score"))
    played = away_score is not None and home_score is not None
    return Game(
        game_id=str(row.get("game_id")),
        date=date,
        away=away,
        home=home,
        away_name=TEAM_NAMES.get(away, away),
        home_name=TEAM_NAMES.get(home, home),
        away_score=away_score,
        home_score=home_score,
        status=("Final/OT" if row.get("overtime") else "Final") if played else "Scheduled",
        label=_label(row),
    )


def _kind(points: int, row: dict, previous: ScoringEvent | None) -> str:
    if points >= 6:
        return "touchdown"
    if points == 3:
        return "field_goal"
    if points == 1:
        return "extra_point"
    if points == 2:
        if row.get("safety"):
            return "safety"
        after_touchdown = previous is not None and previous.kind == "touchdown"
        return "two_point" if after_touchdown or row.get("two_point_attempt") else "safety"
    return "score"


def _scorer(kind: str, row: dict) -> str:
    for column in SCORER_COLUMNS.get(kind, ()):
        if row.get(column):
            return str(row[column])
    return ""


def normalize_plays(rows: list[dict], teams: dict[str, str]) -> list[ScoringEvent]:
    """Scoring plays from one game's nflverse rows (in the order nflverse gives them).

    Points and the scoring side come from the running totals, so a pick-six goes to the
    defense without reading who had the ball. ``teams`` is {'away': abbr, 'home': abbr}.
    """
    events: list[ScoringEvent] = []
    away = home = 0
    for row in rows:
        new_away, new_home = _int(row.get("total_away_score")), _int(row.get("total_home_score"))
        if new_away is None or new_home is None:
            continue
        d_away, d_home = new_away - away, new_home - home
        if d_away <= 0 and d_home <= 0:
            away, home = max(away, new_away), max(home, new_home)
            continue
        side = "home" if d_home > 0 else "away"
        points = d_home if side == "home" else d_away
        kind = _kind(points, row, events[-1] if events else None)
        snap = row.get("quarter_seconds_remaining")
        snap = None if snap is None else float(snap)
        play_id = _int(row.get("play_id"))
        events.append(
            ScoringEvent(
                event_id=str(len(events) if play_id is None else play_id),
                period=_int(row.get("qtr")) or 0,
                clock=snap,
                team=teams.get(side, ""),
                points=points,
                scorer=_scorer(kind, row),
                description=(row.get("desc") or "").strip(),
                score_away=new_away,
                score_home=new_home,
                kind=kind,
                extra={"side": side, "snap_clock": snap, "play_type": row.get("play_type") or ""},
            )
        )
        away, home = new_away, new_home
    return events


def _snap_clock(change: ScoreChange) -> float | None:
    """Clock at the snap of the play this score was matched to, if play-by-play has it."""
    matched = change.extra.get("pbp") or []
    snap = (matched[0].get("extra") or {}).get("snap_clock") if matched else None
    return float(snap) if isinstance(snap, int | float) else None


class NFLAdapter(SportAdapter):
    key = "nfl"
    name = "NFL"
    bug_fields = (
        FieldSpec("away_label", "text", False, "the away (first-listed) team's abbreviation or name"),
        FieldSpec("away_score", "score", True, "the away team's score (number only, room for 2 digits)"),
        FieldSpec("home_label", "text", False, "the home (second-listed) team's abbreviation or name"),
        FieldSpec("home_score", "score", True, "the home team's score (number only, room for 2 digits)"),
        FieldSpec("period", "period", True, "the quarter indicator, e.g. 1st, 2nd, 3rd, 4th, OT"),
        FieldSpec("clock", "clock", True, "the game clock, e.g. 12:20 or :05"),
        FieldSpec("down_distance", "text", False, "down and distance, e.g. 3rd & 7"),
        FieldSpec("shot_clock", "shot_clock", False, "the play clock, a small 1-2 digit number (40 down to 0)"),
    )
    default_rolls = (2.0, 2.0)
    max_clip_seconds = 30.0
    regulation_periods = 4
    period_seconds = 900.0
    overtime_seconds = 600.0
    max_points_per_event = 8
    period_label = "Q"
    shot_clock_max = 40.0
    shot_clock_resets = (40.0, 25.0)
    # nflverse stamps the clock at the snap; the bug shows the score after the play.
    pbp_clock_tolerance = 20.0
    options = (
        {"key": "include_tail", "label": "Extra point / 2-point try",
         "hint": "Add the try after a touchdown to its clip as a short trailing segment.",
         "default": True, "where": "setup"},
    )
    # Without play-by-play: how far back a clock start still counts as the snap, and the
    # fixed start used when there is none.
    snap_lookback = 20.0
    fallback_seconds = 12.0
    # A try whose clock could not be read is still the try if it follows this closely.
    try_window = 90.0

    # -- league data -----------------------------------------------------
    def teams(self) -> list[dict[str, str]]:
        return sorted(({"abbr": abbr, "name": name} for abbr, name in TEAMS), key=lambda t: t["name"])

    def find_games(self, date: str, team: str | None = None) -> list[Game]:
        cache_name = f"games_{date}.json"
        rows = http.read_cache("nfl", cache_name)
        if rows is None:
            try:
                schedule = _load_schedule(season_of(date))
                listed = schedule.filter(pl.col("gameday").cast(pl.String) == date).to_dicts()
            except Exception as exc:  # nflreadpy raises requests', polars' and its own errors
                raise PlayByPlayUnavailable(
                    f"nflverse schedule for {date} is unavailable ({type(exc).__name__})"
                ) from exc
            rows = [_game_row(r, date).to_dict() for r in listed]
            if rows and all(r["status"].startswith("Final") for r in rows):
                http.write_cache("nfl", cache_name, rows)
        games = [Game(**r) for r in rows]
        if team:
            want = _franchise(team)
            games = [g for g in games if want in (_franchise(g.away), _franchise(g.home))]
        return games

    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        cache_name = f"pbp_{game_id}.json"
        cached = http.read_cache("nfl", cache_name)
        if cached is not None:
            return [ScoringEvent.from_dict(e) for e in cached["events"]]

        # nflverse game IDs read season_week_away_home, with the abbreviations the schedule uses
        parts = game_id.split("_")
        if len(parts) != 4 or not parts[0].isdigit():
            raise PlayByPlayUnavailable(f"{game_id!r} is not an nflverse game ID (expected e.g. 2016_21_NE_ATL)")
        try:
            season = _load_pbp(int(parts[0]))
            plays = season.filter(pl.col("game_id") == game_id)
            rows = plays.select([c for c in PBP_COLUMNS if c in plays.columns]).to_dicts()
        except Exception as exc:  # nflreadpy raises requests', polars' and its own errors
            raise PlayByPlayUnavailable(
                f"nflverse play-by-play for {game_id} is unavailable ({type(exc).__name__})"
            ) from exc
        if not rows:
            raise PlayByPlayUnavailable(f"nflverse has no play-by-play for {game_id}")

        events = normalize_plays(rows, {"away": parts[2], "home": parts[3]})
        if events and any(r.get("desc") == "END GAME" for r in rows):
            http.write_cache("nfl", cache_name, {"source": "nflverse", "events": [e.to_dict() for e in events]})
        return events

    # -- reading the bug -------------------------------------------------------
    def parse_field(self, name: str, text: str):
        """Football bugs drop the minutes under a minute: ':05'."""
        if name == "clock" and text.strip().startswith(":"):
            text = "0" + text.strip()
        return super().parse_field(name, text)

    # -- clip rules ----------------------------------------------------------
    def classify(self, change: ScoreChange) -> str:
        if change.points >= 6:
            return "touchdown"
        if change.points == 3:
            return "field_goal"
        if change.points == 1:
            return "extra_point"
        if change.points == 2:
            matched = [e.get("kind") for e in change.extra.get("pbp") or []]
            if matched and matched[-1] in ("two_point", "safety"):
                return matched[-1]
            # From the bug alone: a try happens with the clock where the touchdown left it.
            return "two_point" if change.clock_stopped else "safety"
        return "score"

    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        """The scoring play's snap.

        With play-by-play: the last moment before the score that the bug clock showed the
        play's snap clock in that quarter. Without it: the clock starting to run within
        20 s of the score, else a fixed 12 s before the score.
        """
        snap = _snap_clock(change)
        if snap is not None and change.period is not None:
            shown = (timeline.period == change.period) & (np.abs(timeline.clock - snap) <= 0.5) & (timeline.t < change.t)
            at_snap = np.flatnonzero(shown)
            if len(at_snap):
                return float(timeline.t[at_snap[-1]]), "snap_pbp"
        for start in reversed(timeline.clock_starts):
            if start.t < change.t:
                if change.t - start.t <= self.snap_lookback:
                    return start.t, "clock_start"
                break
        return change.t - self.fallback_seconds, "12s_before"

    def tail_of(self, previous: ScoreChange, change: ScoreChange) -> bool:
        """The extra point or two-point try rides on its touchdown's clip: one or two
        points by the same team with the game clock where the touchdown left it."""
        if previous.team != change.team or previous.points < 6 or change.points not in (1, 2):
            return False
        if None not in (previous.period, change.period) and previous.period != change.period:
            return False
        if previous.clock is None or change.clock is None:
            return 0 <= change.t - previous.t <= self.try_window
        return abs(previous.clock - change.clock) <= 1.0
