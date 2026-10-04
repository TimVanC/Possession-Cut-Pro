"""MLB adapter.

Game lists and play-by-play come from the MLB Stats API (statsapi.mlb.com, no key), with
the MLB-StatsAPI package as a second way in when the direct call fails. The play-by-play
document carries no team names, so each fetch also reads the game's schedule entry, which
has the abbreviations and says whether the game is final.

Baseball has no clock. The "period" is the half-inning, numbered so it only ever goes up:
2n-1 for the top of inning n and 2n for the bottom. A scoring clip starts at the final
pitch of the plate appearance, which the bug shows as the last change of the count.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from . import http
from .base import FieldSpec, Game, PlayByPlayUnavailable, ScoreChange, ScoringEvent, SportAdapter

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.timeline import Timeline

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
PBP_URL = "https://statsapi.mlb.com/api/v1/game/{game_id}/playByPlay"
SCHEDULE_HYDRATE = "team,linescore"
POSTSEASON_TYPES = ("F", "D", "L", "W")
MAX_INNING = 30

TEAMS = (
    ("AZ", "Arizona Diamondbacks"), ("ATH", "Athletics"), ("ATL", "Atlanta Braves"),
    ("BAL", "Baltimore Orioles"), ("BOS", "Boston Red Sox"), ("CHC", "Chicago Cubs"),
    ("CWS", "Chicago White Sox"), ("CIN", "Cincinnati Reds"), ("CLE", "Cleveland Guardians"),
    ("COL", "Colorado Rockies"), ("DET", "Detroit Tigers"), ("HOU", "Houston Astros"),
    ("KC", "Kansas City Royals"), ("LAA", "Los Angeles Angels"), ("LAD", "Los Angeles Dodgers"),
    ("MIA", "Miami Marlins"), ("MIL", "Milwaukee Brewers"), ("MIN", "Minnesota Twins"),
    ("NYM", "New York Mets"), ("NYY", "New York Yankees"), ("PHI", "Philadelphia Phillies"),
    ("PIT", "Pittsburgh Pirates"), ("SD", "San Diego Padres"), ("SF", "San Francisco Giants"),
    ("SEA", "Seattle Mariners"), ("STL", "St. Louis Cardinals"), ("TB", "Tampa Bay Rays"),
    ("TEX", "Texas Rangers"), ("TOR", "Toronto Blue Jays"), ("WSH", "Washington Nationals"),
)

_HALF = r"TOP|T|▲|△|↑|\^|BOTTOM|BOT|B|▼|↓|MIDDLE|MID|END"
_INNING = re.compile(rf"^({_HALF})?(\d{{1,2}})(?:ST|ND|RD|TH)?({_HALF})?$")
_TOP_MARKS = ("TOP", "T", "▲", "△", "↑", "^")


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def half_inning(inning: int, bottom: bool) -> int:
    return 2 * inning if bottom else 2 * inning - 1


def parse_inning(text: str) -> int | None:
    """Half-inning number from the bug's inning indicator, None when it is not one.

    Reads 'TOP 5', 'BOT 5', 'T5', 'B5', '▲5', '▼5', 'Top 5th', 'MID 5', 'END 5' and a bare
    '5th'. Between halves ('MID', 'END') the top is over, so those count as the bottom.
    A bare number gives no half; it is read as the top, the earliest it could be, which
    keeps the number from ever running ahead of the game.
    """
    match = _INNING.match(re.sub(r"\s+", "", text.upper()))
    if not match:
        return None
    inning = int(match.group(2))
    if not 1 <= inning <= MAX_INNING:
        return None
    mark = match.group(1) or match.group(3)
    return half_inning(inning, bottom=mark is not None and mark not in _TOP_MARKS)


def _finished(game: dict) -> bool:
    return (game.get("status") or {}).get("abstractGameState") == "Final"


def _status(game: dict) -> str:
    status = game.get("status") or {}
    text = status.get("detailedState") or status.get("abstractGameState") or ""
    line = game.get("linescore") or {}
    played, scheduled = _int(line.get("currentInning")), _int(line.get("scheduledInnings")) or 9
    if text == "Final" and played and played != scheduled:
        return f"Final/{played}"
    return text


def _label(game: dict) -> str:
    parts: list[str] = []
    series = game.get("seriesDescription") or ""
    if series and series != "Regular Season":
        parts.append(series)
        if game.get("gameType") in POSTSEASON_TYPES and game.get("seriesGameNumber"):
            parts.append(f"Game {game['seriesGameNumber']}")
    if game.get("doubleHeader") in ("Y", "S"):
        parts.append(f"Doubleheader game {game.get('gameNumber')}")
    return " · ".join(parts)


def _game_row(game: dict, date: str) -> Game:
    teams = game.get("teams") or {}
    away, home = teams.get("away") or {}, teams.get("home") or {}
    away_team, home_team = away.get("team") or {}, home.get("team") or {}
    return Game(
        game_id=str(game.get("gamePk")),
        date=date,
        away=away_team.get("abbreviation") or "",
        home=home_team.get("abbreviation") or "",
        away_name=away_team.get("name") or "",
        home_name=home_team.get("name") or "",
        away_score=_int(away.get("score")),
        home_score=_int(home.get("score")),
        status=_status(game),
        label=_label(game),
    )


def _games_in(schedule: dict) -> list[dict]:
    return [g for day in schedule.get("dates") or [] for g in day.get("games") or []]


def normalize_plays(plays: list[dict], teams: dict[str, str]) -> list[ScoringEvent]:
    """Scoring plays from a playByPlay ``allPlays`` list: every play the running score
    went up on, worth the runs that scored on it. ``teams`` is {'away': abbr, 'home': abbr}."""
    events: list[ScoringEvent] = []
    away = home = 0
    for play in plays:
        result, about = play.get("result") or {}, play.get("about") or {}
        new_away, new_home = _int(result.get("awayScore")), _int(result.get("homeScore"))
        if new_away is None or new_home is None:
            continue
        d_away, d_home = new_away - away, new_home - home
        if d_away <= 0 and d_home <= 0:
            away, home = max(away, new_away), max(home, new_home)
            continue
        side = "home" if d_home > 0 else "away"
        inning = _int(about.get("inning")) or 0
        bottom = about.get("halfInning") == "bottom"
        batter = (play.get("matchup") or {}).get("batter") or {}
        events.append(
            ScoringEvent(
                event_id=str(about.get("atBatIndex", len(events))),
                period=half_inning(inning, bottom) if inning else 0,
                clock=None,
                team=teams.get(side, ""),
                points=d_home if side == "home" else d_away,
                scorer=batter.get("fullName") or "",
                description=(result.get("description") or "").strip(),
                score_away=new_away,
                score_home=new_home,
                kind="home_run" if result.get("eventType") == "home_run" else "run",
                extra={
                    "side": side, "inning": inning, "half": "bottom" if bottom else "top",
                    "event": result.get("event") or "", "rbi": _int(result.get("rbi")) or 0,
                },
            )
        )
        away, home = new_away, new_home
    return events


class MLBAdapter(SportAdapter):
    key = "mlb"
    name = "MLB"
    bug_fields = (
        FieldSpec("away_label", "text", False, "the away (first-listed or top) team's abbreviation or name"),
        FieldSpec("away_score", "score", True, "the away team's runs (number only)"),
        FieldSpec("home_label", "text", False, "the home (second-listed or bottom) team's abbreviation or name"),
        FieldSpec("home_score", "score", True, "the home team's runs (number only)"),
        # Unrestricted on purpose: the half is often only an arrow, which the period
        # character set would turn into a low-confidence letter.
        FieldSpec("period", "text", True, "the inning with its half, e.g. TOP 5, BOT 5, ▲5, ▼5, 5th"),
        FieldSpec("outs", "score", False, "the number of outs, where shown as a digit (0, 1 or 2)"),
        FieldSpec("count", "count", False, "the ball-strike count, e.g. 3-2"),
    )
    # The pre-roll is the PRD's "3 s before the final pitch".
    default_rolls = (3.0, 2.0)
    max_clip_seconds = 45.0
    regulation_periods = 9
    period_seconds = 0.0
    overtime_seconds = 0.0
    has_clock = False
    max_points_per_event = 4
    period_label = "Inn "
    options = (
        {"key": "hr_trot", "label": "Home run trot", "hint": "Five more seconds after a home run.",
         "default": True, "where": "export"},
    )
    trot_seconds = 5.0
    # Clip start when the count gives nothing to go on.
    fallback_seconds = 12.0

    # -- league data -----------------------------------------------------
    def teams(self) -> list[dict[str, str]]:
        return sorted(({"abbr": abbr, "name": name} for abbr, name in TEAMS), key=lambda t: t["name"])

    def find_games(self, date: str, team: str | None = None) -> list[Game]:
        cache_name = f"games_{date}.json"
        rows = http.read_cache("mlb", cache_name)
        if rows is None:
            data = http.get_json(SCHEDULE_URL, {"sportId": 1, "date": date, "hydrate": SCHEDULE_HYDRATE})
            listed = _games_in(data)
            rows = [_game_row(g, date).to_dict() for g in listed]
            if rows and all(_finished(g) for g in listed):
                http.write_cache("mlb", cache_name, rows)
        games = [Game(**r) for r in rows]
        if team:
            want = team.upper()
            games = [g for g in games if want in (g.away.upper(), g.home.upper())]
        return games

    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        cache_name = f"pbp_{game_id}.json"
        cached = http.read_cache("mlb", cache_name)
        if cached is not None:
            return [ScoringEvent.from_dict(e) for e in cached["events"]]

        errors: list[str] = []
        plays: list[dict] = []
        game: dict = {}
        source = ""
        for label, fetch in (("statsapi.mlb.com", self._direct), ("MLB-StatsAPI", self._statsapi)):
            try:
                pbp, schedule = fetch(game_id)
                plays = pbp.get("allPlays") or []
                game = next(iter(_games_in(schedule)), {})
                if plays and game:
                    source = label
                    break
                errors.append(f"{label}: no plays")
            except PlayByPlayUnavailable as exc:
                errors.append(f"{label}: {exc}")
            except Exception as exc:  # the package raises requests' errors and its own
                errors.append(f"{label}: {type(exc).__name__}")
        if not source:
            raise PlayByPlayUnavailable(f"MLB play-by-play for {game_id} is unavailable ({'; '.join(errors)})")

        row = _game_row(game, "")
        events = normalize_plays(plays, {"away": row.away, "home": row.home})
        if events and _finished(game):
            http.write_cache("mlb", cache_name, {"source": source, "events": [e.to_dict() for e in events]})
        return events

    @staticmethod
    def _direct(game_id: str) -> tuple[dict, dict]:
        pbp = http.get_json(PBP_URL.format(game_id=game_id))
        schedule = http.get_json(SCHEDULE_URL, {"sportId": 1, "gamePk": game_id, "hydrate": SCHEDULE_HYDRATE})
        return pbp, schedule

    @staticmethod
    def _statsapi(game_id: str) -> tuple[dict, dict]:
        import statsapi

        pbp = statsapi.get("game_playByPlay", {"gamePk": game_id})
        schedule = statsapi.get("schedule", {"sportId": 1, "gamePk": game_id, "hydrate": SCHEDULE_HYDRATE})
        return pbp, schedule

    # -- timing ------------------------------------------------------------
    def elapsed(self, period: int | None, clock: float | None) -> float | None:
        """No clock: game position is the half-inning number itself."""
        return None if period is None else float(period)

    def format_period(self, period: int | None) -> str:
        if period is None:
            return "?"
        return f"{'Bot' if period % 2 == 0 else 'Top'} {(period + 1) // 2}"

    # -- reading the bug -------------------------------------------------------
    def parse_field(self, name: str, text: str):
        if name == "period":
            return parse_inning(text)
        return super().parse_field(name, text)

    # -- clip rules ----------------------------------------------------------
    def classify(self, change: ScoreChange) -> str:
        """The bug cannot tell a home run from any other run; play-by-play can."""
        matched = change.extra.get("pbp") or []
        return "home_run" if any(e.get("kind") == "home_run" for e in matched) else "run"

    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        """The final pitch: the last change of the count before the score appeared.

        Foul balls with two strikes leave the count alone, so a last change too old to fit
        in a clip says nothing about when the final pitch was thrown. Then, and when the
        bug shows no count at all, the clip starts a fixed 12 s before the score.
        """
        reach = self.max_clip_seconds - sum(self.default_rolls)
        for t, _old, _new in reversed(timeline.text_changes("count")):
            if t < change.t:
                if change.t - t <= reach:
                    return t, "final_pitch"
                break
        return change.t - self.fallback_seconds, "12s_before"

    def export_extend(self, clip_kind: str, options: dict) -> float:
        if clip_kind == "home_run" and options.get("hr_trot", True):
            return self.trot_seconds
        return 0.0
