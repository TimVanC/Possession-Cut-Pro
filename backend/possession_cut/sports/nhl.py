"""NHL adapter.

Both the game list and play-by-play come from the league's public web API
(api-web.nhle.com, no key), cached per date and per game ID once final.

A goal clip starts at the faceoff that led to it when the goal came quickly, and 15 s
before the score appears otherwise: hockey has no possession signal on the bug, and a
shift that ends in a goal is rarely worth more than that.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from . import http
from .base import FieldSpec, Game, PlayByPlayUnavailable, ScoreChange, ScoringEvent, SportAdapter

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.timeline import Timeline

SCORE_URL = "https://api-web.nhle.com/v1/score/{date}"
PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
FINAL_STATES = ("OFF", "FINAL")

TEAMS = (
    ("ANA", "Anaheim Ducks"), ("BOS", "Boston Bruins"), ("BUF", "Buffalo Sabres"),
    ("CGY", "Calgary Flames"), ("CAR", "Carolina Hurricanes"), ("CHI", "Chicago Blackhawks"),
    ("COL", "Colorado Avalanche"), ("CBJ", "Columbus Blue Jackets"), ("DAL", "Dallas Stars"),
    ("DET", "Detroit Red Wings"), ("EDM", "Edmonton Oilers"), ("FLA", "Florida Panthers"),
    ("LAK", "Los Angeles Kings"), ("MIN", "Minnesota Wild"), ("MTL", "Montreal Canadiens"),
    ("NSH", "Nashville Predators"), ("NJD", "New Jersey Devils"), ("NYI", "New York Islanders"),
    ("NYR", "New York Rangers"), ("OTT", "Ottawa Senators"), ("PHI", "Philadelphia Flyers"),
    ("PIT", "Pittsburgh Penguins"), ("SJS", "San Jose Sharks"), ("SEA", "Seattle Kraken"),
    ("STL", "St. Louis Blues"), ("TBL", "Tampa Bay Lightning"), ("TOR", "Toronto Maple Leafs"),
    ("UTA", "Utah Mammoth"), ("VAN", "Vancouver Canucks"), ("VGK", "Vegas Golden Knights"),
    ("WSH", "Washington Capitals"), ("WPG", "Winnipeg Jets"),
)
TEAM_NAMES = dict(TEAMS)

_OVERTIME = re.compile(r"^(\d?)OT(\d?)$")


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value) -> str:
    """The feed wraps display strings per language: {'default': 'Toews', 'fr': ...}."""
    if isinstance(value, dict):
        return str(value.get("default") or "")
    return str(value or "")


def _status(game: dict) -> str:
    state = game.get("gameState") or ""
    if state in FINAL_STATES:
        outcome = game.get("gameOutcome") or {}
        last = outcome.get("lastPeriodType")
        if last == "SO":
            return "Final/SO"
        if last == "OT":
            extra = _int(outcome.get("otPeriods")) or 1
            return "Final/OT" if extra == 1 else f"Final/{extra}OT"
        return "Final"
    return {"LIVE": "Live", "CRIT": "Live", "FUT": "Scheduled", "PRE": "Pregame"}.get(state, state.title())


def _label(game: dict) -> str:
    series = game.get("seriesStatus") or {}
    number = _int(series.get("gameNumberOfSeries"))
    parts = (
        {1: "Preseason", 3: "Stanley Cup Playoffs"}.get(game.get("gameType"), ""),
        series.get("seriesTitle") or "",
        f"Game {number}" if number else "",
    )
    return " · ".join(p for p in parts if p)


def _game_row(game: dict, date: str) -> Game:
    away, home = game.get("awayTeam") or {}, game.get("homeTeam") or {}

    def name(team: dict) -> str:
        return TEAM_NAMES.get(team.get("abbrev") or "") or _text(team.get("name") or team.get("commonName"))

    return Game(
        game_id=str(game.get("id")),
        date=date,
        away=away.get("abbrev") or "",
        home=home.get("abbrev") or "",
        away_name=name(away),
        home_name=name(home),
        away_score=_int(away.get("score")),
        home_score=_int(home.get("score")),
        status=_status(game),
        label=_label(game),
    )


def normalize_plays(data: dict) -> list[ScoringEvent]:
    """Goals from a gamecenter play-by-play document, in game order.

    Shootout attempts are left out: the feed lists each one as a goal, but the bug never
    counts them (the winner gets a single goal once it is over). As with the NBA, each
    goal is read off the running score rather than trusted to be worth one.
    """
    sides: dict[int, str] = {}
    abbr: dict[str, str] = {}
    for side in ("away", "home"):
        team = data.get(f"{side}Team") or {}
        sides[team.get("id")] = side
        abbr[side] = (team.get("abbrev") or "").upper()
    names: dict[int, str] = {}
    for spot in data.get("rosterSpots") or []:
        first, last = _text(spot.get("firstName")), _text(spot.get("lastName"))
        names[spot.get("playerId")] = f"{first[0]}. {last}" if first else last

    events: list[ScoringEvent] = []
    away = home = 0
    for play in sorted(data.get("plays") or [], key=lambda p: p.get("sortOrder") or 0):
        period = play.get("periodDescriptor") or {}
        if play.get("typeDescKey") != "goal" or period.get("periodType") == "SO":
            continue
        details = play.get("details") or {}
        owner = sides.get(details.get("eventOwnerTeamId"))
        new_away, new_home = _int(details.get("awayScore")), _int(details.get("homeScore"))
        if new_away is None or new_home is None:
            new_away, new_home = away + (owner == "away"), home + (owner == "home")
        d_away, d_home = new_away - away, new_home - home
        if d_away <= 0 and d_home <= 0:
            continue
        side = "home" if d_home > 0 else "away"
        scorer = names.get(details.get("scoringPlayerId"), "")
        assists = [names[p] for p in (details.get("assist1PlayerId"), details.get("assist2PlayerId")) if p in names]
        events.append(
            ScoringEvent(
                event_id=str(play.get("eventId", len(events))),
                period=_int(period.get("number")) or 0,
                clock=http.parse_iso_clock(play.get("timeRemaining")),
                team=abbr[side],
                points=d_home if side == "home" else d_away,
                scorer=scorer,
                description=_describe(scorer, details, assists),
                score_away=new_away,
                score_home=new_home,
                kind="goal",
                extra={"side": side, "period_type": period.get("periodType") or "REG"},
            )
        )
        away, home = new_away, new_home
    return events


def _describe(scorer: str, details: dict, assists: list[str]) -> str:
    """'D. Toews (3) wrist shot, assists: B. Nelson, M. Necas' (the feed has no sentence)."""
    head = scorer or "Goal"
    total = _int(details.get("scoringPlayerTotal"))
    if total:
        head += f" ({total})"
    if details.get("shotType"):
        head += f" {details['shotType']} shot"
    return f"{head}, " + ("assists: " + ", ".join(assists) if assists else "unassisted")


class NHLAdapter(SportAdapter):
    key = "nhl"
    name = "NHL"
    bug_fields = (
        FieldSpec("away_label", "text", False, "the away (first-listed) team's abbreviation or name"),
        FieldSpec("away_score", "score", True, "the away team's goals (number only)"),
        FieldSpec("home_label", "text", False, "the home (second-listed) team's abbreviation or name"),
        FieldSpec("home_score", "score", True, "the home team's goals (number only)"),
        FieldSpec("period", "period", True, "the period indicator, e.g. 1st, 2nd, 3rd, OT, SO"),
        FieldSpec("clock", "clock", True, "the game clock, e.g. 14:27 or 38.4"),
    )
    default_rolls = (1.0, 2.0)
    max_clip_seconds = 30.0
    regulation_periods = 3
    period_seconds = 1200.0
    # Regular-season overtime. Playoff overtime periods are 20 minutes; the adapter is not
    # told which game it is reading, and elapsed() stays monotonic either way.
    overtime_seconds = 300.0
    max_points_per_event = 1
    period_label = "P"
    # How far back a goal clip reaches when the faceoff was longer ago than this.
    lookback_seconds = 15.0

    # -- league data -----------------------------------------------------
    def teams(self) -> list[dict[str, str]]:
        return sorted(({"abbr": abbr, "name": name} for abbr, name in TEAMS), key=lambda t: t["name"])

    def find_games(self, date: str, team: str | None = None) -> list[Game]:
        cache_name = f"games_{date}.json"
        rows = http.read_cache("nhl", cache_name)
        if rows is None:
            data = http.get_json(SCORE_URL.format(date=date))
            listed = [g for g in data.get("games") or [] if g.get("gameDate", date) == date]
            rows = [_game_row(g, date).to_dict() for g in listed]
            if rows and all(g.get("gameState") in FINAL_STATES for g in listed):
                http.write_cache("nhl", cache_name, rows)
        games = [Game(**r) for r in rows]
        if team:
            want = team.upper()
            games = [g for g in games if want in (g.away.upper(), g.home.upper())]
        return games

    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        cache_name = f"pbp_{game_id}.json"
        cached = http.read_cache("nhl", cache_name)
        if cached is not None:
            return [ScoringEvent.from_dict(e) for e in cached["events"]]

        data = http.get_json(PBP_URL.format(game_id=game_id))
        if not isinstance(data, dict) or not data.get("plays"):
            raise PlayByPlayUnavailable(f"NHL play-by-play for {game_id} is unavailable (no plays in the feed)")
        events = normalize_plays(data)
        if events and data.get("gameState") in FINAL_STATES:
            http.write_cache("nhl", cache_name, {"source": "api-web", "events": [e.to_dict() for e in events]})
        return events

    # -- reading the bug -------------------------------------------------------
    def parse_field(self, name: str, text: str):
        """Hockey's overtime is period 4, not the 5 a four-quarter sport would make it,
        and the feed numbers the shootout one past the first overtime."""
        if name != "period":
            return super().parse_field(name, text)
        key = re.sub(r"[^A-Z0-9]", "", text.upper())
        if key == "SO":
            return self.regulation_periods + 2
        overtime = _OVERTIME.match(key)
        if overtime:
            return self.regulation_periods + int(overtime.group(1) or overtime.group(2) or 1)
        value = super().parse_field(name, text)
        return value if value is not None and value <= self.regulation_periods else None

    # -- clip rules ----------------------------------------------------------
    def classify(self, change: ScoreChange) -> str:
        return "goal"

    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        """Later of the last faceoff (the clock starting after a stoppage) or 15 s before
        the score appeared."""
        floor = change.t - self.lookback_seconds
        for start in reversed(timeline.clock_starts):
            if start.t < change.t:
                if start.t >= floor:
                    return start.t, "faceoff"
                break
        return floor, "15s_before"
