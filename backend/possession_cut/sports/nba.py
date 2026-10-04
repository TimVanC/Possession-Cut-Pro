"""NBA adapter.

Play-by-play sources, tried in order and cached per game ID:
1. cdn.nba.com liveData (recent seasons; fast and reliable)
2. stats.nba.com playbyplayv3 (every season; throttles, so retried with backoff)
3. the same endpoint through nba_api, in case its header set gets through when ours does not

Game lookup uses stats.nba.com scoreboardv3 by date, which carries teams, final scores
and labels ("NBA Finals", "NYK leads 3-1") for any season.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import http
from .base import FieldSpec, Game, PlayByPlayUnavailable, ScoreChange, ScoringEvent, SportAdapter

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.timeline import Timeline

log = logging.getLogger(__name__)

STATS_HEADERS = {
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
STATS_PBP = "https://stats.nba.com/stats/playbyplayv3"
STATS_SCOREBOARD = "https://stats.nba.com/stats/scoreboardv3"


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_actions(actions: list[dict]) -> list[ScoringEvent]:
    """Scoring events from an NBA action list (CDN liveData or stats v3 share this shape).

    Points come from the running score, not from the action type, so the two sources
    (and any future variant) are read the same way.
    """
    events: list[ScoringEvent] = []
    home = away = 0
    for action in actions:
        new_home, new_away = _int(action.get("scoreHome")), _int(action.get("scoreAway"))
        if new_home is None or new_away is None:
            continue
        d_home, d_away = new_home - home, new_away - away
        if d_home <= 0 and d_away <= 0:
            home, away = max(home, new_home), max(away, new_away)
            continue
        side = "home" if d_home > 0 else "away"
        points = d_home if side == "home" else d_away
        kind_raw = (action.get("actionType") or "").lower()
        is_ft = "free" in kind_raw
        events.append(
            ScoringEvent(
                event_id=str(action.get("actionNumber", len(events))),
                period=int(action.get("period") or 0),
                clock=http.parse_iso_clock(action.get("clock")),
                team=(action.get("teamTricode") or "").upper(),
                points=points,
                scorer=action.get("playerNameI") or action.get("playerName") or "",
                description=(action.get("description") or "").strip(),
                score_away=new_away,
                score_home=new_home,
                kind="free_throw" if is_ft else "field_goal",
                extra={"side": side, "sub_type": action.get("subType") or ""},
            )
        )
        home, away = new_home, new_away
    return events


def teams_from_events(events: list[ScoringEvent]) -> dict[str, str]:
    """{'away': 'SAS', 'home': 'NYK'} as seen in the scoring events."""
    out: dict[str, str] = {}
    for ev in events:
        side = ev.extra.get("side")
        if side and ev.team and side not in out:
            out[side] = ev.team
        if len(out) == 2:
            break
    return out


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

    # -- league data -----------------------------------------------------
    def teams(self) -> list[dict[str, str]]:
        try:
            from nba_api.stats.static import teams as static_teams

            return sorted(
                ({"abbr": t["abbreviation"], "name": t["full_name"]} for t in static_teams.get_teams()),
                key=lambda t: t["name"],
            )
        except Exception:  # pragma: no cover - static data ships with the package
            return []

    def find_games(self, date: str, team: str | None = None) -> list[Game]:
        cache_name = f"games_{date}.json"
        rows = http.read_cache("nba", cache_name)
        if rows is None:
            data = http.get_json(STATS_SCOREBOARD, {"GameDate": date, "LeagueID": "00"}, STATS_HEADERS)
            rows = []
            for g in (data.get("scoreboard") or {}).get("games", []):
                away, home = g.get("awayTeam") or {}, g.get("homeTeam") or {}
                label = " · ".join(x for x in (g.get("gameLabel"), g.get("gameSubLabel"), g.get("seriesText")) if x)
                rows.append(
                    Game(
                        game_id=str(g.get("gameId")),
                        date=date,
                        away=away.get("teamTricode") or "",
                        home=home.get("teamTricode") or "",
                        away_name=f"{away.get('teamCity', '')} {away.get('teamName', '')}".strip(),
                        home_name=f"{home.get('teamCity', '')} {home.get('teamName', '')}".strip(),
                        away_score=_int(away.get("score")),
                        home_score=_int(home.get("score")),
                        status=(g.get("gameStatusText") or "").strip(),
                        label=label,
                    ).to_dict()
                )
            if rows and all(r["status"].lower().startswith("final") for r in rows):
                http.write_cache("nba", cache_name, rows)
        games = [Game(**r) for r in rows]
        if team:
            want = team.upper()
            games = [g for g in games if want in (g.away.upper(), g.home.upper())]
        return games

    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        cache_name = f"pbp_{game_id}.json"
        cached = http.read_cache("nba", cache_name)
        if cached is not None:
            return [ScoringEvent.from_dict(e) for e in cached["events"]]

        errors: list[str] = []
        actions: list[dict] | None = None
        source = ""
        for label, fetch in (
            ("cdn", lambda: http.get_json(CDN_PBP.format(game_id=game_id), headers=STATS_HEADERS, attempts=2)),
            ("stats", lambda: http.get_json(
                STATS_PBP, {"GameID": game_id, "StartPeriod": "0", "EndPeriod": "0"}, STATS_HEADERS)),
            ("nba_api", lambda: self._nba_api_pbp(game_id)),
        ):
            try:
                data = fetch()
                actions = (data.get("game") or {}).get("actions") or []
                if actions:
                    source = label
                    break
                errors.append(f"{label}: no actions")
            except PlayByPlayUnavailable as exc:
                errors.append(f"{label}: {exc}")
            except Exception as exc:  # nba_api raises its own assortment
                errors.append(f"{label}: {type(exc).__name__}")
        if not actions:
            raise PlayByPlayUnavailable(f"NBA play-by-play for {game_id} is unavailable ({'; '.join(errors)})")

        events = normalize_actions(actions)
        finished = any((a.get("actionType") or "").lower() == "game" for a in actions) or source != "cdn"
        if events and finished:
            http.write_cache("nba", cache_name, {"source": source, "events": [e.to_dict() for e in events]})
        return events

    @staticmethod
    def _nba_api_pbp(game_id: str) -> dict:
        from nba_api.stats.endpoints import playbyplayv3

        return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=20).get_dict()

    # -- clip rules ----------------------------------------------------------
    def classify(self, change: ScoreChange) -> str:
        # +1 is always a free throw. A bigger jump with the clock stopped throughout is a
        # trip to the line whose first make was not seen (the bug was hidden between shots).
        if change.points == 1 or change.clock_stopped:
            return "free_throws"
        return "field_goal"

    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        """Latest of: shot clock reset to 24/14, game clock starting after a stoppage,
        the opponent scoring, or this team's previous score. Falls back to the maximum
        clip length."""
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

    def describe_points(self, points: int, kind: str) -> str:
        if kind == "free_throws":
            return f"{points} FT" if points == 1 else f"{points} FTs"
        return {2: "2-pointer", 3: "3-pointer"}.get(points, f"+{points}")
