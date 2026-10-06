"""NBA adapter.

Play-by-play sources, tried in order and cached per game ID:
1. cdn.nba.com liveData (recent seasons; fast and reliable)
2. stats.nba.com playbyplayv3 (every season; throttles, so retried with backoff)
3. the same endpoint through nba_api, in case its header set gets through when ours does not

Game lookup uses stats.nba.com scoreboardv3 by date, which carries teams, final scores
and labels ("NBA Finals", "NYK leads 3-1") for any season.

NBA.com refuses requests from cloud servers: stats.nba.com stalls and cdn.nba.com answers
403. ESPN's public game data does not, so when NBA.com fails the lookup falls back to
ESPN's scoreboard, and a game found there (id ``espn:<event>``) gets its play-by-play
from ESPN's game summary. Team codes are translated to the NBA's so nothing downstream
can tell the difference.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
import time
from typing import TYPE_CHECKING

from . import http
from .base import FieldSpec, Game, PlayByPlayUnavailable, ScoreChange, ScoringEvent, SportAdapter

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline.timeline import ShotReset, Timeline

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
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
ESPN_PREFIX = "espn:"
# ESPN's team codes where they differ from the NBA's
ESPN_TRICODES = {"SA": "SAS", "NY": "NYK", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
# After NBA.com fails once it is left alone for this long; ESPN is used meanwhile.
NBA_COM_RETRY_SECONDS = 600.0
_nba_com_down_until = 0.0


def _nba_com_reachable() -> bool:
    return time.time() >= _nba_com_down_until


def _nba_com_failed(reason: str) -> None:
    global _nba_com_down_until
    _nba_com_down_until = time.time() + NBA_COM_RETRY_SECONDS
    log.info("NBA.com unavailable (%s); using ESPN's data for the next %d minutes", reason, NBA_COM_RETRY_SECONDS // 60)


def espn_tricode(abbreviation: str | None) -> str:
    code = (abbreviation or "").upper()
    return ESPN_TRICODES.get(code, code)


def _initial_name(full: str) -> str:
    """'Karl-Anthony Towns' -> 'K. Towns', the shape NBA.com uses."""
    parts = full.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) >= 2 else full


def games_from_espn(data: dict, date: str) -> list[Game]:
    games: list[Game] = []
    for ev in data.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        by_side = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        home, away = by_side.get("home") or {}, by_side.get("away") or {}
        status = ((comp.get("status") or {}).get("type") or {}).get("description") or ""
        label = " · ".join(n.get("headline") for n in comp.get("notes") or [] if n.get("headline"))
        games.append(
            Game(
                game_id=f"{ESPN_PREFIX}{ev.get('id')}",
                date=date,
                away=espn_tricode((away.get("team") or {}).get("abbreviation")),
                home=espn_tricode((home.get("team") or {}).get("abbreviation")),
                away_name=(away.get("team") or {}).get("displayName") or "",
                home_name=(home.get("team") or {}).get("displayName") or "",
                away_score=_int(away.get("score")),
                home_score=_int(home.get("score")),
                status=status,
                label=label,
            )
        )
    return games


def normalize_espn_plays(summary: dict) -> list[ScoringEvent]:
    """Scoring events from an ESPN game summary, in the same shape as NBA.com's.

    Points come from the running score, as with NBA.com, so the two sources agree play
    for play (a test holds them to that on a real game).
    """
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    side_of: dict[str, str] = {}
    code_of: dict[str, str] = {}
    for c in comp.get("competitors") or []:
        team_id = str((c.get("team") or {}).get("id") or c.get("id") or "")
        side_of[team_id] = c.get("homeAway") or ""
        code_of[team_id] = espn_tricode((c.get("team") or {}).get("abbreviation"))
    events: list[ScoringEvent] = []
    home = away = 0
    for play in summary.get("plays") or []:
        if not play.get("scoringPlay"):
            continue
        new_home, new_away = _int(play.get("homeScore")), _int(play.get("awayScore"))
        if new_home is None or new_away is None:
            continue
        d_home, d_away = new_home - home, new_away - away
        if d_home <= 0 and d_away <= 0:
            home, away = max(home, new_home), max(away, new_away)
            continue
        side = "home" if d_home > 0 else "away"
        team_id = str((play.get("team") or {}).get("id") or "")
        team = code_of.get(team_id) or next((code for tid, code in code_of.items() if side_of.get(tid) == side), "")
        kind_text = ((play.get("type") or {}).get("text") or "").lower()
        text = (play.get("text") or "").strip()
        is_ft = "free throw" in kind_text or "free throw" in text.lower()
        scorer = _initial_name(text.split(" makes ", 1)[0].strip()) if " makes " in text else ""
        events.append(
            ScoringEvent(
                event_id=str(play.get("id") or play.get("sequenceNumber") or len(events)),
                period=int((play.get("period") or {}).get("number") or 0),
                clock=http.parse_iso_clock((play.get("clock") or {}).get("displayValue")),
                team=team,
                points=d_home if side == "home" else d_away,
                scorer=scorer,
                description=text,
                score_away=new_away,
                score_home=new_home,
                kind="free_throw" if is_ft else "field_goal",
                extra={"side": side, "sub_type": kind_text},
            )
        )
        home, away = new_home, new_away
    return events


def espn_game_finished(summary: dict) -> bool:
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    status = (comp.get("status") or {}).get("type") or {}
    return bool(status.get("completed")) or status.get("state") == "post"


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


DEFENSE_IN_TEXT = re.compile(r"\b(STEAL|BLOCK)\b")
ESPN_DEFENSE = re.compile(r"\(([^()]+?) (steals|blocks)\)")  # "... turnover (Josh Hart steals)"
ESPN_BLOCK = re.compile(r"^(.+?) blocks ")  # "Mitchell Robinson blocks Dylan Harper's layup"


def normalize_defense(actions: list[dict]) -> list[ScoringEvent]:
    """Blocks and steals from an NBA action list. cdn.nba.com has them as their own
    actions; stats.nba.com leaves the type blank and says it in the description."""
    out: list[ScoringEvent] = []
    for action in actions:
        kind = (action.get("actionType") or "").lower()
        if kind not in ("steal", "block"):
            found = DEFENSE_IN_TEXT.search((action.get("description") or "").upper())
            if not found or (action.get("actionType") or ""):
                continue
            kind = found.group(1).lower()
        out.append(
            ScoringEvent(
                event_id=str(action.get("actionNumber", len(out))),
                period=int(action.get("period") or 0),
                clock=http.parse_iso_clock(action.get("clock")),
                team=(action.get("teamTricode") or "").upper(),
                points=0,
                scorer=action.get("playerNameI") or action.get("playerName") or "",
                description=(action.get("description") or "").strip(),
                score_away=_int(action.get("scoreAway")),
                score_home=_int(action.get("scoreHome")),
                kind=kind,
                extra={},
            )
        )
    return out


def normalize_espn_defense(summary: dict) -> list[ScoringEvent]:
    """Blocks and steals from an ESPN summary, where they sit inside the shot or
    turnover they caused: "... turnover (Julian Champagnie steals)"."""
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    code_of: dict[str, str] = {}
    for c in comp.get("competitors") or []:
        team_id = str((c.get("team") or {}).get("id") or c.get("id") or "")
        code_of[team_id] = espn_tricode((c.get("team") or {}).get("abbreviation"))
    out: list[ScoringEvent] = []
    for play in summary.get("plays") or []:
        text = " ".join((play.get("text") or "").split())
        found = ESPN_DEFENSE.search(text)
        if found:
            player, kind = found.group(1).strip(), "steal" if found.group(2) == "steals" else "block"
        elif blocked := ESPN_BLOCK.match(text):
            player, kind = blocked.group(1).strip(), "block"
        else:
            continue
        offence = str((play.get("team") or {}).get("id") or "")
        team = next((code for tid, code in code_of.items() if tid != offence), "")  # the defence
        out.append(
            ScoringEvent(
                event_id=f"{play.get('id') or play.get('sequenceNumber') or len(out)}d",
                period=int((play.get("period") or {}).get("number") or 0),
                clock=http.parse_iso_clock((play.get("clock") or {}).get("displayValue")),
                team=team,
                points=0,
                scorer=_initial_name(player),
                description=f"{_initial_name(player)} {kind.upper()}",
                score_away=_int(play.get("awayScore")),
                score_home=_int(play.get("homeScore")),
                kind=kind,
                extra={},
            )
        )
    return out


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
    def __init__(self) -> None:
        self._defense: dict[str, list[ScoringEvent]] = {}

    # The shot clock resets as the ball drops; the score follows later. How much later
    # depends on the broadcast: 0.5-2 s on some, 2.2-3.8 s on ESPN/ABC's bug. A reset this
    # close before the score is the make itself, and nothing from it on can be the start
    # of the possession.
    make_lag_max = 4.5
    held_lag_max = 8.0  # ...or further back if the shot clock never ran in between (and-one)
    make_lag_guard = 2.5  # assumed when the make's own reset cannot be seen
    # "Ends right after the make": this long after the ball drops, but not before the new
    # score has been on screen for a moment, and never later than score + post-roll.
    make_tail = 2.5
    score_hold = 0.5
    # A free throw clip runs from this long before the ball drops to this long after it.
    ft_lead, ft_tail = 2.0, 2.0
    assumed_lag = 1.0  # bug lag to assume when the game's baskets do not reveal it
    # A jump of 2+ with the clock stopped is free throws only if it had been stopped a
    # while before the score showed (27 s at the least in a real game). After a basket
    # through a foul it has been stopped for just the bug's lag.
    ft_stopped_min = 8.0

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
            data = None
            if _nba_com_reachable():
                try:
                    # one short try: on a server this stalls, and the answer is a minute away otherwise
                    data = http.get_json(STATS_SCOREBOARD, {"GameDate": date, "LeagueID": "00"}, STATS_HEADERS, attempts=1, timeout=8.0)
                except PlayByPlayUnavailable as exc:
                    _nba_com_failed(str(exc))
            if data is None:
                rows = [g.to_dict() for g in games_from_espn(http.get_json(ESPN_SCOREBOARD, {"dates": date.replace("-", "")}), date)]
                if rows and all(r["status"].lower().startswith("final") for r in rows):
                    http.write_cache("nba", cache_name, rows)
                return self._filter(rows, team)
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
        return self._filter(rows, team)

    @staticmethod
    def _filter(rows: list[dict], team: str | None) -> list[Game]:
        games = [Game(**r) for r in rows]
        if team:
            want = team.upper()
            games = [g for g in games if want in (g.away.upper(), g.home.upper())]
        return games

    def fetch_pbp(self, game_id: str) -> list[ScoringEvent]:
        cache_name = f"pbp_{game_id.replace(':', '_')}.json"  # "espn:" ids; a colon is not a file name on Windows
        cached = http.read_cache("nba", cache_name)
        if cached is not None:
            return [ScoringEvent.from_dict(e) for e in cached["events"]]

        if game_id.startswith(ESPN_PREFIX):
            summary = http.get_json(ESPN_SUMMARY, {"event": game_id[len(ESPN_PREFIX):]})
            events = normalize_espn_plays(summary)
            if not events:
                raise PlayByPlayUnavailable(f"ESPN has no plays for {game_id}")
            defense = normalize_espn_defense(summary)
            self._defense[game_id] = defense
            if espn_game_finished(summary):
                http.write_cache("nba", cache_name, {
                    "source": "espn", "events": [e.to_dict() for e in events], "defense": [e.to_dict() for e in defense],
                })
            return events
        if not _nba_com_reachable():
            raise PlayByPlayUnavailable("NBA.com is not answering from this server; pick the game again to use ESPN's data")

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
            if all("ReadTimeout" in e or "403" in e for e in errors):
                _nba_com_failed("; ".join(errors))
            raise PlayByPlayUnavailable(f"NBA play-by-play for {game_id} is unavailable ({'; '.join(errors)})")

        events = normalize_actions(actions)
        defense = normalize_defense(actions)
        self._defense[game_id] = defense
        finished = any((a.get("actionType") or "").lower() == "game" for a in actions) or source != "cdn"
        if events and finished:
            http.write_cache("nba", cache_name, {
                "source": source, "events": [e.to_dict() for e in events], "defense": [e.to_dict() for e in defense],
            })
        return events

    def fetch_defense(self, game_id: str) -> list[ScoringEvent]:
        cached = http.read_cache("nba", f"pbp_{game_id.replace(':', '_')}.json")
        if cached is not None and "defense" in cached:
            return [ScoringEvent.from_dict(e) for e in cached["defense"]]
        if game_id not in self._defense:
            self.fetch_pbp(game_id)  # fetches and remembers both
        return list(self._defense.get(game_id, []))

    @staticmethod
    def _nba_api_pbp(game_id: str) -> dict:
        from nba_api.stats.endpoints import playbyplayv3

        return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=20).get_dict()

    # -- clip rules ----------------------------------------------------------
    def classify(self, change: ScoreChange) -> str:
        # The play-by-play, when matched, knows what kind of score this was.
        kinds = {p.get("kind") for p in change.extra.get("pbp") or []} - {None}
        if kinds:
            return "free_throws" if kinds == {"free_throw"} else "field_goal"
        # +1 is always a free throw. A bigger jump with the clock stopped for a long time
        # is a trip to the line whose first make was not seen (bug hidden between shots).
        if change.points == 1 or (change.clock_stopped and change.stopped_before >= self.ft_stopped_min):
            return "free_throws"
        return "field_goal"

    def make_time(self, timeline: Timeline, change: ScoreChange) -> tuple[float | None, ShotReset | None]:
        """When the ball actually dropped, and the shot clock reset that shows it.

        The reset is the last one before the score appeared, close enough to be the make.
        With the shot clock off (end of a period) the game clock stopping does the same job:
        it stops on a make in the last minutes.
        """
        anchored = bool(change.extra.get("anchored"))  # the score first showed after a break
        for reset in reversed(timeline.shot_resets):
            if reset.t > change.t or math.isnan(reset.value):
                continue
            age = change.t - reset.t
            if anchored:
                # the make came before the cutaway only if the shot clock never ran again
                found = age <= self.make_lag_max and reset.t_start >= change.t - 0.5
            else:
                held = reset.t_start >= change.t - 1.0
                found = age <= self.make_lag_max or (held and age <= self.held_lag_max)
            if found:
                return reset.t, reset
            break
        if not anchored:
            return timeline.last_clock_stop(change.t, self.make_lag_max), None
        return None, None

    def bug_lag(self, timeline: Timeline) -> float | None:
        """How long after a make this broadcast's bug shows the score: the median over the
        game's baskets, both teams. None when too few makes show their reset."""
        cache = timeline.__dict__
        if "_nba_bug_lag" not in cache:
            resets = [r for r in timeline.shot_resets if not math.isnan(r.value)]
            lags: list[float] = []
            k = 0
            for t in sorted(timeline.score_change_times("away") + timeline.score_change_times("home")):
                while k < len(resets) and resets[k].t <= t:
                    k += 1
                if k and t - resets[k - 1].t <= self.make_lag_max:
                    lags.append(t - resets[k - 1].t)
            cache["_nba_bug_lag"] = statistics.median(lags) if len(lags) >= 4 else None
        return cache["_nba_bug_lag"]

    def clip_end(self, timeline: Timeline, change: ScoreChange, options: dict) -> float:
        latest = change.t + self.default_rolls[1]
        made, _ = self.make_time(timeline, change)
        if made is None:
            return latest
        return min(latest, max(change.t + self.score_hold, made + self.make_tail))

    def score_window(self, timeline: Timeline, change: ScoreChange) -> tuple[float, float]:
        # Free throws leave no trace on the shot clock, so the ball is taken to have
        # dropped one bug lag before the score showed.
        lag = self.bug_lag(timeline)
        if lag is None:
            lag = self.assumed_lag
        return lag + self.ft_lead, min(1.0, max(self.score_hold, self.ft_tail - lag))

    def possession_start(self, timeline: Timeline, change: ScoreChange) -> tuple[float, str]:
        """Latest of: shot clock reset to 24/14, game clock starting after a stoppage,
        the opponent scoring, this team's previous score, or the bug coming back from a
        break. Only what happened before the make counts. Falls back to the maximum clip
        length."""
        made, make_reset = self.make_time(timeline, change)
        # the shot clock going blank and the game clock stopping are the make too, and
        # are timed to the nearest sample: keep clear of the make by more than that
        limit = change.t - self.make_lag_guard if made is None else made - 0.75
        candidates: list[tuple[float, str]] = []
        for reset in reversed(timeline.shot_resets):
            if make_reset is not None and reset.index >= make_reset.index:
                continue
            if reset.t <= limit and reset.t_start <= (change.t - 1.0 if made is None else limit):
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
        for _gone, back in reversed(timeline.not_live_intervals):
            if back <= limit:
                # Play was already under way when the bug came back from a cutaway: the
                # possession began out of sight, and the clip starts where the bug returns.
                if back - _gone >= 4.0:
                    candidates.append((back + self.default_rolls[0], "bug_returned"))
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
