"""NFL, NHL and MLB adapters against recorded league responses, and their clip rules on
simulated bug reads (no video, no network).

Fixtures: NHL VGK @ COL 2026-04-11 (overtime) and VAN @ SJS the same night (shootout),
MLB NYY @ AZ 2026-09-19 (three home runs), NFL Super Bowl LI from nflverse.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from possession_cut.pipeline.clips import build_clips
from possession_cut.pipeline.events import detect_score_events
from possession_cut.pipeline.matching import label_clips, match_events, sides_of
from possession_cut.pipeline.ocr import CHARSETS
from possession_cut.pipeline.timeline import build_timeline
from possession_cut.sports import (
    PlayByPlayUnavailable,
    ScoreChange,
    available_sports,
    get_adapter,
    mlb,
    nfl,
    nhl,
)
from possession_cut.sports import http as sport_http

from .conftest import FIXTURES
from .test_timeline import make_raw, running

NFL, NHL, MLB = get_adapter("nfl"), get_adapter("nhl"), get_adapter("mlb")

NHL_DATE, NHL_GAME, NHL_SHOOTOUT = "2026-04-11", "2025021274", "2025021275"
MLB_DATE, MLB_GAME = "2026-09-19", "825029"
NFL_DATE, NFL_GAME = "2017-02-05", "2016_21_NE_ATL"


def fixture(sport: str, name: str):
    return json.loads((FIXTURES / sport / name).read_text(encoding="utf-8"))


def _no_network(*_args, **_kwargs):
    raise AssertionError("a test reached for the real network")


@pytest.fixture()
def net(monkeypatch, settings):
    """Serve the recorded NHL and MLB responses instead of the network; count the calls.

    ``fail`` holds URL fragments that should be down; ``hooks`` may answer a request first.
    """
    calls: list[str] = []
    fail: set[str] = set()
    hooks: list = []

    def fake_get_json(url, params=None, headers=None, **_kw):
        calls.append(url)
        params = params or {}
        if any(tag in url for tag in fail):
            raise PlayByPlayUnavailable(f"{url} is down (test)")
        for hook in hooks:
            answer = hook(url, params)
            if answer is not None:
                return answer
        if url == nhl.SCORE_URL.format(date=NHL_DATE):
            return fixture("nhl", f"score_{NHL_DATE}.json")
        if "/v1/score/" in url:
            return {"games": []}
        for game_id in (NHL_GAME, NHL_SHOOTOUT):
            if url == nhl.PBP_URL.format(game_id=game_id):
                return fixture("nhl", f"play-by-play_{game_id}.json")
        if url == mlb.SCHEDULE_URL:
            assert params["sportId"] == 1 and params["hydrate"] == "team,linescore"
            if params.get("date") == MLB_DATE:
                return fixture("mlb", f"schedule_{MLB_DATE}.json")
            if str(params.get("gamePk")) == MLB_GAME:
                return fixture("mlb", f"schedule_gamePk_{MLB_GAME}.json")
            return {"totalGames": 0, "dates": []}
        if url == mlb.PBP_URL.format(game_id=MLB_GAME):
            return fixture("mlb", f"playByPlay_{MLB_GAME}.json")
        raise PlayByPlayUnavailable(f"{url} returned 404")

    monkeypatch.setattr(sport_http, "get_json", fake_get_json)
    monkeypatch.setattr(mlb.MLBAdapter, "_statsapi", staticmethod(_no_network))
    return SimpleNamespace(calls=calls, fail=fail, hooks=hooks)


@pytest.fixture()
def nflverse(monkeypatch, settings):
    """Stand in for nflreadpy: the 2016 season is the recorded Super Bowl LI rows."""

    source = SimpleNamespace(
        calls=[],
        schedule=pl.DataFrame(fixture("nfl", f"schedule_{NFL_DATE}.json"), infer_schema_length=None),
        plays=pl.DataFrame(fixture("nfl", f"pbp_{NFL_GAME}.json"), infer_schema_length=None),
    )

    def load_schedule(season: int) -> pl.DataFrame:
        source.calls.append(("schedule", season))
        return source.schedule if season == 2016 else source.schedule.clear()

    def load_pbp(season: int) -> pl.DataFrame:
        source.calls.append(("pbp", season))
        if season != 2016:
            raise ValueError("Season must be between 1999 and 2026")
        return source.plays

    monkeypatch.setattr(nfl, "_load_schedule", load_schedule)
    monkeypatch.setattr(nfl, "_load_pbp", load_pbp)
    return source


def check_scoring_events(adapter, events, game) -> None:
    """What every adapter's play-by-play must satisfy: game order, a running score that
    only one team moves per event and by exactly its points, labels, and the final."""
    away = home = 0
    played = -1.0
    for e in events:
        side = e.extra["side"]
        assert side in ("away", "home") and e.team == (game.away if side == "away" else game.home)
        assert e.points > 0 and e.scorer and e.description
        moved = (e.points, 0) if side == "away" else (0, e.points)
        assert (e.score_away - away, e.score_home - home) == moved
        elapsed = adapter.elapsed(e.period, e.clock)
        assert elapsed is not None and elapsed >= played
        away, home, played = e.score_away, e.score_home, elapsed
    assert (away, home) == (game.away_score, game.home_score)
    assert len({e.event_id for e in events}) == len(events)
    assert sides_of(events) == {"away": game.away, "home": game.home}


def with_text(raw, name: str, values: list[str]):
    """Add a text field make_raw has no column for (count, down and distance)."""
    raw.texts[name] = list(values)
    raw.confs[name] = np.ones(len(values), dtype=np.float32)
    return raw


def cut(adapter, raw, follow: str = "home", pbp=None, options: dict | None = None):
    """Timeline -> score events -> play-by-play match -> clips, chained the way analyze() does."""
    tl = build_timeline(raw, adapter)
    events = detect_score_events(tl, adapter)
    result = None
    if pbp:
        result = match_events(events, pbp, clock_tolerance=adapter.pbp_clock_tolerance)
        for change in events:
            match = result.matches.get(change.index)
            if match is not None:
                change.extra["pbp"] = [e.to_dict() for e in match.events]
    clips = build_clips(tl, events, adapter, follow, options)
    label_clips(clips, result)
    return tl, events, clips


def score_change(points: int, team: str = "home", clock: float | None = 735.0, period: int | None = 2, t: float = 100.0, **kw):
    return ScoreChange(
        index=0, team=team, points=points, t=t, t_prev=t - 0.5, period=period, clock=clock, clock_before=clock,
        score_before=0, score_after=points, score_away=0, score_home=points, **kw,
    )


# -- registry and constants --------------------------------------------------------------


def test_all_four_sports_are_registered():
    assert [s["key"] for s in available_sports()] == ["nba", "nfl", "nhl", "mlb"]
    for adapter, key, name, n_teams in ((NFL, "nfl", "NFL", 32), (NHL, "nhl", "NHL", 32), (MLB, "mlb", "MLB", 30)):
        assert get_adapter(key) is adapter and (adapter.key, adapter.name) == (key, name)
        teams = adapter.teams()
        assert len(teams) == n_teams and len({t["abbr"] for t in teams}) == n_teams
        assert all(t["abbr"] and t["name"] for t in teams)
    assert {"abbr": "NE", "name": "New England Patriots"} in NFL.teams()
    assert {"abbr": "VGK", "name": "Vegas Golden Knights"} in NHL.teams()
    assert {"abbr": "AZ", "name": "Arizona Diamondbacks"} in MLB.teams()
    with pytest.raises(ValueError, match="Unknown sport"):
        get_adapter("curling")


def test_bug_fields_and_sport_constants():
    core = ["away_label", "away_score", "home_label", "home_score", "period"]
    assert [f.name for f in NFL.bug_fields] == [*core, "clock", "down_distance", "shot_clock"]
    assert [f.name for f in NHL.bug_fields] == [*core, "clock"]
    assert [f.name for f in MLB.bug_fields] == [*core, "outs", "count"]
    for adapter in (NFL, NHL, MLB):
        assert all(f.charset in CHARSETS for f in adapter.bug_fields)
        required = {f.name for f in adapter.bug_fields if f.required}
        assert {"away_score", "home_score", "period"} <= required and "away_label" not in required

    assert (NFL.regulation_periods, NFL.period_seconds, NFL.overtime_seconds, NFL.period_label) == (4, 900.0, 600.0, "Q")
    assert (NFL.default_rolls, NFL.max_clip_seconds, NFL.max_points_per_event) == ((2.0, 2.0), 30.0, 8)
    assert (NFL.shot_clock_max, NFL.shot_clock_resets, NFL.pbp_clock_tolerance) == (40.0, (40.0, 25.0), 20.0)
    assert (NHL.regulation_periods, NHL.period_seconds, NHL.overtime_seconds, NHL.period_label) == (3, 1200.0, 300.0, "P")
    assert (NHL.default_rolls, NHL.max_clip_seconds, NHL.max_points_per_event) == ((1.0, 2.0), 30.0, 1)
    assert (MLB.regulation_periods, MLB.has_clock, MLB.period_label) == (9, False, "Inn ")
    assert (MLB.default_rolls, MLB.max_clip_seconds, MLB.max_points_per_event) == ((3.0, 2.0), 45.0, 4)
    assert MLB.options == (
        {"key": "hr_trot", "label": "Home run trot", "hint": "Five more seconds after a home run.",
         "default": True, "where": "export"},
    )
    assert [o["key"] for o in NFL.options] == ["include_tail"], "the core's switch for the try after a touchdown"


@pytest.mark.parametrize(
    ("key", "points"),
    [
        ("nfl", [(1, 900.0), (1, 2.0), (2, 900.0), (2, 899.0), (4, 0.0), (5, 600.0), (5, 0.0)]),
        ("nhl", [(1, 1200.0), (1, 5.0), (2, 1200.0), (3, 0.0), (4, 300.0), (4, 10.0), (5, 300.0)]),
        ("mlb", [(1, None), (2, None), (17, None), (18, None), (19, None)]),
    ],
)
def test_elapsed_never_goes_backward_across_periods(key, points):
    adapter = get_adapter(key)
    values = [adapter.elapsed(period, clock) for period, clock in points]
    assert None not in values and values == sorted(values) and values[0] < values[-1]
    assert adapter.elapsed(None, 10.0) is None


def test_elapsed_meets_at_the_period_boundary():
    assert NFL.elapsed(1, 0.0) == NFL.elapsed(2, 900.0) == 900.0
    assert NFL.elapsed(4, 0.0) == NFL.elapsed(5, 600.0) == 3600.0
    assert NHL.elapsed(1, 0.0) == NHL.elapsed(2, 1200.0) == 1200.0
    assert NHL.elapsed(3, 0.0) == NHL.elapsed(4, 300.0) == 3600.0
    assert MLB.elapsed(9, None) == 9.0 and MLB.elapsed(10, None) == 10.0, "top and bottom of the 5th"


# -- NHL ---------------------------------------------------------------------------------


def test_nhl_games_by_date_and_team(net):
    games = NHL.find_games(NHL_DATE)
    assert len(games) == 15 and all(g.date == NHL_DATE and g.status.startswith("Final") for g in games)
    g = next(g for g in games if g.game_id == NHL_GAME)
    assert (g.away, g.home, g.away_score, g.home_score) == ("VGK", "COL", 3, 2)
    assert (g.away_name, g.home_name) == ("Vegas Golden Knights", "Colorado Avalanche")
    assert g.status == "Final/OT" and g.label == ""
    shootout = next(g for g in games if g.game_id == NHL_SHOOTOUT)
    assert (shootout.away, shootout.away_score, shootout.home, shootout.home_score) == ("VAN", 4, "SJS", 3)
    assert shootout.status == "Final/SO"
    assert NHL.find_games(NHL_DATE, team="col") == [g]
    assert NHL.find_games(NHL_DATE, team="QUE") == []
    assert NHL.find_games("2026-04-10") == [], "no games that night"
    # a finished slate is cached: the second lookup does not hit the network
    before = len(net.calls)
    assert NHL.find_games(NHL_DATE, team="VGK") == [g]
    assert len(net.calls) == before


def test_nhl_live_slate_is_not_cached_and_playoff_games_are_labeled(net):
    live = {  # the shape of a playoff night's score response, cut down, with the game still on
        "games": [{
            "id": 2025030213, "gameType": 3, "gameDate": "2026-05-10", "gameState": "LIVE",
            "awayTeam": {"id": 7, "name": {"default": "Sabres"}, "abbrev": "BUF", "score": 1},
            "homeTeam": {"id": 8, "name": {"default": "Canadiens"}, "abbrev": "MTL", "score": 2},
            "seriesStatus": {"round": 2, "seriesTitle": "2nd Round", "gameNumberOfSeries": 3},
        }],
    }
    net.hooks.append(lambda url, params: live if url.endswith("/score/2026-05-10") else None)
    (g,) = NHL.find_games("2026-05-10")
    assert (g.away, g.away_score, g.home, g.home_score, g.status) == ("BUF", 1, "MTL", 2, "Live")
    assert g.label == "Stanley Cup Playoffs · 2nd Round · Game 3" and g.home_name == "Montreal Canadiens"
    NHL.find_games("2026-05-10")
    assert len(net.calls) == 2, "scores can still change, so the slate is fetched again"


def test_nhl_play_by_play_is_normalized_and_cached(net):
    events = NHL.fetch_pbp(NHL_GAME)
    game = NHL.find_games(NHL_DATE, team="VGK")[0]
    check_scoring_events(NHL, events, game)
    assert [(e.period, e.clock, e.team, e.score_away, e.score_home) for e in events] == [
        (1, 643.0, "COL", 0, 1), (1, 373.0, "VGK", 1, 1), (2, 1071.0, "VGK", 2, 1),
        (2, 544.0, "COL", 2, 2), (4, 221.0, "VGK", 3, 2),
    ]
    assert all(e.kind == "goal" and e.points == 1 for e in events)
    first, winner = events[0], events[-1]
    assert first.scorer == "D. Toews" and first.extra == {"side": "home", "period_type": "REG"}
    assert first.description == "D. Toews (3) wrist shot, assists: B. Nelson, M. Necas"
    assert winner.scorer == "J. Eichel" and winner.extra["period_type"] == "OT" and "unassisted" in winner.description

    n = len(net.calls)
    again = NHL.fetch_pbp(NHL_GAME)
    assert len(net.calls) == n, "served from the per-game disk cache"
    assert [e.to_dict() for e in again] == [e.to_dict() for e in events]


def test_nhl_shootout_attempts_are_not_goals(net):
    feed = fixture("nhl", f"play-by-play_{NHL_SHOOTOUT}.json")
    in_shootout = [
        p for p in feed["plays"] if p["typeDescKey"] == "goal" and p["periodDescriptor"]["periodType"] == "SO"
    ]
    assert len(in_shootout) == 3, "the feed lists each made attempt as a goal"
    events = NHL.fetch_pbp(NHL_SHOOTOUT)
    assert len(events) == 6 and all(e.extra["period_type"] == "REG" and e.period <= 3 for e in events)
    assert (events[-1].score_away, events[-1].score_home) == (3, 3), "level after 65 minutes"
    game = NHL.find_games(NHL_DATE, team="SJS")[0]
    assert (game.away_score, game.home_score) == (4, 3), "the final adds one goal for the shootout winner"


def test_nhl_goal_without_a_running_score_is_counted_from_its_team():
    data = {
        "awayTeam": {"id": 1, "abbrev": "AAA"}, "homeTeam": {"id": 2, "abbrev": "HHH"},
        "rosterSpots": [{"playerId": 9, "firstName": {"default": "Pat"}, "lastName": {"default": "Smith"}}],
        "plays": [
            {"eventId": 5, "sortOrder": 20, "typeDescKey": "goal", "timeRemaining": "01:00",
             "periodDescriptor": {"number": 1, "periodType": "REG"}, "details": {"eventOwnerTeamId": 2, "scoringPlayerId": 9}},
            {"eventId": 4, "sortOrder": 10, "typeDescKey": "goal", "timeRemaining": "15:30",
             "periodDescriptor": {"number": 1, "periodType": "REG"}, "details": {"eventOwnerTeamId": 1}},
            {"eventId": 6, "sortOrder": 30, "typeDescKey": "shot-on-goal", "timeRemaining": "00:30",
             "periodDescriptor": {"number": 1, "periodType": "REG"}, "details": {"eventOwnerTeamId": 1}},
        ],
    }
    events = nhl.normalize_plays(data)
    assert [(e.event_id, e.team, e.clock, e.score_away, e.score_home) for e in events] == [
        ("4", "AAA", 930.0, 1, 0), ("5", "HHH", 60.0, 1, 1),
    ], "ordered by sortOrder, scored by the team that owns the event"
    assert events[1].scorer == "P. Smith" and events[1].description == "P. Smith, unassisted"


def test_nhl_unavailable_play_by_play_raises_cleanly(net):
    with pytest.raises(PlayByPlayUnavailable, match="404"):
        NHL.fetch_pbp("1999020001")
    net.fail.add("gamecenter")
    with pytest.raises(PlayByPlayUnavailable, match="down"):
        NHL.fetch_pbp(NHL_GAME)
    net.fail.clear()
    net.hooks.append(lambda url, params: {"gameState": "FUT", "plays": []} if "gamecenter" in url else None)
    with pytest.raises(PlayByPlayUnavailable, match="no plays"):
        NHL.fetch_pbp(NHL_GAME)
    net.hooks.clear()
    net.fail.add("/score/")
    with pytest.raises(PlayByPlayUnavailable):
        NHL.find_games(NHL_DATE)


@pytest.mark.parametrize(
    ("text", "period"),
    [("1st", 1), ("2ND", 2), ("3rd", 3), ("P2", 2), ("OT", 4), ("2OT", 5), ("OT2", 5), ("SO", 5), ("4th", None), ("", None)],
)
def test_nhl_overtime_is_the_fourth_period(text, period):
    assert NHL.parse_field("period", text) == period
    assert NHL.parse_field("clock", "14:27") == 867.0 and NHL.parse_field("home_score", "3") == 3
    assert [NHL.format_period(p) for p in (1, 3, 4, 5)] == ["P1", "P3", "OT", "2OT"]


def test_nhl_goal_long_after_the_faceoff_starts_15s_before_the_score():
    rows = [{"clock": "10:00"}] * 12  # t 0-5.5: whistle, clock stopped
    rows += running(599.5, 80)  # t 6-45.5: faceoff, then 40 s of play
    rows += [{"clock": "9:20"}] * 4  # t 46-47.5: goal, clock stopped, the bug still shows 10
    rows += [{"clock": "9:20", "home": 11}] * 30  # t 48: the goal is on the bug
    tl, events, clips = cut(NHL, make_raw(rows))
    faceoff = tl.clock_starts[-1].t
    assert 5.0 <= faceoff <= 6.0 and events[0].t == 48.0
    (clip,) = clips
    assert (clip.kind, clip.points, clip.start_cause) == ("goal", 1, "15s_before")
    assert clip.segments == [[48.0 - 15.0 - 1.0, 48.0 + 2.0]], "15 s back plus the 1 s pre-roll; 2 s after the score shows"
    assert not clip.warnings


def test_nhl_quick_goal_starts_at_the_faceoff():
    rows = running(700, 20)  # t 0-9.5: play
    rows += [{"clock": "11:30"}] * 16  # t 10-17.5: whistle
    rows += running(689.5, 12)  # t 18-23.5: faceoff, six seconds of play
    rows += [{"clock": "11:24"}] * 4  # t 24-25.5: goal
    rows += [{"clock": "11:24", "home": 11}] * 20  # t 26: on the bug
    tl, events, clips = cut(NHL, make_raw(rows))
    faceoff = tl.clock_starts[-1].t
    assert 17.0 <= faceoff <= 18.0 and events[0].t == 26.0
    (clip,) = clips
    assert clip.start_cause == "faceoff"
    assert clip.segments == [[pytest.approx(faceoff - 1.0), 28.0]], "the faceoff, less the pre-roll"
    assert NHL.possession_start(tl, events[0]) == (faceoff, "faceoff")


def test_nhl_overtime_winner_is_matched_and_labeled(net):
    pbp = NHL.fetch_pbp(NHL_GAME)
    ot = {"period": "OT", "home": 2}
    rows = [{"clock": "5:00", "away": 2, **ot}] * 8  # t 0-3.5: overtime about to start
    rows += running(299.5, 158, away=2, **ot)  # t 4-82.5: 1:19 of play
    rows += [{"clock": "3:41", "away": 2, **ot}] * 4  # t 83-84.5: goal at 3:41
    rows += [{"clock": "3:41", "away": 3, **ot}] * 20  # t 85: on the bug
    tl, events, clips = cut(NHL, make_raw(rows), "away", pbp)
    assert (events[0].period, events[0].clock) == (4, 221.0), "OT reads as period 4, as the feed numbers it"
    (clip,) = clips
    assert clip.pbp_event_id == pbp[-1].event_id and clip.scorer == "J. Eichel" and not clip.warnings
    assert clip.segments == [[69.0, 87.0]] and clip.start_cause == "15s_before"


# -- MLB ---------------------------------------------------------------------------------


def test_mlb_games_by_date_and_team(net):
    games = MLB.find_games(MLB_DATE)
    assert len(games) == 15 and all(g.date == MLB_DATE for g in games)
    g = next(g for g in games if g.game_id == MLB_GAME)
    assert (g.away, g.home, g.away_score, g.home_score) == ("NYY", "AZ", 3, 5)
    assert (g.away_name, g.home_name) == ("New York Yankees", "Arizona Diamondbacks")
    assert g.status == "Final" and g.label == ""
    extras = next(g for g in games if g.game_id == "823003")
    assert (extras.away, extras.away_score, extras.home, extras.home_score, extras.status) == ("WSH", 8, "STL", 5, "Final/11")
    assert MLB.find_games(MLB_DATE, team="az") == [g]
    assert MLB.find_games(MLB_DATE, team="MON") == []
    assert MLB.find_games("2026-12-25") == []
    before = len(net.calls)
    assert MLB.find_games(MLB_DATE, team="NYY") == [g]
    assert len(net.calls) == before, "a finished slate is cached"


def test_mlb_postseason_and_doubleheader_labels():
    series = {"gameType": "W", "seriesDescription": "World Series", "seriesGameNumber": 7, "doubleHeader": "N"}
    assert mlb._label(series) == "World Series · Game 7"
    assert mlb._label({"gameType": "R", "seriesDescription": "Regular Season", "seriesGameNumber": 2,
                       "doubleHeader": "S", "gameNumber": 2}) == "Doubleheader game 2"
    assert mlb._status({"status": {"abstractGameState": "Live", "detailedState": "In Progress"}}) == "In Progress"


def test_mlb_play_by_play_is_normalized_and_cached(net):
    events = MLB.fetch_pbp(MLB_GAME)
    assert len(net.calls) == 2, "the plays, and the schedule entry that names the teams"
    game = MLB.find_games(MLB_DATE, team="AZ")[0]
    check_scoring_events(MLB, events, game)
    assert [(e.period, e.team, e.points, e.kind) for e in events] == [
        (9, "NYY", 1, "run"), (10, "AZ", 2, "run"), (11, "NYY", 1, "home_run"),
        (11, "NYY", 1, "home_run"), (12, "AZ", 1, "run"), (18, "AZ", 2, "home_run"),
    ], "period is the half-inning: 2n-1 for the top of inning n, 2n for the bottom"
    assert all(e.clock is None for e in events)
    walk_off = events[-1]
    assert walk_off.scorer == "Pavin Smith" and "homers" in walk_off.description
    assert walk_off.extra == {"side": "home", "inning": 9, "half": "bottom", "event": "Home Run", "rbi": 2}
    assert (walk_off.score_away, walk_off.score_home) == (3, 5)
    assert events[1].extra["event"] == "Triple" and events[1].kind == "run"

    n = len(net.calls)
    again = MLB.fetch_pbp(MLB_GAME)
    assert len(net.calls) == n, "served from the per-game disk cache"
    assert [e.to_dict() for e in again] == [e.to_dict() for e in events]


def test_mlb_play_by_play_falls_back_to_the_statsapi_package(net, monkeypatch):
    asked: list[str] = []

    def package(game_id):
        asked.append(game_id)
        return fixture("mlb", f"playByPlay_{MLB_GAME}.json"), fixture("mlb", f"schedule_gamePk_{MLB_GAME}.json")

    monkeypatch.setattr(mlb.MLBAdapter, "_statsapi", staticmethod(package))
    net.fail.add("statsapi.mlb.com")
    events = MLB.fetch_pbp(MLB_GAME)
    assert asked == [MLB_GAME]
    assert (events[-1].score_away, events[-1].score_home) == (3, 5) and events[-1].team == "AZ"
    assert sum(e.kind == "home_run" for e in events) == 3


def test_mlb_unavailable_play_by_play_raises_cleanly(net, monkeypatch):
    monkeypatch.setattr(mlb.MLBAdapter, "_statsapi", staticmethod(lambda game_id: (_ for _ in ()).throw(TimeoutError())))
    with pytest.raises(PlayByPlayUnavailable, match="unavailable.*404.*TimeoutError"):
        MLB.fetch_pbp("1")
    net.fail.add("statsapi.mlb.com")
    with pytest.raises(PlayByPlayUnavailable, match="unavailable"):
        MLB.fetch_pbp(MLB_GAME)
    with pytest.raises(PlayByPlayUnavailable):
        MLB.find_games(MLB_DATE)


def test_mlb_game_in_progress_is_not_cached(net):
    live = fixture("mlb", f"schedule_gamePk_{MLB_GAME}.json")
    live["dates"][0]["games"][0]["status"].update(abstractGameState="Live", detailedState="In Progress")
    net.hooks.append(lambda url, params: live if url == mlb.SCHEDULE_URL and "gamePk" in params else None)
    assert len(MLB.fetch_pbp(MLB_GAME)) == 6
    MLB.fetch_pbp(MLB_GAME)
    assert len(net.calls) == 4, "fetched again: more runs may come"


@pytest.mark.parametrize(
    ("text", "period"),
    [
        ("TOP 5", 9), ("BOT 5", 10), ("T5", 9), ("B5", 10), ("▲5", 9), ("▼5", 10), ("▲ 5", 9), ("5 ▼", 10),
        ("Top 5th", 9), ("Bot 5th", 10), ("BOTTOM 5", 10), ("5th", 9), ("5", 9), ("1st", 1), ("Bot 1st", 2),
        ("MID 5", 10), ("END 5th", 10), ("T10", 19), ("BOT 12", 24),
        ("", None), ("FINAL", None), ("45", None), ("0", None), ("TOP", None), ("3-2", None),
    ],
)
def test_mlb_inning_text_becomes_a_half_inning_number(text, period):
    assert MLB.parse_field("period", text) == period


def test_mlb_other_fields_and_period_names():
    assert MLB.parse_field("away_score", "12") == 12 and MLB.parse_field("home_score", "x") is None
    assert MLB.parse_field("count", " 3-2 ") == "3-2" and MLB.parse_field("outs", "2") == "2"
    assert [MLB.format_period(p) for p in (1, 2, 9, 10, 18, 19)] == ["Top 1", "Bot 1", "Top 5", "Bot 5", "Bot 9", "Top 10"]
    assert MLB.format_period(None) == "?"
    assert mlb.half_inning(5, bottom=False) == 9 and mlb.half_inning(5, bottom=True) == 10


def mlb_raw(count: list[str] | None, period: str = "BOT 9"):
    """A 60 s stretch at 2 fps: 3-3, then the home side's 5 shows at t=46."""
    rows = [{"period": period, "away": 3, "home": 3}] * 92 + [{"period": period, "away": 3, "home": 5}] * 28
    raw = make_raw(rows)
    return raw if count is None else with_text(raw, "count", count)


def test_mlb_clip_starts_3s_before_the_final_pitch_and_a_home_run_gets_its_trot(net):
    pbp = MLB.fetch_pbp(MLB_GAME)
    # strike one registers at t=25; the home run pitch (about t=40) leaves the count alone
    tl, events, clips = cut(MLB, mlb_raw(["0-0"] * 50 + ["0-1"] * 70), "home", pbp)
    assert tl.text_changes("count") == [(25.0, "0-0", "0-1")]
    change = events[0]
    assert (change.t, change.points, change.period, change.clock) == (46.0, 2, 18, None)
    assert [e["event_id"] for e in change.extra["pbp"]] == [pbp[-1].event_id], "matched on inning half and running score"
    (clip,) = clips
    assert (clip.kind, clip.points, clip.start_cause) == ("home_run", 2, "final_pitch")
    assert clip.segments == [[25.0 - 3.0, 46.0 + 2.0]], "3 s before the last count change, 2 s after the score"
    assert clip.scorer == "Pavin Smith" and not clip.warnings
    assert MLB.export_extend(clip.kind, {}) == 5.0
    assert MLB.export_extend(clip.kind, {"hr_trot": False}) == 0.0
    assert MLB.export_extend("run", {}) == 0.0


def test_mlb_without_play_by_play_a_run_is_just_a_run():
    _, events, clips = cut(MLB, mlb_raw(["0-0"] * 50 + ["0-1"] * 70))
    assert MLB.classify(events[0]) == "run" and clips[0].kind == "run"
    assert clips[0].segments == [[22.0, 48.0]]
    events[0].extra["pbp"] = [{"kind": "run"}, {"kind": "home_run"}]
    assert MLB.classify(events[0]) == "home_run"


def test_mlb_falls_back_to_12s_when_the_count_cannot_place_the_pitch():
    _, _, clips = cut(MLB, mlb_raw(None))
    assert clips[0].start_cause == "12s_before" and clips[0].segments == [[46.0 - 12.0 - 3.0, 48.0]], "no count on the bug"

    # two-strike fouls: the count last moved 44 s before the score, too long ago to be the final pitch
    tl, events, clips = cut(MLB, mlb_raw(["0-0"] * 4 + ["0-2"] * 116))
    assert tl.text_changes("count") == [(2.0, "0-0", "0-2")]
    assert clips[0].start_cause == "12s_before" and clips[0].segments == [[31.0, 48.0]]

    # the count resetting for the next batter on the very sample the run shows is not a pitch
    tl, events, _ = cut(MLB, mlb_raw(["0-0"] * 50 + ["0-1"] * 42 + ["0-0"] * 28))
    assert [t for t, _, _ in tl.text_changes("count")] == [25.0, 46.0]
    assert MLB.possession_start(tl, events[0]) == (25.0, "final_pitch")


def test_mlb_timeline_runs_on_innings_without_a_clock():
    rows = [{"period": "▲5", "away": 1}] * 10 + [{"period": "▲5", "away": 2}] * 10
    rows += [{"period": "▼5", "away": 2}] * 10 + [{"period": "TOP 6", "away": 2}] * 10
    tl = build_timeline(make_raw(rows), MLB)
    assert np.isnan(tl.clock).all() and tl.live.all()
    assert tl.period.tolist() == [9.0] * 20 + [10.0] * 10 + [11.0] * 10
    (event,) = detect_score_events(tl, MLB)
    assert (event.team, event.points, event.period, event.clock) == ("away", 1, 9, None)


# -- NFL ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("date", "season"),
    [("2017-02-05", 2016), ("2017-01-01", 2016), ("2016-12-25", 2016), ("2016-09-11", 2016), ("2017-09-07", 2017)],
)
def test_nfl_season_of_a_date(date, season):
    assert nfl.season_of(date) == season


def test_nfl_games_by_date_and_team(nflverse):
    games = NFL.find_games(NFL_DATE)
    assert nflverse.calls == [("schedule", 2016)], "a February game belongs to the season that began the year before"
    (g,) = games
    assert (g.game_id, g.date, g.away, g.home, g.away_score, g.home_score) == (NFL_GAME, NFL_DATE, "NE", "ATL", 34, 28)
    assert (g.away_name, g.home_name) == ("New England Patriots", "Atlanta Falcons")
    assert g.status == "Final/OT" and g.label == "Super Bowl LI"
    assert NFL.find_games(NFL_DATE, team="ne") == games and NFL.find_games(NFL_DATE, team="ATL") == games
    assert NFL.find_games(NFL_DATE, team="DAL") == []
    assert nflverse.calls == [("schedule", 2016)], "a finished slate is cached"
    assert NFL.find_games("2017-02-06") == [] and NFL.find_games("2017-09-10") == []
    assert nflverse.calls[1:] == [("schedule", 2016), ("schedule", 2017)]


def test_nfl_schedule_keeps_old_abbreviations_and_unplayed_slates_are_not_cached(nflverse):
    nflverse.schedule = pl.DataFrame([
        {"game_id": "2016_18_OAK_HOU", "season": 2016, "game_type": "WC", "week": 18, "gameday": "2017-01-07",
         "away_team": "OAK", "away_score": 14, "home_team": "HOU", "home_score": 27, "overtime": 0},
        {"game_id": "2016_18_DET_SEA", "season": 2016, "game_type": "WC", "week": 18, "gameday": "2017-01-07",
         "away_team": "DET", "away_score": None, "home_team": "SEA", "home_score": None, "overtime": None},
        {"game_id": "2016_17_NE_MIA", "season": 2016, "game_type": "REG", "week": 17, "gameday": "2017-01-01",
         "away_team": "NE", "away_score": 35, "home_team": "MIA", "home_score": 14, "overtime": 0},
    ])
    played, upcoming = NFL.find_games("2017-01-07")
    assert (played.away, played.away_name, played.status, played.label) == ("OAK", "Oakland Raiders", "Final", "Wild Card")
    assert (upcoming.away_score, upcoming.home_score, upcoming.status) == (None, None, "Scheduled")
    assert NFL.find_games("2017-01-07", team="LV") == [played], "the franchise's current abbreviation finds it too"
    assert NFL.find_games("2017-01-07", team="oak") == [played]
    assert len(nflverse.calls) == 3, "one game is still to be played, so nothing was cached"
    assert NFL.find_games("2017-01-01")[0].label == "Week 17"


def test_nfl_super_bowl_li_scoring_sequence(nflverse):
    events = NFL.fetch_pbp(NFL_GAME)
    game = NFL.find_games(NFL_DATE)[0]
    check_scoring_events(NFL, events, game)
    assert [(e.team, e.kind, e.score_away, e.score_home) for e in events] == [
        ("ATL", "touchdown", 0, 6), ("ATL", "extra_point", 0, 7),
        ("ATL", "touchdown", 0, 13), ("ATL", "extra_point", 0, 14),
        ("ATL", "touchdown", 0, 20), ("ATL", "extra_point", 0, 21),
        ("NE", "field_goal", 3, 21),
        ("ATL", "touchdown", 3, 27), ("ATL", "extra_point", 3, 28),
        ("NE", "touchdown", 9, 28),  # the extra point hit the upright
        ("NE", "field_goal", 12, 28),
        ("NE", "touchdown", 18, 28), ("NE", "two_point", 20, 28),
        ("NE", "touchdown", 26, 28), ("NE", "two_point", 28, 28),
        ("NE", "touchdown", 34, 28),
    ]
    assert [e.points for e in events] == [6, 1, 6, 1, 6, 1, 3, 6, 1, 6, 3, 6, 2, 6, 2, 6]
    assert all(e.clock is not None and e.extra["snap_clock"] == e.clock for e in events)
    first = events[0]
    assert (first.event_id, first.period, first.clock, first.scorer) == ("888", 2, 740.0, "D.Freeman")
    assert first.extra == {"side": "home", "snap_clock": 740.0, "play_type": "run"}
    pick_six = events[4]
    assert pick_six.scorer == "R.Alford" and "INTERCEPTED" in pick_six.description, "scored by the defense"
    assert [e.scorer for e in events if e.kind == "two_point"] == ["J.White", "D.Amendola"]
    assert {e.scorer for e in events if e.kind == "field_goal"} == {"S.Gostkowski"}
    assert {e.scorer for e in events if e.kind == "extra_point"} == {"M.Bryant"}
    winner = events[-1]
    assert (winner.period, winner.clock, winner.scorer) == (5, 668.0, "J.White") and "TOUCHDOWN" in winner.description


def test_nfl_play_by_play_is_cached_per_game(nflverse):
    events = NFL.fetch_pbp(NFL_GAME)
    assert nflverse.calls == [("pbp", 2016)]
    again = NFL.fetch_pbp(NFL_GAME)
    assert nflverse.calls == [("pbp", 2016)], "the season file is not loaded a second time"
    assert [e.to_dict() for e in again] == [e.to_dict() for e in events]


def test_nfl_game_without_its_end_marker_is_not_cached(nflverse):
    nflverse.plays = nflverse.plays.filter(pl.col("desc") != "END GAME")
    assert len(NFL.fetch_pbp(NFL_GAME)) == 16
    NFL.fetch_pbp(NFL_GAME)
    assert nflverse.calls == [("pbp", 2016), ("pbp", 2016)]


def test_nfl_unavailable_play_by_play_raises_cleanly(nflverse, monkeypatch):
    with pytest.raises(PlayByPlayUnavailable, match="not an nflverse game ID"):
        NFL.fetch_pbp("0042500404")
    assert nflverse.calls == []
    with pytest.raises(PlayByPlayUnavailable, match="no play-by-play"):
        NFL.fetch_pbp("2016_01_CAR_DEN")
    with pytest.raises(PlayByPlayUnavailable, match="ValueError"):
        NFL.fetch_pbp("2099_01_NE_ATL")
    monkeypatch.setattr(nfl, "_load_schedule", lambda season: (_ for _ in ()).throw(ConnectionError("offline")))
    with pytest.raises(PlayByPlayUnavailable, match="ConnectionError"):
        NFL.find_games("2016-09-11")


def test_nfl_two_points_is_a_try_after_a_touchdown_and_a_safety_otherwise():
    def row(play_id, home, away=0, **more):
        return {"play_id": play_id, "qtr": 1, "quarter_seconds_remaining": 900 - play_id, "desc": f"play {play_id}",
                "total_home_score": home, "total_away_score": away, **more}

    teams = {"away": "AAA", "home": "HHH"}
    flagged = [
        row(1, 2, safety=1.0, safety_player_name="D.End"),  # a safety to open the scoring
        row(2, 8, safety=0.0, td_player_name="R.Back"),
        row(3, 8, safety=0.0),  # the try fails
        row(4, 8, away=2, safety=1.0),  # a safety straight after a touchdown is still a safety
        row(5, 14, away=2, safety=0.0),
        row(6, 14, away=4, safety=0.0, two_point_attempt=1.0),  # the defense runs the try back
    ]
    events = nfl.normalize_plays(flagged, teams)
    assert [(e.team, e.points, e.kind) for e in events] == [
        ("HHH", 2, "safety"), ("HHH", 6, "touchdown"), ("AAA", 2, "safety"), ("HHH", 6, "touchdown"), ("AAA", 2, "two_point"),
    ]
    assert events[0].scorer == "D.End" and events[1].scorer == "R.Back"
    bare = nfl.normalize_plays([row(1, 2), row(2, 8), row(3, 10), row(4, 13), row(5, 15)], teams)
    assert [e.kind for e in bare] == ["safety", "touchdown", "two_point", "field_goal", "safety"], "context when the columns are missing"


def test_nfl_clock_under_a_minute_and_quarter_text():
    assert NFL.parse_field("clock", ":05") == 5.0 and NFL.parse_field("clock", " :48 ") == 48.0
    assert NFL.parse_field("clock", "12:20") == 740.0 and NFL.parse_field("clock", "15:00") == 900.0
    assert NFL.parse_field("period", "4th") == 4 and NFL.parse_field("period", "OT") == 5
    assert NFL.parse_field("shot_clock", "40") == 40.0
    assert NFL.parse_field("down_distance", "3rd & 7") == "3rd & 7"
    assert [NFL.format_period(p) for p in (1, 4, 5)] == ["Q1", "Q4", "OT"]


def test_nfl_classify_by_points_and_context():
    assert [NFL.classify(score_change(p)) for p in (6, 7, 8, 3, 1)] == [
        "touchdown", "touchdown", "touchdown", "field_goal", "extra_point",
    ]
    assert NFL.classify(score_change(2, clock_stopped=True)) == "two_point"
    assert NFL.classify(score_change(2, clock_stopped=False)) == "safety"
    # play-by-play knows better than the clock: it is stopped after a safety as well
    assert NFL.classify(score_change(2, clock_stopped=True, extra={"pbp": [{"kind": "safety"}]})) == "safety"
    assert NFL.classify(score_change(2, clock_stopped=False, extra={"pbp": [{"kind": "two_point"}]})) == "two_point"


def test_nfl_try_rides_on_its_touchdown():
    touchdown = score_change(6, clock=735.0, t=100.0)
    assert NFL.tail_of(touchdown, score_change(1, clock=735.0, t=140.0))
    assert NFL.tail_of(touchdown, score_change(2, clock=734.0, t=140.0))
    assert not NFL.tail_of(touchdown, score_change(3, clock=735.0, t=140.0))
    assert not NFL.tail_of(touchdown, score_change(2, clock=600.0, t=400.0)), "the clock moved: a safety, not the try"
    assert not NFL.tail_of(touchdown, score_change(1, clock=735.0, t=140.0, team="away"))
    assert not NFL.tail_of(touchdown, score_change(1, clock=735.0, t=140.0, period=3))
    assert not NFL.tail_of(score_change(3, clock=735.0), score_change(1, clock=735.0, t=140.0))
    # clock unreadable: go by how soon the points followed
    assert NFL.tail_of(score_change(6, clock=None, t=100.0), score_change(1, clock=None, t=150.0))
    assert not NFL.tail_of(score_change(6, clock=None, t=100.0), score_change(2, clock=None, t=400.0))


def nfl_touchdown_drive():
    """Q4 of Super Bowl LI as the bug showed it: NE 12-28, Amendola's touchdown snapped at
    6:00, the 18 on the bug at t=18, the two-point try good at t=55."""
    q4 = {"period": "4th", "home": 28}
    rows = [{"clock": "6:00", "away": 12, **q4}] * 20  # t 0-9.5: set at 6:00
    rows += running(359.5, 8, away=12, **q4)  # t 10-13.5: the snap; the clock runs to 5:56
    rows += [{"clock": "5:56", "away": 12, **q4}] * 8  # t 14-17.5: touchdown, clock stopped
    rows += [{"clock": "5:56", "away": 18, **q4}] * 74  # t 18-54.5
    rows += [{"clock": "5:56", "away": 20, **q4}] * 20  # t 55-64.5
    return make_raw(rows)


def test_nfl_clip_starts_at_the_snap_from_play_by_play_and_carries_the_try(nflverse):
    pbp = NFL.fetch_pbp(NFL_GAME)
    tl, events, clips = cut(NFL, nfl_touchdown_drive(), "away", pbp)
    assert [(e.t, e.points, e.clock) for e in events] == [(18.0, 6, 356.0), (55.0, 2, 356.0)]
    touchdown, two_point = events
    assert touchdown.extra["pbp"][0]["extra"]["snap_clock"] == 360.0, "matched although the bug clock is 4 s past the snap"
    assert NFL.possession_start(tl, touchdown) == (9.5, "snap_pbp"), "the last sample still showing 6:00"
    assert NFL.classify(two_point) == "two_point"
    (clip,) = clips
    assert (clip.kind, clip.points, clip.start_cause) == ("touchdown", 8, "snap_pbp")
    assert clip.segments == [[9.5 - 2.0, 18.0 + 2.0], [55.0 - 3.0, 55.0 + 1.0]], "2 s pre-roll; the try as a trailing segment"
    assert clip.scorer == "D.Amendola / J.White" and "TWO-POINT CONVERSION" in clip.description
    assert not clip.warnings and (clip.score_before, clip.score_after) == (12, 20)

    _, _, without_try = cut(NFL, nfl_touchdown_drive(), "away", pbp, {"include_tail": False})
    assert [(c.points, c.segments) for c in without_try] == [(6, [[7.5, 20.0]])]


def test_nfl_touchdown_then_extra_point_is_one_clip_without_play_by_play():
    q2 = {"period": "2nd", "away": 0}
    rows = [{"clock": "12:20", "home": 0, **q2}] * 20  # t 0-9.5
    rows += running(739.5, 10, home=0, **q2)  # t 10-14.5: the snap; 12:15 when he scores
    rows += [{"clock": "12:15", "home": 0, **q2}] * 12  # t 15-20.5
    rows += [{"clock": "12:15", "home": 6, **q2}] * 58  # t 21: the 6
    rows += [{"clock": "12:15", "home": 7, **q2}] * 20  # t 50: the extra point
    tl, events, clips = cut(NFL, make_raw(rows))
    assert [(e.t, e.points, e.clock_stopped) for e in events] == [(21.0, 6, True), (50.0, 1, True)]
    snap = tl.clock_starts[-1].t
    assert 9.0 <= snap <= 10.0
    (clip,) = clips
    assert (clip.kind, clip.points, clip.start_cause) == ("touchdown", 7, "clock_start")
    assert clip.segments == [[pytest.approx(snap - 2.0), 23.0], [47.0, 51.0]]
    assert [c.points for c in clip.changes] == [6, 1]


def test_nfl_lone_field_goal_is_its_own_clip(nflverse):
    pbp = NFL.fetch_pbp(NFL_GAME)
    q4 = {"period": "4th", "home": 28}
    rows = [{"clock": "9:48", "away": 9, **q4}] * 16  # t 0-7.5
    rows += running(587.5, 8, away=9, **q4)  # t 8-11.5: snap and kick
    rows += [{"clock": "9:44", "away": 9, **q4}] * 8  # t 12-15.5
    rows += [{"clock": "9:44", "away": 12, **q4}] * 30  # t 16: the 3 points
    _, events, clips = cut(NFL, make_raw(rows), "away", pbp)
    (clip,) = clips
    assert (clip.kind, clip.points, clip.scorer, clip.start_cause) == ("field_goal", 3, "S.Gostkowski", "snap_pbp")
    assert clip.segments == [[7.5 - 2.0, 16.0 + 2.0]] and len(clip.changes) == 1


def test_nfl_falls_back_to_12s_when_the_clock_started_long_before():
    rows = [{"clock": "13:20"}] * 10  # t 0-4.5
    rows += running(799.5, 60)  # t 5-34.5: one long play, as far as the clock can tell
    rows += [{"clock": "12:50", "home": 13}] * 20  # t 35: three points
    tl, events, clips = cut(NFL, make_raw(rows))
    assert len(tl.clock_starts) == 1 and events[0].t - tl.clock_starts[0].t > 20.0
    assert NFL.possession_start(tl, events[0]) == (35.0 - 12.0, "12s_before")
    assert clips[0].kind == "field_goal" and clips[0].segments == [[21.0, 37.0]]
