"""ESPN as the NBA data source when NBA.com will not answer (it refuses cloud servers)."""

from __future__ import annotations

import json

import pytest

from possession_cut.pipeline.window import biggest_run_start
from possession_cut.sports import get_adapter, nba
from possession_cut.sports import http as sport_http
from possession_cut.sports.base import PlayByPlayUnavailable
from possession_cut.sports.nba import (
    ESPN_SCOREBOARD,
    ESPN_SUMMARY,
    STATS_SCOREBOARD,
    games_from_espn,
    normalize_actions,
    normalize_espn_plays,
)

from .conftest import FIXTURES

NBA = get_adapter("nba")
EVENT = "401859966"


def fixture(name: str):
    return json.loads((FIXTURES / "nba" / name).read_text(encoding="utf-8"))


@pytest.fixture()
def net(monkeypatch, settings):
    """Recorded answers; NBA.com can be made to stall like it does from a server."""
    calls: list[str] = []
    nba_com_down = {"on": False}

    def fake_get_json(url, params=None, headers=None, **_kw):
        calls.append(url)
        if "nba.com" in url and nba_com_down["on"]:
            raise PlayByPlayUnavailable(f"{url} failed after 1 attempts (ReadTimeout)")
        if url == STATS_SCOREBOARD:
            return fixture("stats_scoreboardv3_2026-06-10.json") if params["GameDate"] == "2026-06-10" else {"scoreboard": {"games": []}}
        if url == nba.CDN_PBP.format(game_id="0042500404"):
            return fixture("cdn_playbyplay_0042500404.json")
        if url == ESPN_SCOREBOARD:
            return fixture("espn_scoreboard_2026-06-10.json") if params["dates"] == "20260610" else {"events": []}
        if url == ESPN_SUMMARY and params["event"] == EVENT:
            return fixture(f"espn_summary_{EVENT}.json")
        raise PlayByPlayUnavailable(f"{url} returned 404")

    monkeypatch.setattr(sport_http, "get_json", fake_get_json)
    monkeypatch.setattr(nba, "_nba_com_down_until", 0.0)
    return type("Net", (), {"calls": calls, "down": nba_com_down})


def test_espn_scoreboard_gives_the_same_game_with_nba_team_codes():
    (game,) = games_from_espn(fixture("espn_scoreboard_2026-06-10.json"), "2026-06-10")
    assert game.game_id == f"espn:{EVENT}"
    assert (game.away, game.home) == ("SAS", "NYK"), "ESPN says SA and NY; the rest of the app speaks NBA codes"
    assert (game.away_name, game.home_name) == ("San Antonio Spurs", "New York Knicks")
    assert (game.away_score, game.home_score) == (106, 107) and game.status == "Final"
    assert game.label == "NBA Finals - Game 4"


def test_espn_plays_agree_with_nba_com_play_for_play():
    espn = normalize_espn_plays(fixture(f"espn_summary_{EVENT}.json"))
    official = normalize_actions(fixture("cdn_playbyplay_0042500404.json")["game"]["actions"])
    assert len(espn) == len(official) == 109
    for a, b in zip(espn, official, strict=True):
        assert (a.period, a.team, a.points, a.score_away, a.score_home, a.kind) == (b.period, b.team, b.points, b.score_away, b.score_home, b.kind)
        assert abs((a.clock or 0) - (b.clock or 0)) <= 1.0, (a.description, b.description)
        assert a.extra["side"] == b.extra["side"]
    towns = next(e for e in espn if "Towns" in e.description)
    assert towns.scorer == "K. Towns" and towns.description.startswith("Karl-Anthony Towns makes")
    assert espn[-1].scorer == "O. Anunoby" and espn[-1].clock == pytest.approx(2.1) and espn[-1].kind == "field_goal"
    assert [e.event_id for e in espn] == sorted({e.event_id for e in espn}, key=[e.event_id for e in espn].index), "ids are unique"


def test_auto_start_from_espn_matches_nba_com():
    espn = biggest_run_start(normalize_espn_plays(fixture(f"espn_summary_{EVENT}.json")), "home")
    official = biggest_run_start(normalize_actions(fixture("cdn_playbyplay_0042500404.json")["game"]["actions"]), "home")
    assert espn is not None and (espn.deficit, espn.period, espn.score_away, espn.score_home) == (29, 3, 81, 52)
    assert (espn.deficit, espn.period, espn.clock) == (official.deficit, official.period, official.clock)


def test_lookup_falls_back_to_espn_when_nba_com_stalls(net):
    net.down["on"] = True
    games = NBA.find_games("2026-06-10")
    assert [g.game_id for g in games] == [f"espn:{EVENT}"]
    assert STATS_SCOREBOARD in net.calls and ESPN_SCOREBOARD in net.calls
    assert NBA.find_games("2026-06-10", team="NYK") and not NBA.find_games("2026-06-10", team="BOS"), "team filter works on NBA codes"

    # NBA.com is remembered as down: the next lookup goes straight to ESPN
    net.calls.clear()
    assert NBA.find_games("2026-06-11") == []
    assert STATS_SCOREBOARD not in net.calls and ESPN_SCOREBOARD in net.calls

    # and play-by-play for an ESPN game comes from ESPN, once
    events = NBA.fetch_pbp(f"espn:{EVENT}")
    assert len(events) == 109 and events[-1].score_home == 107
    net.calls.clear()
    assert len(NBA.fetch_pbp(f"espn:{EVENT}")) == 109 and net.calls == [], "cached"


def test_nba_com_is_preferred_when_it_answers(net):
    games = NBA.find_games("2026-06-10")
    assert [g.game_id for g in games] == ["0042500404"]
    assert ESPN_SCOREBOARD not in net.calls
    assert len(NBA.fetch_pbp("0042500404")) == 109


def test_an_nba_com_game_without_nba_com_fails_clearly(net):
    net.down["on"] = True
    NBA.find_games("2026-06-10")  # marks NBA.com down
    with pytest.raises(PlayByPlayUnavailable, match="pick the game again"):
        NBA.fetch_pbp("0042500404")
