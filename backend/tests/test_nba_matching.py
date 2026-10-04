"""NBA adapter against recorded responses (2026 Finals Game 4, SAS @ NYK), play-by-play
matching, and the start/end window including "auto: start of biggest run"."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from possession_cut.pipeline.clips import build_clips
from possession_cut.pipeline.matching import (
    label_clips,
    load_sidecar,
    match_events,
    sidecar_path,
    sides_of,
)
from possession_cut.pipeline.window import biggest_run_start, resolve_window, time_at_game_time
from possession_cut.sports import PlayByPlayUnavailable, ScoringEvent, get_adapter
from possession_cut.sports import http as sport_http
from possession_cut.sports.nba import (
    CDN_PBP,
    STATS_PBP,
    STATS_SCOREBOARD,
    normalize_actions,
    teams_from_events,
)
from possession_cut.synth.script import AWAY, HOME, displayed_clock_seconds, random_game

from .conftest import FIXTURES
from .helpers import run_logic

GAME_ID = "0042500404"
NBA = get_adapter("nba")


def fixture(name: str):
    return json.loads((FIXTURES / "nba" / name).read_text(encoding="utf-8"))


@pytest.fixture()
def nba_net(monkeypatch, settings):
    """Serve the recorded responses instead of the network; count the calls."""
    calls: list[str] = []
    fail: set[str] = set()

    def fake_get_json(url, params=None, headers=None, **_kw):
        calls.append(url)
        if any(tag in url for tag in fail):
            raise PlayByPlayUnavailable(f"{url} is down (test)")
        if url == STATS_SCOREBOARD:
            assert params["LeagueID"] == "00"
            if params["GameDate"] == "2026-06-10":
                return fixture("stats_scoreboardv3_2026-06-10.json")
            return {"scoreboard": {"games": []}}
        if url == CDN_PBP.format(game_id=GAME_ID):
            return fixture(f"cdn_playbyplay_{GAME_ID}.json")
        if url == STATS_PBP and params["GameID"] == GAME_ID:
            return fixture(f"stats_playbyplayv3_{GAME_ID}.json")
        raise PlayByPlayUnavailable(f"{url} returned 404")

    monkeypatch.setattr(sport_http, "get_json", fake_get_json)
    return type("Net", (), {"calls": calls, "fail": fail})


def test_parse_iso_clock():
    assert sport_http.parse_iso_clock("PT11M42.00S") == 702.0
    assert sport_http.parse_iso_clock("PT00M02.10S") == pytest.approx(2.1)
    assert sport_http.parse_iso_clock("9:40") == 580.0
    assert sport_http.parse_iso_clock("") is None and sport_http.parse_iso_clock(None) is None


def test_game_lookup_by_date_and_team(nba_net):
    games = NBA.find_games("2026-06-10")
    assert len(games) == 1
    g = games[0]
    assert (g.game_id, g.away, g.home) == (GAME_ID, "SAS", "NYK")
    assert (g.away_score, g.home_score) == (106, 107) and g.status == "Final"
    assert "NBA Finals" in g.label
    assert NBA.find_games("2026-06-10", team="nyk") == games
    assert NBA.find_games("2026-06-10", team="BOS") == []
    assert NBA.find_games("2026-06-11") == []
    # a finished slate is cached: the second lookup does not hit the network
    before = len(nba_net.calls)
    NBA.find_games("2026-06-10", team="SAS")
    assert len(nba_net.calls) == before


def test_play_by_play_is_normalized_and_cached(nba_net):
    events = NBA.fetch_pbp(GAME_ID)
    assert teams_from_events(events) == {"away": "SAS", "home": "NYK"}
    assert events[-1].score_away == 106 and events[-1].score_home == 107
    assert sum(e.points for e in events if e.team == "NYK") == 107
    assert sum(e.points for e in events if e.team == "SAS") == 106
    assert {e.points for e in events} == {1, 2, 3}
    first = events[0]
    assert (first.period, first.clock, first.team, first.points, first.kind) == (1, 702.0, "SAS", 1, "free_throw")
    assert first.scorer == "D. Fox" and "Free Throw 1 of 2" in first.description
    last = events[-1]
    assert (last.period, last.team, last.points) == (4, "NYK", 2) and last.clock == pytest.approx(2.1)
    assert "Anunoby" in last.description
    # scores never go down and every event moves exactly one team
    for a, b in zip(events, events[1:], strict=False):
        assert b.score_home >= a.score_home and b.score_away >= a.score_away
        assert (b.score_home - a.score_home) + (b.score_away - a.score_away) == b.points

    n = len(nba_net.calls)
    again = NBA.fetch_pbp(GAME_ID)
    assert len(nba_net.calls) == n, "served from the per-game disk cache"
    assert [e.to_dict() for e in again] == [e.to_dict() for e in events]


def test_play_by_play_falls_back_to_stats_when_the_cdn_is_down(nba_net):
    nba_net.fail.add("cdn.nba.com")
    events = NBA.fetch_pbp(GAME_ID)
    assert any("stats.nba.com" in url for url in nba_net.calls)
    assert events[-1].score_away == 106 and events[-1].score_home == 107
    assert sum(e.points for e in events if e.team == "NYK") == 107


def test_both_sources_agree_on_every_scoring_play():
    cdn = normalize_actions(fixture(f"cdn_playbyplay_{GAME_ID}.json")["game"]["actions"])
    stats = normalize_actions(fixture(f"stats_playbyplayv3_{GAME_ID}.json")["game"]["actions"])
    key = lambda e: (e.period, e.clock, e.team, e.points, e.score_away, e.score_home)  # noqa: E731
    assert [key(e) for e in cdn] == [key(e) for e in stats]


def test_unavailable_play_by_play_raises_cleanly(nba_net, monkeypatch):
    monkeypatch.setattr(type(NBA), "_nba_api_pbp", staticmethod(lambda game_id: (_ for _ in ()).throw(TimeoutError())))
    with pytest.raises(PlayByPlayUnavailable, match="unavailable"):
        NBA.fetch_pbp("0020000001")


def test_team_list_is_available_offline():
    teams = NBA.teams()
    assert len(teams) == 30 and {"abbr": "NYK", "name": "New York Knicks"} in teams


def test_auto_start_finds_the_29_point_hole(nba_net):
    events = NBA.fetch_pbp(GAME_ID)
    run = biggest_run_start(events, "home")
    assert run is not None and run.deficit == 29
    assert (run.period, run.clock) == (3, 580.0), "Q3 9:40, SAS 81 NYK 52"
    assert (run.score_away, run.score_home) == (81, 52)
    after = [e for e in events if e.team == "NYK" and (e.score_home or 0) > 52]
    assert sum(e.points for e in after) == 55, "the comeback: 55 Knicks points from there"
    assert biggest_run_start(events, "away") is not None  # the Spurs trailed at the very end
    never_behind = [replace(e) for e in events if e.team == "SAS"]
    assert biggest_run_start(never_behind, "away") is None


# -- matching ------------------------------------------------------------------------


def script_pbp(script) -> list[ScoringEvent]:
    events = [ScoringEvent.from_dict(row) for row in script.play_by_play()]
    for ev in events:
        ev.extra["side"] = "home" if ev.team == script.teams["home"]["abbr"] else "away"
    return events


def test_every_detected_event_matches_its_play(coverage_script):
    _, events, clips = run_logic(coverage_script, HOME)
    pbp = script_pbp(coverage_script)
    assert sides_of(pbp) == {"away": "SA", "home": "NY"}
    result = match_events(events, pbp)
    assert len(result.matches) == len(events) and not result.unmatched_changes and not result.unmatched_pbp
    assert all(m.how == "clock" and m.clock_diff <= 3.0 for m in result.matches.values())
    label_clips(clips, result)
    for clip in clips:
        assert clip.scorer and clip.description and clip.pbp_event_id
        assert not clip.warnings
    and_one = clips[3]
    assert "Free Throw" in and_one.description and " + " in and_one.description


def test_unmatched_events_are_reported_in_both_directions(coverage_script):
    tl, events, _ = run_logic(coverage_script, HOME)
    pbp = script_pbp(coverage_script)
    dropped = pbp.pop(4)  # play-by-play is missing a play the bug saw
    missed = events.pop(9)  # and the bug missed a play that play-by-play has
    result = match_events(events, pbp)
    assert len(result.unmatched_pbp) == 1
    assert (result.unmatched_pbp[0].score_away, result.unmatched_pbp[0].score_home) == (missed.score_away, missed.score_home)
    assert len(result.unmatched_changes) == 1
    assert result.unmatched_changes[0].score_home == dropped.score_home
    clips = build_clips(tl, events, NBA, HOME)
    label_clips(clips, result)
    flagged = [c for c in clips if any("no play-by-play match" in w for w in c.warnings)]
    assert len(flagged) == 1 and flagged[0].confidence < 0.8
    assert flagged[0].score_before < dropped.score_home <= flagged[0].score_after


def test_misread_score_still_matches_by_period_clock_team_points(coverage_script):
    _, events, _ = run_logic(coverage_script, HOME)
    pbp = script_pbp(coverage_script)
    # the bug's running score is off by ten for one event; period, clock, team, points still agree
    events[0] = replace(events[0], score_before=events[0].score_before + 10, score_after=events[0].score_after + 10)
    result = match_events(events, pbp)
    assert events[0].index in result.matches and result.matches[events[0].index].events[0].event_id == pbp[0].event_id


def test_free_throws_read_as_one_change_match_both_plays(coverage_script):
    _, events, _ = run_logic(coverage_script, HOME)
    pbp = script_pbp(coverage_script)
    fts = [e for e in events if e.team == "home" and e.points == 1][:2]
    merged = replace(fts[1], points=2, score_before=fts[0].score_before)
    rest = [e for e in events if e.index not in (fts[0].index, fts[1].index)] + [merged]
    result = match_events(sorted(rest, key=lambda e: e.t), pbp)
    m = result.matches[merged.index]
    assert m.how == "merged" and len(m.events) == 2 and not result.unmatched_pbp


def test_sidecar_play_by_play(coverage_video):
    path, truth = coverage_video
    assert sidecar_path(path).exists()
    events, sides = load_sidecar(path)
    assert sides == {"away": "SA", "home": "NY"} and len(events) == len(truth["score_events"])
    assert all(e.extra.get("side") in ("away", "home") for e in events)
    assert load_sidecar(str(path) + ".nope") is None


# -- start / end window ----------------------------------------------------------------


def test_game_time_start_and_end(coverage_script):
    tl, events, _ = run_logic(coverage_script, HOME)
    t = time_at_game_time(tl, NBA, 3, 60.0)
    state = coverage_script.state_at(t)
    shown = displayed_clock_seconds
    assert state.period == 3 and shown(state.clock) <= 60.0 and shown(coverage_script.state_at(t - 0.5).clock) > 60.0
    w = resolve_window(tl, events, NBA, "home", {"mode": "game_time", "period": 3, "clock": 60.0}, {"mode": "end"})
    clips = build_clips(tl, events, NBA, "home", t_min=w.t_min, t_max=w.t_max)
    assert [c.score_after for c in clips] == [100, 103, 105, 107, 109] and w.start_label == "Q3 1:00"

    w2 = resolve_window(tl, events, NBA, "home", {"mode": "start"}, {"mode": "game_time", "period": 3, "clock": 60.0})
    clips2 = build_clips(tl, events, NBA, "home", t_min=w2.t_min, t_max=w2.t_max)
    assert [c.score_after for c in clips2] == [90, 93, 95, 98] and w2.end_label == "Q3 1:00"

    missing = resolve_window(tl, events, NBA, "home", {"mode": "game_time", "period": 1, "clock": 300.0}, None)
    assert missing.t_min is not None and missing.t_min <= tl.t[2], "a time before the file starts means from the beginning"


@pytest.mark.parametrize("seed", [2, 5, 9])
def test_auto_run_start_from_play_by_play_and_from_the_bug_agree(seed):
    g = random_game(seed)
    pbp = script_pbp(g)
    for team in (HOME, AWAY):
        tl, events, _ = run_logic(g, team)
        result = match_events(events, pbp)
        with_pbp = resolve_window(tl, events, NBA, team, {"mode": "auto_run"}, None, pbp, result)
        bug_only = resolve_window(tl, events, NBA, team, {"mode": "auto_run"}, None)
        run = biggest_run_start(pbp, team)
        if run is None:
            assert with_pbp.t_min is None and bug_only.t_min is None
            continue
        assert with_pbp.t_min == pytest.approx(bug_only.t_min), "same anchor either way"
        assert with_pbp.run_start["deficit"] == bug_only.run_start["deficit"] == run.deficit
        clips = build_clips(tl, events, NBA, team, t_min=with_pbp.t_min)
        own_before = run.score_home if team == HOME else run.score_away
        final = g.events[-1].score_home if team == HOME else g.events[-1].score_away
        assert sum(c.points for c in clips) == final - own_before, "every score after the low point is in the cut"
        if clips:  # the low point can be the last basket of the game
            assert clips[0].score_before == own_before, "and the first clip is the first score after it"
        assert "Biggest run: down" in with_pbp.start_label
