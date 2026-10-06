"""The "defensive plays" option: blocks and steals from play-by-play as short clips."""

from __future__ import annotations

import json

import pytest

from possession_cut.pipeline.clips import DEFENSE_AFTER, DEFENSE_BEFORE, build_clips
from possession_cut.sports import get_adapter, nba
from possession_cut.sports import http as sport_http
from possession_cut.sports.base import PlayByPlayUnavailable, ScoringEvent
from possession_cut.sports.nba import ESPN_SUMMARY, normalize_defense, normalize_espn_defense
from possession_cut.synth.script import AWAY, HOME, format_period

from .conftest import FIXTURES
from .helpers import run_logic

NBA = get_adapter("nba")


def fixture(name: str):
    return json.loads((FIXTURES / "nba" / name).read_text(encoding="utf-8"))


# -- from each source ------------------------------------------------------------------------------


def test_blocks_and_steals_from_cdn_and_stats_agree():
    cdn = normalize_defense(fixture("cdn_playbyplay_0042500404.json")["game"]["actions"])
    stats = normalize_defense(fixture("stats_playbyplayv3_0042500404.json")["game"]["actions"])
    assert len(cdn) == len(stats) == 24
    assert sum(1 for e in cdn if e.kind == "block") == 8 and sum(1 for e in cdn if e.kind == "steal") == 16
    assert all(e.points == 0 and e.team in ("NYK", "SAS") for e in cdn)
    for a, b in zip(cdn, stats, strict=True):
        assert (a.kind, a.team, a.period, a.clock) == (b.kind, b.team, b.period, b.clock)
    first = cdn[0]
    assert (first.kind, first.team, first.period, first.clock, first.scorer) == ("steal", "SAS", 1, 560.0, "J. Champagnie")
    assert first.description == "J. Champagnie STEAL (1 STL)"
    knicks = [e for e in cdn if e.team == "NYK"]
    assert sum(1 for e in knicks if e.kind == "block") == 4 and sum(1 for e in knicks if e.kind == "steal") == 6


def test_blocks_and_steals_from_espn_match_nba_com():
    espn = normalize_espn_defense(fixture("espn_summary_401859966.json"))
    cdn = normalize_defense(fixture("cdn_playbyplay_0042500404.json")["game"]["actions"])
    assert len(espn) == 24
    for a, b in zip(espn, cdn, strict=True):
        assert (a.kind, a.team, a.period) == (b.kind, b.team, b.period), (a.description, b.description)
        assert abs((a.clock or 0) - (b.clock or 0)) <= 1.0
    assert espn[0].description == "J. Champagnie STEAL" and espn[0].scorer == "J. Champagnie"
    assert len({e.event_id for e in espn}) == 24
    assert not {e.event_id for e in espn} & {e.event_id for e in nba.normalize_espn_plays(fixture("espn_summary_401859966.json"))}


@pytest.fixture()
def net(monkeypatch, settings):
    calls: list[str] = []

    def fake_get_json(url, params=None, headers=None, **_kw):
        calls.append(url)
        if url == nba.CDN_PBP.format(game_id="0042500404"):
            return fixture("cdn_playbyplay_0042500404.json")
        if url == ESPN_SUMMARY and params["event"] == "401859966":
            return fixture("espn_summary_401859966.json")
        raise PlayByPlayUnavailable(f"{url} returned 404")

    monkeypatch.setattr(sport_http, "get_json", fake_get_json)
    monkeypatch.setattr(nba, "_nba_com_down_until", 0.0)
    return calls


def test_defense_is_fetched_once_with_the_play_by_play(net):
    adapter = get_adapter("nba")
    assert len(adapter.fetch_pbp("0042500404")) == 109
    assert len(adapter.fetch_defense("0042500404")) == 24
    assert len(net) == 1, "one download covers scores and defence"
    net.clear()
    assert len(get_adapter("nba").fetch_defense("0042500404")) == 24 and net == [], "and it is cached"

    assert len(get_adapter("nba").fetch_defense("espn:401859966")) == 24
    assert net == [ESPN_SUMMARY]


# -- in the cut ------------------------------------------------------------------------------------


def steals_in(script, team: str) -> list[tuple[float, str, ScoringEvent]]:
    """The scripted steals by ``team``, as the analysis would hand them to the clip builder."""
    out = []
    for t, who, cause in script.possession_starts:
        if who == team and cause == "steal":
            play = ScoringEvent(
                event_id=f"s{int(t)}", period=script.period_track.at(t) or 0, clock=script.clock_track.at(t),
                team=script.teams[team]["abbr"], points=0, scorer="A. Player", description="A. Player STEAL", kind="steal",
            )
            out.append((t, team, play))
    return out


def test_a_steal_that_leads_to_a_score_joins_that_clip(coverage_script):
    tl, events, plain = run_logic(coverage_script, HOME)
    steals = steals_in(coverage_script, HOME)
    assert len(steals) == 1
    t_steal = steals[0][0]
    with_defense = build_clips(tl, events, NBA, HOME, {"include_defense": True}, defense=steals)
    assert len(with_defense) == len(plain), "the steal led to the and-one: one clip, not two"
    clip = next(c for c in with_defense if c.src_in <= t_steal <= c.src_out)
    assert clip.kind == "field_goal" and clip.points == 3, "the score outranks the steal that set it up"
    assert clip.src_in == pytest.approx(t_steal - DEFENSE_BEFORE, abs=0.01) and clip.start_cause == "defense"
    assert len(clip.changes) == 3 and clip.changes[0].points == 0


def test_a_steal_with_no_score_after_it_stands_alone(coverage_script):
    tl, events, plain = run_logic(coverage_script, AWAY)
    steals = steals_in(coverage_script, AWAY)
    assert len(steals) == 1
    t_steal, _, play = steals[0]
    clips = build_clips(tl, events, NBA, AWAY, {"include_defense": True}, defense=steals)
    assert len(clips) == len(plain) + 1
    clip = next(c for c in clips if c.kind == "steal")
    assert clip.points == 0 and clip.scorer == "A. Player" and clip.description == "A. Player STEAL"
    assert clip.segments == [[round(t_steal - DEFENSE_BEFORE, 3), round(t_steal + DEFENSE_AFTER, 3)]]
    assert clip.period == play.period and clip.clock == play.clock
    assert clip.score_away == clip.score_after and clip.confidence >= 0.9
    assert clip.pbp_event_id == play.event_id
    assert format_period(clip.period)  # a period the review screen can label

    # outside the window, or for the other team, nothing is added
    assert len(build_clips(tl, events, NBA, AWAY, {}, t_max=t_steal - 5, defense=steals)) < len(clips)
    assert len(build_clips(tl, events, NBA, HOME, {}, defense=steals)) == len(run_logic(coverage_script, HOME)[2])
