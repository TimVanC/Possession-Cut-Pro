"""Scripted basketball game model for the synthetic broadcast.

A ``ScriptBuilder`` plays out possessions and records exactly what a broadcast score
bug would show at every instant: game clock, shot clock, both scores, the period,
when the bug is hidden (commercials, some replays), when a replay re-airs an earlier
game state, and when a score animation covers the score.

Because the script knows the truth, it also emits the ground-truth cut list the
pipeline is graded against. Nothing in the pipeline imports this module.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import asdict, dataclass, field

AWAY, HOME = "away", "home"

DEFAULT_TEAMS = {
    AWAY: {
        "abbr": "SA",
        "name": "Spurs",
        "city": "San Antonio",
        "roster": ["V. Wembanyama", "D. Fox", "S. Castle", "D. Vassell", "H. Barnes"],
    },
    HOME: {
        "abbr": "NY",
        "name": "Knicks",
        "city": "New York",
        "roster": ["J. Brunson", "K. Towns", "M. Bridges", "O. Anunoby", "J. Hart"],
    },
}


class ScriptError(ValueError):
    pass


def other(team: str) -> str:
    return HOME if team == AWAY else AWAY


# -- how a bug displays values --------------------------------------------


def format_clock(seconds: float) -> str:
    """Game clock as a score bug shows it: M:SS, or SS.t under a minute."""
    s = max(0.0, seconds)
    if s >= 60.0:
        whole = int(math.floor(s + 1e-6))
        return f"{whole // 60}:{whole % 60:02d}"
    tenths = int(math.floor(s * 10 + 1e-6))
    return f"{tenths // 10}.{tenths % 10}"


def format_shot_clock(value: float | None) -> str:
    """Shot clock rounds up, so it reads 24 for the first second."""
    if value is None:
        return ""
    return str(int(math.ceil(max(value, 0.0) - 1e-6)))


def format_period(period: int) -> str:
    if period <= 4:
        return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[period]
    return "OT" if period == 5 else f"{period - 4}OT"


def displayed_clock_seconds(seconds: float) -> float:
    """The numeric value of what ``format_clock`` shows."""
    s = max(0.0, seconds)
    if s >= 60.0:
        return float(math.floor(s + 1e-6))
    return math.floor(s * 10 + 1e-6) / 10.0


# -- tracks -----------------------------------------------------------------


class LinearTrack:
    """Piecewise-linear value: keyframes of (t, value, rate)."""

    def __init__(self) -> None:
        self.t: list[float] = []
        self.v: list[float | None] = []
        self.rate: list[float] = []

    def add(self, t: float, value: float | None, rate: float) -> None:
        if self.t and t < self.t[-1] - 1e-9:
            raise ScriptError("track keyframes must be added in time order")
        if self.t and abs(t - self.t[-1]) < 1e-9:
            self.v[-1], self.rate[-1] = value, rate
            return
        self.t.append(t)
        self.v.append(value)
        self.rate.append(rate)

    def at(self, t: float) -> float | None:
        i = bisect.bisect_right(self.t, t) - 1
        if i < 0:
            return None
        v = self.v[i]
        if v is None:
            return None
        return max(0.0, v + self.rate[i] * (t - self.t[i]))


class StepTrack:
    def __init__(self) -> None:
        self.items: list[tuple[float, object]] = []

    def add(self, t: float, value: object) -> None:
        self.items.append((t, value))

    def finalize(self) -> None:
        self.items.sort(key=lambda kv: kv[0])
        self._t = [kv[0] for kv in self.items]

    def at(self, t: float):
        i = bisect.bisect_right(self._t, t) - 1
        return None if i < 0 else self.items[i][1]


# -- records ----------------------------------------------------------------


@dataclass
class ScoreEvent:
    index: int
    team: str
    points: int
    kind: str  # "fg" | "ft"
    make_time: float
    visible_time: float  # when the bug first shows the new score
    period: int
    clock: float  # game clock at the make, seconds remaining
    score_away: int
    score_home: int
    possession_start: float | None = None
    start_cause: str | None = None
    # False when the bug gives no signal for this possession change (shot clock off before
    # and after, so nothing jumps): the PRD's start rule cannot see it.
    start_observable: bool = True
    trip: int | None = None
    and_one_of: int | None = None
    scorer: str = ""
    description: str = ""


@dataclass
class BugState:
    scene: str  # live | commercial | replay_hidden | replay_bug
    visible: bool
    period: int | None = None
    period_text: str = ""
    clock: float | None = None
    clock_text: str = ""
    shot_clock: float | None = None
    shot_text: str = ""
    away_score: int = 0
    home_score: int = 0
    anim_team: str | None = None
    anim_progress: float = 0.0
    source_time: float = 0.0  # the game moment being shown (earlier than t in a replay)


@dataclass
class GameScript:
    seed: int
    teams: dict
    duration: float
    clock_track: LinearTrack
    shot_track: LinearTrack
    score_track: StepTrack
    period_track: StepTrack
    hidden: list[tuple[float, float, str]]
    replays: list[tuple[float, float, float]]
    anims: list[tuple[float, float, str]]
    events: list[ScoreEvent]
    possession_starts: list[tuple[float, str, str]]
    make_times: list[float] = field(default_factory=list)

    # -- what the bug shows --------------------------------------------
    def _base_state(self, t: float) -> BugState:
        period = self.period_track.at(t)
        clock = self.clock_track.at(t)
        shot = self.shot_track.at(t)
        away, home = self.score_track.at(t) or (0, 0)
        anim_team, progress = None, 0.0
        for a0, a1, team in self.anims:
            if a0 <= t < a1:
                anim_team, progress = team, (t - a0) / max(a1 - a0, 1e-6)
                break
        return BugState(
            scene="live",
            visible=True,
            period=period,
            period_text=format_period(period) if period else "",
            clock=clock,
            clock_text=format_clock(clock) if clock is not None else "",
            shot_clock=shot,
            shot_text=format_shot_clock(shot),
            away_score=away,
            home_score=home,
            anim_team=anim_team,
            anim_progress=progress,
            source_time=t,
        )

    def state_at(self, t: float) -> BugState:
        for h0, h1, kind in self.hidden:
            if h0 <= t < h1:
                return BugState(scene=kind, visible=False, source_time=t)
        for r0, r1, src0 in self.replays:
            if r0 <= t < r1:
                state = self._base_state(src0 + (t - r0))
                state.scene = "replay_bug"
                return state
        return self._base_state(t)

    def not_live_intervals(self) -> list[tuple[float, float]]:
        spans = [(a, b) for a, b, _ in self.hidden] + [(a, b) for a, b, _ in self.replays]
        return _merge_intervals(spans)

    def is_live(self, t: float) -> bool:
        return self.state_at(t).scene == "live"

    # -- ground truth ----------------------------------------------------
    def cutlist(
        self,
        team: str,
        *,
        include_free_throws: bool = True,
        include_and_one_ft: bool = True,
        pre_roll: float = 1.0,
        post_roll: float = 1.5,
        ft_before: float = 3.0,
        ft_after: float = 1.0,
        min_len: float = 3.0,
        max_len: float = 30.0,
    ) -> list[dict]:
        """Target clips for one team, following the PRD's NBA boundary rules."""
        clips: list[dict] = []
        by_event: dict[int, dict] = {}
        for ev in self.events:
            if ev.team != team:
                continue
            if ev.kind == "fg":
                end = ev.visible_time + post_roll
                start = (ev.possession_start or ev.make_time) - pre_roll
                if end - start > max_len:
                    start = end - max_len
                if end - start < min_len:
                    start = end - min_len
                clip = {"kind": "field_goal", "events": [ev], "segments": [[start, end]], "trip": None}
                clips.append(clip)
                by_event[ev.index] = clip
            else:
                seg = [ev.visible_time - ft_before, ev.visible_time + ft_after]
                if ev.and_one_of is not None:
                    if include_and_one_ft and ev.and_one_of in by_event:
                        parent = by_event[ev.and_one_of]
                        parent["segments"].append(seg)
                        parent["events"].append(ev)
                    continue
                if not include_free_throws:
                    continue
                if clips and clips[-1]["kind"] == "free_throws" and clips[-1]["trip"] == ev.trip:
                    clips[-1]["segments"].append(seg)
                    clips[-1]["events"].append(ev)
                else:
                    clips.append({"kind": "free_throws", "events": [ev], "segments": [seg], "trip": ev.trip})

        not_live = self.not_live_intervals()
        for clip in clips:
            segs = _merge_intervals([tuple(s) for s in clip["segments"]])
            segs = _subtract_intervals(segs, not_live)
            clip["segments"] = [[max(0.0, a), min(self.duration, b)] for a, b in segs if b - a > 0.05]

        merged: list[dict] = []
        for clip in clips:
            if not clip["segments"]:
                continue
            if merged and clip["segments"][0][0] <= merged[-1]["segments"][-1][1]:
                prev = merged[-1]
                prev["events"].extend(clip["events"])
                prev["segments"] = [
                    list(s)
                    for s in _merge_intervals([tuple(s) for s in prev["segments"] + clip["segments"]])
                ]
                if prev["kind"] != clip["kind"]:
                    prev["kind"] = "field_goal"
            else:
                merged.append(clip)

        out = []
        for i, clip in enumerate(merged):
            events: list[ScoreEvent] = clip["events"]
            first, last = events[0], events[-1]
            points = sum(e.points for e in events)
            own_after = last.score_home if team == HOME else last.score_away
            fg = next((e for e in events if e.kind == "fg"), None)
            out.append(
                {
                    "order": i,
                    "team": team,
                    "kind": clip["kind"],
                    "points": points,
                    "period": first.period,
                    "clock": round(first.clock, 2),
                    "score_before": own_after - points,
                    "score_after": own_after,
                    "score_away": last.score_away,
                    "score_home": last.score_home,
                    "possession_start": round(fg.possession_start, 3) if fg else None,
                    "start_cause": fg.start_cause if fg else None,
                    "start_observable": fg.start_observable if fg else True,
                    "visible_times": [round(e.visible_time, 3) for e in events],
                    "segments": [[round(a, 3), round(b, 3)] for a, b in clip["segments"]],
                    "src_in": round(clip["segments"][0][0], 3),
                    "src_out": round(clip["segments"][-1][1], 3),
                    "event_indexes": [e.index for e in events],
                }
            )
        return out

    def play_by_play(self) -> list[dict]:
        """Scoring events in the normalized shape sport adapters return."""
        rows = []
        for ev in self.events:
            rows.append(
                {
                    "event_id": f"synthetic-{ev.index}",
                    "period": ev.period,
                    "clock": float(int(ev.clock)) if ev.clock >= 60 else round(ev.clock, 1),
                    "team": self.teams[ev.team]["abbr"],
                    "points": ev.points,
                    "scorer": ev.scorer,
                    "description": ev.description,
                    "score_away": ev.score_away,
                    "score_home": ev.score_home,
                    "kind": "free_throw" if ev.kind == "ft" else "field_goal",
                }
            )
        return rows

    def to_truth(self, *, fps: float, width: int, height: int, layout: dict | None = None) -> dict:
        return {
            "version": 1,
            "generator": "possession_cut.synth",
            "sport": "nba",
            "seed": self.seed,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": round(self.duration, 3),
            "teams": {k: {kk: vv for kk, vv in v.items() if kk != "roster"} for k, v in self.teams.items()},
            "layout": layout or {},
            "intervals": {
                "commercial": [[round(a, 3), round(b, 3)] for a, b, k in self.hidden if k == "commercial"],
                "replay_hidden": [[round(a, 3), round(b, 3)] for a, b, k in self.hidden if k == "replay_hidden"],
                "replay_bug": [[round(a, 3), round(b, 3), round(s, 3)] for a, b, s in self.replays],
                "not_live": [[round(a, 3), round(b, 3)] for a, b in self.not_live_intervals()],
                "score_animation": [[round(a, 3), round(b, 3), team] for a, b, team in self.anims],
            },
            "possession_starts": [
                {"t": round(t, 3), "team": team, "cause": cause} for t, team, cause in self.possession_starts
            ],
            "score_events": [
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(ev).items()} for ev in self.events
            ],
            "cutlists": {AWAY: self.cutlist(AWAY), HOME: self.cutlist(HOME)},
            "pbp": self.play_by_play(),
        }


def _merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract_intervals(
    spans: list[tuple[float, float]], holes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    out = []
    for a, b in spans:
        cur = a
        for h0, h1 in holes:
            if h1 <= cur or h0 >= b:
                continue
            if h0 > cur:
                out.append((cur, h0))
            cur = max(cur, h1)
        if cur < b:
            out.append((cur, b))
    return out


# -- builder ----------------------------------------------------------------


class ScriptBuilder:
    """Plays a game forward one possession at a time."""

    def __init__(self, seed: int = 7, start_scores: tuple[int, int] = (0, 0), teams: dict | None = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.teams = teams or DEFAULT_TEAMS
        self.t = 0.0
        self.period: int | None = None
        self.clock = 0.0
        self.clock_running = False
        self.sc: float | None = None
        self.sc_running = False
        self.score = {AWAY: start_scores[0], HOME: start_scores[1]}
        self.situation = "dead"  # dead | live | after_make | off_reb
        self.live_cause = ""
        self.inbound_at = 0.0
        self.ball = HOME

        self.clock_track = LinearTrack()
        self.shot_track = LinearTrack()
        self.score_track = StepTrack()
        self.period_track = StepTrack()
        self.hidden: list[tuple[float, float, str]] = []
        self.replays: list[tuple[float, float, float]] = []
        self.anims: list[tuple[float, float, str]] = []
        self.events: list[ScoreEvent] = []
        self.possession_starts: list[tuple[float, str, str]] = []
        self.make_times: list[float] = []
        self._trip = 0
        self.last_commercial_end = 0.0

        self.score_track.add(0.0, (self.score[AWAY], self.score[HOME]))

    # -- low level ---------------------------------------------------------
    def _u(self, a: float, b: float) -> float:
        return self.rng.uniform(a, b)

    def _advance(self, dt: float) -> None:
        if dt < -1e-9:
            raise ScriptError("cannot go back in time")
        dt = max(dt, 0.0)
        if self.clock_running:
            self.clock -= dt
            if self.clock < -1e-6:
                raise ScriptError(f"game clock ran past zero at t={self.t + dt:.2f}")
        if self.sc is not None and self.sc_running:
            self.sc -= dt
            if self.sc < -1e-6:
                raise ScriptError(f"shot clock violation at t={self.t + dt:.2f}")
        self.t += dt

    def _set_clock(self, running: bool) -> None:
        self.clock_running = running
        self.clock_track.add(self.t, max(self.clock, 0.0), -1.0 if running else 0.0)

    def _set_shot(self, value: float | None, running: bool) -> None:
        if value is not None and self.clock <= value:
            value = None  # shot clock is switched off when the period will end first
        self.sc = value
        self.sc_running = bool(running and value is not None)
        self.shot_track.add(self.t, value, -1.0 if self.sc_running else 0.0)

    def _clock_stops_on_make(self) -> bool:
        return self.clock <= (120.0 if (self.period or 0) >= 4 else 60.0)

    def _stoppage(self, dur: float, replay: tuple[str, float] | None, anchor: float | None = None) -> None:
        """Dead time with the clock stopped, optionally showing a replay."""
        if replay:
            kind, rlen = replay
            lead = 2.5
            if dur < lead + rlen + 1.0:
                dur = lead + rlen + 1.0
            r0 = self.t + lead
            if kind == "bug":
                # re-air the seconds leading up to the whistle/make, bug and all
                src0 = (anchor if anchor is not None else self.t) - (rlen - 1.0)
                src0 = max(src0, 0.0)
                self.replays.append((r0, r0 + rlen, src0))
            else:
                self.hidden.append((r0, r0 + rlen, "replay_hidden"))
        self._advance(dur)

    def _record_score(self, team: str, points: int, kind: str, **extra) -> ScoreEvent:
        t_make = self.t
        self.score[team] += points
        if kind == "fg":
            anim = extra.pop("anim", None)
            if anim is None:
                anim = self.rng.random() < 0.6
            if anim:
                t_bug = t_make + self._u(0.3, 0.7)
                t_vis = t_bug + self._u(0.6, 1.0)
                self.anims.append((t_bug, t_vis, team))
            else:
                t_vis = t_make + self._u(0.6, 1.8)
            self.make_times.append(t_make)
        else:
            extra.pop("anim", None)
            t_vis = t_make + self._u(0.5, 1.2)
        self.score_track.add(t_vis, (self.score[AWAY], self.score[HOME]))
        roster = self.teams[team]["roster"]
        scorer = self.rng.choice(roster)
        if kind == "ft":
            desc = f"{scorer} Free Throw ({points} PT)"
        elif points == 3:
            desc = f"{scorer} {self.rng.randint(23, 29)}' 3PT Jump Shot (3 PTS)"
        else:
            shot = self.rng.choice(["Driving Layup", "Pullup Jump Shot", "Dunk", "Floating Jump Shot"])
            desc = f"{scorer} {shot} (2 PTS)"
        ev = ScoreEvent(
            index=len(self.events),
            team=team,
            points=points,
            kind=kind,
            make_time=t_make,
            visible_time=t_vis,
            period=self.period or 0,
            clock=max(self.clock, 0.0),
            score_away=self.score[AWAY],
            score_home=self.score[HOME],
            scorer=scorer,
            description=desc,
            **extra,
        )
        self.events.append(ev)
        return ev

    # -- plays ---------------------------------------------------------------
    def period_start(self, period: int, clock: float, lead: float = 2.5, ball: str = HOME) -> None:
        self.period = period
        self.period_track.add(self.t, period)
        self.clock = clock
        self._set_clock(False)
        self._set_shot(24.0, False)
        self._advance(lead)
        self.situation = "dead"
        self.ball = ball

    def available_clock(self) -> float:
        """Game clock left once the next possession has actually started."""
        if self.situation == "after_make" and self.clock_running:
            return self.clock - max(0.0, self.inbound_at - self.t)
        return self.clock

    def possession(self, team: str, dur: float | None, outcome: str, **kw) -> None:
        # 1. the possession starts
        observable = True
        shown_before = format_shot_clock(self.sc)
        if self.situation == "after_make":
            wait = max(0.0, self.inbound_at - self.t)
            if self.clock_running and wait >= self.clock:
                self.run_out()
                return
            self._advance(wait)
            start, cause = self.t, "after_make"
            was_stopped = not self.clock_running
            if was_stopped:
                self._set_clock(True)
            self._set_shot(24.0, True)
            # with the shot clock off and the game clock running, only the opponent's
            # score marks this possession, and that shows before the inbound
            observable = self.sc is not None or was_stopped
        elif self.situation == "dead":
            start, cause = self.t, "clock_start"
            self._set_clock(True)
            self._set_shot(self.sc, True)
        elif self.situation == "live":
            start, cause = self.t, self.live_cause
            self._set_shot(24.0, True)
            observable = shown_before != format_shot_clock(self.sc)
        elif self.situation == "off_reb":
            start, cause = self.t, "off_rebound"
            self._set_shot(14.0, True)
            observable = shown_before != format_shot_clock(self.sc)
        else:  # pragma: no cover
            raise ScriptError(f"unknown situation {self.situation}")
        self.possession_starts.append((start, team, cause))

        # 2. it runs
        if outcome == "buzzer":
            self._advance(self.clock)
            self.clock = 0.0
            self._set_clock(False)
            self._set_shot(None, False)
            self.situation = "dead"
            return
        assert dur is not None
        if dur >= self.clock:
            raise ScriptError(f"possession of {dur:.1f}s does not fit in {self.clock:.1f}s of clock")
        self._advance(dur)

        # 3. it ends
        if outcome in ("make2", "make3", "and1_2", "and1_3"):
            points = 3 if outcome.endswith("3") else 2
            ev = self._record_score(
                team, points, "fg", anim=kw.get("anim"), possession_start=start, start_cause=cause,
                start_observable=observable,
            )
            self._set_shot(24.0, False)
            if outcome.startswith("and1"):
                self._set_clock(False)
                walk = kw.get("walk", self._u(14.0, 19.0))
                self._stoppage(walk, kw.get("replay"), anchor=ev.make_time + 1.5)
                self._trip += 1
                made = kw.get("ft", True)
                if made:
                    self._record_score(team, 1, "ft", trip=self._trip, and_one_of=ev.index)
                self._after_last_ft(team, made)
            else:
                if self._clock_stops_on_make():
                    self._set_clock(False)
                self.situation = "after_make"
                self.inbound_at = self.t + kw.get("inbound", self._u(2.8, 4.5))
                self.ball = other(team)
        elif outcome in ("miss_def", "steal"):
            self.situation = "live"
            self.live_cause = "def_rebound" if outcome == "miss_def" else "steal"
            self.ball = other(team)
        elif outcome == "miss_off":
            if self.sc is not None and self.sc > 13.0:
                # above 13.0 the bug already reads 14, so the reset would be invisible
                raise ScriptError("offensive rebound needs the shot clock at 13 or less to show a reset")
            self.situation = "off_reb"
            self.ball = team
        elif outcome == "dead_turnover":
            self._set_clock(False)
            self._set_shot(24.0, False)
            self._stoppage(kw.get("dead", self._u(5.0, 9.0)), kw.get("replay"))
            self.situation = "dead"
            self.ball = other(team)
        elif outcome == "foul_dead":
            self._set_clock(False)
            keep = self.sc if (self.sc is None or self.sc >= 14.0) else 14.0
            self._set_shot(keep, False)
            self._stoppage(kw.get("dead", self._u(4.0, 7.0)), kw.get("replay"))
            self.situation = "dead"
            self.ball = team
        elif outcome == "shooting_foul":
            self._set_clock(False)
            self._set_shot(24.0, False)
            fts = kw.get("fts", [True, True])
            self._stoppage(kw.get("walk", self._u(12.0, 17.0)), kw.get("replay"))
            self._trip += 1
            for i, made in enumerate(fts):
                if i > 0:
                    self._advance(kw.get("gap", self._u(8.0, 11.0)))
                if made:
                    self._record_score(team, 1, "ft", trip=self._trip)
            self._after_last_ft(team, fts[-1])
        else:
            raise ScriptError(f"unknown outcome {outcome}")

    def _after_last_ft(self, team: str, made: bool) -> None:
        if made:
            self._advance(self._u(3.0, 5.0))
        else:
            self._advance(1.5)  # rebound
        self.situation = "dead"
        self.ball = other(team)

    def run_out(self) -> None:
        """Let the period clock expire without another possession starting."""
        if not self.clock_running:
            self._set_clock(True)
        self._advance(self.clock)
        self.clock = 0.0
        self._set_clock(False)
        self._set_shot(None, False)
        self.situation = "dead"

    def timeout(self, commercial: float = 18.0, pre: float = 3.0, post: float = 2.5) -> None:
        if self.situation in ("live", "off_reb"):
            raise ScriptError("a timeout needs a dead ball")
        if self.clock_running:
            self._set_clock(False)
        self.situation = "dead"
        self._advance(pre)
        self.hidden.append((self.t, self.t + commercial, "commercial"))
        self._advance(commercial)
        self.last_commercial_end = self.t
        self._advance(post)

    def period_end(self, hold: float = 2.5, commercial: float = 12.0) -> None:
        if self.clock > 1e-6:
            self.run_out()
        self._advance(hold)
        if commercial > 0:
            self.hidden.append((self.t, self.t + commercial, "commercial"))
            self._advance(commercial)
            self.last_commercial_end = self.t

    def build(self, tail: float = 4.0) -> GameScript:
        self._advance(tail)
        self.score_track.finalize()
        self.period_track.finalize()
        return GameScript(
            seed=self.seed,
            teams=self.teams,
            duration=self.t,
            clock_track=self.clock_track,
            shot_track=self.shot_track,
            score_track=self.score_track,
            period_track=self.period_track,
            hidden=sorted(self.hidden),
            replays=sorted(self.replays),
            anims=sorted(self.anims),
            events=self.events,
            possession_starts=self.possession_starts,
            make_times=self.make_times,
        )


# -- ready-made games ---------------------------------------------------------


def coverage_game(seed: int = 7) -> GameScript:
    """A short, fixed game that exercises every situation the pipeline must handle.

    Home (NY) scores nine times: plain makes, a three, a two-shot foul, an and-one,
    a put-back after an offensive rebound, a make after a dead-ball foul, and makes
    inside the last minute (tenths on the clock, shot clock off). Around them: a
    commercial, a replay with the bug hidden, two replays that re-air an earlier
    clock and score, a period break, and score animations that cover the score.
    """
    b = ScriptBuilder(seed=seed, start_scores=(96, 88))
    b.period_start(3, 150.0, lead=2.5, ball=HOME)
    b.possession(HOME, 9.0, "make2", anim=True)
    b.possession(AWAY, 8.0, "miss_def")
    b.possession(HOME, 7.0, "make3", anim=False)
    b.possession(AWAY, 10.0, "make2", anim=True)
    b.possession(HOME, 11.0, "shooting_foul", fts=[True, True], replay=("bug", 5.0))
    b.possession(AWAY, 6.0, "steal")
    b.possession(HOME, 5.0, "and1_2", ft=True, replay=("bug", 6.0))
    b.possession(AWAY, 9.0, "make3", anim=True)
    b.timeout(commercial=18.0)
    b.possession(HOME, 12.0, "miss_off")
    b.possession(HOME, 5.0, "make2", anim=False)
    b.possession(AWAY, 7.0, "dead_turnover", replay=("hidden", 5.0))
    b.possession(HOME, 9.0, "make3", anim=True)
    b.possession(AWAY, 8.0, "shooting_foul", fts=[False, True])
    b.possession(HOME, 10.0, "make2", anim=True)
    b.possession(AWAY, None, "buzzer")
    b.period_end(hold=2.5, commercial=12.0)
    b.period_start(4, 75.0, lead=2.5, ball=AWAY)
    b.possession(AWAY, 8.0, "miss_def")
    b.possession(HOME, 6.0, "make2", anim=True)
    b.possession(AWAY, 6.0, "make2", anim=False)
    b.possession(HOME, 7.0, "foul_dead", dead=5.0)
    b.possession(HOME, 6.0, "make2", anim=False)
    b.possession(AWAY, 10.0, "make3", anim=True)
    b.possession(HOME, 8.0, "steal")
    b.possession(AWAY, None, "buzzer")
    return b.build(tail=4.0)


def random_game(
    seed: int = 1,
    periods: tuple[int, ...] = (1, 2, 3, 4),
    period_seconds: float = 180.0,
    start_scores: tuple[int, int] = (0, 0),
    commercial_every: tuple[float, float] = (110.0, 190.0),
) -> GameScript:
    """A randomized game with realistic stoppages, replays and commercial breaks."""
    b = ScriptBuilder(seed=seed, start_scores=start_scores)
    rng = b.rng
    for n, period in enumerate(periods):
        b.period_start(period, period_seconds, lead=rng.uniform(2.0, 3.5), ball=rng.choice([AWAY, HOME]))
        next_commercial = b.t + rng.uniform(*commercial_every)
        team = b.ball
        while True:
            if b.situation in ("after_make", "dead") and b.t >= next_commercial and b.clock > 30:
                b.timeout(commercial=rng.uniform(15.0, 28.0), pre=rng.uniform(2.0, 4.0), post=rng.uniform(2.0, 3.5))
                next_commercial = b.t + rng.uniform(*commercial_every)
            avail = b.available_clock()
            if avail < 8.5:
                b.possession(team, None, "buzzer")
                break
            shot = b.sc if (b.situation == "dead" and b.sc is not None) else 24.0
            if b.situation == "off_reb":
                shot = 14.0
            longest = min(avail - 1.0, shot - 1.5, 21.0)
            roll = rng.random()
            replay = None
            if rng.random() < 0.5:
                replay = (rng.choice(["bug", "hidden"]), rng.uniform(4.5, 7.0))
            if roll < 0.26:
                outcome, kw = "make2", {}
            elif roll < 0.40:
                outcome, kw = "make3", {}
            elif roll < 0.62:
                outcome, kw = "miss_def", {}
            elif roll < 0.68:
                outcome, kw = "steal", {}
            elif roll < 0.75:
                outcome, kw = "dead_turnover", {"replay": replay if replay and replay[0] == "hidden" else None}
            elif roll < 0.82:
                outcome, kw = "miss_off", {}
            elif roll < 0.92:
                n_ft = 3 if rng.random() < 0.15 else 2
                outcome, kw = "shooting_foul", {"fts": [rng.random() < 0.78 for _ in range(n_ft)], "replay": replay}
            elif roll < 0.96:
                outcome = "and1_3" if rng.random() < 0.15 else "and1_2"
                kw = {"ft": rng.random() < 0.8, "replay": replay}
            else:
                outcome, kw = "foul_dead", {"replay": None}

            lo = 4.0
            if outcome == "miss_off":
                # the shot clock must read 13 or less for the reset to 14 to be visible
                lo = max(lo, shot - 12.5)
                if b.situation == "off_reb" or lo > longest or avail < lo + 12.0:
                    outcome, kw, lo = "miss_def", {}, 4.0
            if longest < lo + 0.5:
                b.possession(team, None, "buzzer")
                break
            dur = rng.uniform(lo, longest)
            b.possession(team, dur, outcome, **kw)
            team = b.ball
        last = n == len(periods) - 1
        b.period_end(hold=rng.uniform(2.0, 3.0), commercial=0.0 if last else rng.uniform(12.0, 22.0))
    return b.build(tail=4.0)
