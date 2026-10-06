"""The analysis job: sample -> timeline -> events -> play-by-play -> clips.

Every stage writes its output to the job folder so a stage can be inspected or rerun on
its own. The expensive one (sampling the bug across the whole file) is cached against the
calibration it was read with, so changing the team, the start point or an option reruns
in seconds.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..ai.caption import CutFacts, suggested_title
from ..ai.claude import ClaudeClient
from ..sports.base import PlayByPlayUnavailable, ScoringEvent, SportAdapter
from .calibration import Calibration
from .camera import refine_clips
from .clips import ClipDraft, build_clips
from .events import detect_score_events
from .export import thumbnail
from .matching import MatchResult, label_clips, load_sidecar, match_events, sides_of
from .ocr_fallback import reread_with_claude
from .probe import Probe
from .sampler import RawSamples, sample_bug
from .timeline import Timeline, build_timeline
from .window import resolve_window, time_at_game_time

log = logging.getLogger(__name__)


@dataclass
class JobSpec:
    sport: str = "nba"
    team: str | None = None  # abbreviation, or "away"/"home"
    game_id: str | None = None
    game: dict = field(default_factory=dict)  # lookup row: away, home, scores, label, date
    start_spec: dict = field(default_factory=lambda: {"mode": "start"})
    end_spec: dict = field(default_factory=lambda: {"mode": "end"})
    options: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    clips: list[ClipDraft]
    summary: dict
    timeline: Timeline
    follow_side: str


def _loosely_same(a: str, b: str) -> bool:
    """'NY' ~ 'NYK', 'SA' ~ 'SAS', 'KNICKS' ~ 'NYK' is not attempted."""
    a, b = a.strip().upper(), b.strip().upper()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a) or (len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2])


def resolve_side(spec: JobSpec, cal: Calibration, pbp_sides: dict[str, str]) -> tuple[str, list[str]]:
    """Which side of the bug ("away" or "home") is the followed team."""
    notes: list[str] = []
    team = (spec.team or "").strip()
    if team.lower() in ("away", "home"):
        return team.lower(), notes
    for source in (pbp_sides, {"away": spec.game.get("away", ""), "home": spec.game.get("home", "")}):
        for side in ("away", "home"):
            if team and source.get(side, "").upper() == team.upper():
                return side, notes
    for side in ("away", "home"):
        if team and _loosely_same(cal.teams.get(side, ""), team):
            return side, notes
    notes.append(f"Could not tell which side of the bug {team or 'the followed team'} is; following the home side.")
    return "home", notes


def swap_sides(tl: Timeline) -> None:
    tl.score_away, tl.score_home = tl.score_home, tl.score_away
    tl.away_read, tl.home_read = tl.home_read, tl.away_read
    tl.__dict__.pop("_score_changes", None)


def sampling_signature(probe: Probe, cal: Calibration, fps: float) -> str:
    payload = json.dumps(
        {"bug": cal.bug, "fields": cal.fields, "fps": fps, "size": probe.size_bytes, "duration": round(probe.duration, 2)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _final_scores(tl: Timeline) -> tuple[int | None, int | None]:
    def last(arr: np.ndarray) -> int | None:
        known = arr[~np.isnan(arr)]
        return int(known[-1]) if len(known) else None

    return last(tl.score_away), last(tl.score_home)


def _team_names(spec: JobSpec, sides: dict[str, str], cal: Calibration, follow: str) -> tuple[str, str, str, str]:
    """(team short name, team abbr, opponent short name, opponent abbr)."""
    other = "home" if follow == "away" else "away"

    def abbr(side: str) -> str:
        return spec.game.get(side) or sides.get(side) or cal.teams.get(side) or side.title()

    def short(side: str) -> str:
        full = spec.game.get(f"{side}_name", "")
        return full.split()[-1] if full else abbr(side)

    return short(follow), abbr(follow), short(other), abbr(other)


def analyze(
    probe: Probe,
    cal: Calibration,
    reference: np.ndarray | None,
    mask: np.ndarray | None,
    adapter: SportAdapter,
    spec: JobSpec,
    job_dir: Path,
    fps: float = 2.0,
    workers: int = 4,
    claude: ClaudeClient | None = None,
    progress: Callable[[float, str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> AnalysisResult:
    """Run every analysis stage for one job. ``progress(fraction, stage, message)``."""

    def step(frac: float, stage: str, message: str) -> None:
        if progress:
            progress(frac, stage, message)

    warnings: list[str] = list(cal.warnings)
    job_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. sample + OCR (cached against the calibration)
    signature = sampling_signature(probe, cal, fps)
    raw_path, sig_path = job_dir / "timeline_raw.parquet", job_dir / "timeline_raw.sig"
    sampler_stats: dict = {}
    if raw_path.exists() and sig_path.exists() and sig_path.read_text().strip() == signature:
        step(0.80, "sampling", "Using the saved score bug reads")
        raw = RawSamples.load(raw_path, fps)
        sampler_stats = {"cached": True, "samples": len(raw)}
    else:
        step(0.0, "sampling", "Reading the score bug")
        raw = sample_bug(
            probe, cal, adapter, reference, mask, fps=fps, workers=workers,
            progress=lambda f, m: step(0.80 * f, "sampling", m), should_stop=should_stop,
        )
        if should_stop and should_stop():
            raise InterruptedError("analysis cancelled")
        raw.save(raw_path)
        sig_path.write_text(signature)
        sampler_stats = raw.stats

    # -- 2. timeline and score events
    step(0.82, "timeline", "Building the game timeline")
    tl = build_timeline(raw, adapter)
    events = detect_score_events(tl, adapter)

    # -- 3. Claude re-read of the uncertain samples
    fallback = {"skipped": "nothing uncertain"}
    if raw.crops:
        step(0.84, "ocr_fallback", "Re-reading uncertain samples")
        fallback = reread_with_claude(raw, tl, events, adapter, claude)
        if fallback.get("updated"):
            tl = build_timeline(raw, adapter)
            events = detect_score_events(tl, adapter)
            raw.save(raw_path)

    # -- 4. play-by-play
    step(0.87, "pbp", "Fetching play-by-play")
    pbp: list[ScoringEvent] | None = None
    pbp_sides: dict[str, str] = {}
    pbp_source = ""
    sidecar = load_sidecar(probe.path)
    if sidecar is not None:
        pbp, pbp_sides = sidecar
        pbp_source = "file next to the video"
    elif spec.game_id:
        try:
            pbp = adapter.fetch_pbp(spec.game_id)
            pbp_sides = sides_of(pbp)
            pbp_source = adapter.name
        except PlayByPlayUnavailable as exc:
            warnings.append(f"Play-by-play unavailable, clips are unlabeled: {exc}")
        except Exception as exc:  # a league API misbehaving must not sink the job
            log.exception("play-by-play fetch failed")
            warnings.append(f"Play-by-play failed ({type(exc).__name__}); clips are unlabeled.")
    else:
        warnings.append("No game selected, so clips are not labeled with play-by-play.")
    if pbp is not None:
        (job_dir / "pbp.json").write_text(
            json.dumps({"source": pbp_source, **pbp_sides, "events": [e.to_dict() for e in pbp]}, indent=1),
            encoding="utf-8",
        )

    # -- 5. is the bug's first-listed team really the away team?
    away_final, home_final = _final_scores(tl)
    want_away = spec.game.get("away_score") if spec.game else None
    want_home = spec.game.get("home_score") if spec.game else None
    if pbp:
        want_away = want_away if want_away is not None else pbp[-1].score_away
        want_home = want_home if want_home is not None else pbp[-1].score_home
    if None not in (away_final, home_final, want_away, want_home) and want_away != want_home:
        if (away_final, home_final) == (want_home, want_away):
            swap_sides(tl)
            events = detect_score_events(tl, adapter)
            warnings.append("This bug lists the home team first; sides were swapped to match the game.")
        elif (away_final, home_final) != (want_away, want_home):
            warnings.append(
                f"Final score read from the bug ({away_final}-{home_final}) differs from the game's "
                f"({want_away}-{want_home}). The file may not cover the whole game, or OCR needs a look."
            )

    follow, side_notes = resolve_side(spec, cal, pbp_sides)
    warnings.extend(side_notes)

    # -- 6. match, window, clips
    step(0.90, "matching", "Matching plays and finding clip boundaries")
    matches: MatchResult | None = None
    if pbp:
        matches = match_events(events, pbp, pbp_sides, clock_tolerance=adapter.pbp_clock_tolerance)
        for ch in events:
            m = matches.matches.get(ch.index)
            if m is not None:
                ch.extra["pbp"] = [e.to_dict() for e in m.events]
    window = resolve_window(tl, events, adapter, follow, spec.start_spec, spec.end_spec, pbp, matches)
    warnings.extend(window.notes or [])
    defense: list[tuple[float, str, ScoringEvent]] = []
    if spec.options.get("include_defense") and pbp is not None:
        if sidecar is not None or not spec.game_id:
            warnings.append("Defensive plays need the league's play-by-play; none were added.")
        else:
            try:
                want = pbp_sides.get(follow, "").upper()
                for play in adapter.fetch_defense(spec.game_id):
                    if play.team.upper() != want:
                        continue
                    t = time_at_game_time(tl, adapter, play.period, play.clock)
                    if t is not None:
                        defense.append((t, follow, play))
            except PlayByPlayUnavailable as exc:
                warnings.append(f"Defensive plays unavailable: {exc}")
            except Exception as exc:  # the cut must not sink over a side feature
                log.exception("defensive plays failed")
                warnings.append(f"Defensive plays failed ({type(exc).__name__}).")
    clips = build_clips(tl, events, adapter, follow, spec.options, t_min=window.t_min, t_max=window.t_max, defense=defense)
    label_clips(clips, matches)

    # -- 6b. the bug cannot see what the director shows: keep clip edges on the game camera
    camera_stats: dict = {"skipped": "turned off"}
    if clips and spec.options.get("trim_cutaways", True):
        step(0.92, "camera", "Checking the camera at clip edges")
        try:
            camera_stats = refine_clips(probe, clips, adapter.min_clip_seconds)
        except Exception:  # a refinement: never worth failing the analysis over
            log.exception("camera check failed")
            camera_stats = {"skipped": "the camera check failed; clips are untrimmed"}

    # -- 7. thumbnails
    step(0.94, "thumbnails", "Making thumbnails")
    thumbs = job_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)

    def make_thumb(item: tuple[int, ClipDraft]) -> None:
        k, clip = item
        t = max(clip.src_in, clip.changes[0].t - 1.0)
        thumbnail(probe, t, cal.crop, thumbs / f"{k:03d}.jpg")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(make_thumb, enumerate(clips)))

    # -- 8. artifacts and summary
    tl.save(job_dir / "timeline.parquet")
    (job_dir / "events.json").write_text(json.dumps([asdict(e) for e in events], indent=1), encoding="utf-8")
    (job_dir / "cutlist.json").write_text(json.dumps([c.to_dict() for c in clips], indent=1, default=str), encoding="utf-8")

    sides_in_cut = {follow} | ({"home" if follow == "away" else "away"} if spec.options.get("include_opponent") else set())
    unmatched_pbp = []
    if matches is not None:
        for ev in matches.unmatched_pbp:
            side = ev.extra.get("side")
            if side not in sides_in_cut:
                continue
            t = time_at_game_time(tl, adapter, ev.period, ev.clock)
            if t is None or (window.t_min is not None and t < window.t_min) or (window.t_max is not None and t > window.t_max):
                continue
            unmatched_pbp.append(
                {"event_id": ev.event_id, "period": ev.period, "clock": ev.clock, "team": ev.team,
                 "points": ev.points, "description": ev.description, "t": round(t, 2)}
            )

    team, team_abbr, opp, opp_abbr = _team_names(spec, pbp_sides, cal, follow)
    away_final, home_final = _final_scores(tl)
    own_final, opp_final = (home_final, away_final) if follow == "home" else (away_final, home_final)
    scorers: dict[str, int] = {}
    for clip in clips:
        if matches is None:
            break
        for ch in clip.changes:
            m = matches.matches.get(ch.index)
            for ev in (m.events if m else []):
                if ev.scorer and ev.extra.get("side") == follow:
                    scorers[ev.scorer] = scorers.get(ev.scorer, 0) + ev.points
    facts = CutFacts(
        sport=adapter.name, team=team, team_abbr=team_abbr, opponent=opp, opponent_abbr=opp_abbr,
        game_label=spec.game.get("label", ""), date=spec.game.get("date", ""),
        final_score=(f"{team_abbr} {own_final}, {opp_abbr} {opp_final}" if None not in (own_final, opp_final) else ""),
        won=(own_final > opp_final) if None not in (own_final, opp_final) and own_final != opp_final else None,
        start_label=window.start_label,
        deficit=(window.run_start or {}).get("deficit") if spec.start_spec.get("mode") == "auto_run" else None,
        clips=len(clips), points=sum(c.points for c in clips if c.team == follow),
        duration_seconds=round(sum(c.duration for c in clips), 1),
        top_scorers=[f"{name} {pts}" for name, pts in sorted(scorers.items(), key=lambda kv: -kv[1])[:3]],
        plays=[c.description for c in clips if c.description][-3:],
    )

    notes = tl.notes
    live = tl.live
    side_miss = {
        side: float((~read & live).sum() / max(1, int(live.sum())))
        for side, read in (("away", tl.away_read), ("home", tl.home_read))
    }
    one_sided = [
        side for side in ("away", "home")
        if side_miss[side] > 0.3 and side_miss[side] > 2 * side_miss["home" if side == "away" else "away"]
    ]
    if one_sided:
        # one score box is off: that team's baskets are gone, and the generic note would hide it
        for side in one_sided:
            who = pbp_sides.get(side) or cal.teams.get(side) or f"The {side} team"
            warnings.append(
                f"{who}'s score could not be read in {side_miss[side]:.0%} of live samples, so its baskets are "
                f"mostly missing. Fix the {side} score box in calibration and analyze again."
            )
    elif notes.get("score_unreadable", 0) > 0.15:
        warnings.append("The score was unreadable in over 15% of live samples; check the score boxes in calibration.")
    if notes.get("clock_unreadable", 0) > 0.15 and adapter.has_clock:
        warnings.append("The game clock was unreadable in over 15% of live samples; check the clock box in calibration.")
    if notes.get("shot_clock_seen", 1) < 0.2 and "shot_clock" in cal.fields:
        warnings.append("The shot clock was rarely readable; clip starts lean on the game clock and scores.")
    if "shot_clock" not in cal.fields and any(s.name == "shot_clock" for s in adapter.bug_fields):
        warnings.append("No shot clock in this bug; clip starts use the game clock and scores, and may run long.")
    if fallback.get("skipped") and raw.crops and "nothing uncertain" not in fallback["skipped"]:
        warnings.append(f"Claude OCR fallback not used: {fallback['skipped']}")

    summary = {
        "follow_side": follow,
        "team": team, "team_abbr": team_abbr, "opponent": opp, "opponent_abbr": opp_abbr,
        "clips": len(clips),
        "runtime_seconds": round(sum(c.duration for c in clips), 2),
        "events_detected": len(events),
        "start_label": window.start_label, "end_label": window.end_label,
        "t_min": window.t_min, "t_max": window.t_max,
        "run_start": window.run_start,
        "pbp": {"available": pbp is not None, "source": pbp_source, **(matches.summary() if matches else {})},
        "unmatched_pbp": unmatched_pbp,
        "unmatched_detected": [
            {"t": round(c.t, 2), "team": c.team, "points": c.points, "period": c.period, "clock": c.clock,
             "score_away": c.score_away, "score_home": c.score_home}
            for c in (matches.unmatched_changes if matches else []) if c.team in sides_in_cut
        ],
        "final_score": {"away": away_final, "home": home_final},
        "suggested_title": suggested_title(facts),
        "facts": facts.to_dict(),
        "timeline": notes,
        "sampler": sampler_stats,
        "ocr_fallback": fallback,
        "camera": camera_stats,
        "warnings": warnings,
        "claude_spent_usd": round(claude.usage.cost_usd, 4) if claude else 0.0,
    }
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    step(1.0, "done", f"{len(clips)} clips, {summary['runtime_seconds']:.0f}s")
    return AnalysisResult(clips=clips, summary=summary, timeline=tl, follow_side=follow)
