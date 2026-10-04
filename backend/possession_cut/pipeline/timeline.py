"""Stage 4: turn raw OCR samples into a clean game timeline.

One row per sample: t_video, bug_visible, period, clock, shot_clock, score_home,
score_away, confidence, plus the two derived flags everything downstream relies on:
``live`` and ``clock_running``.

Cleaning rules (PRD "Build the timeline"):

- Scores never decrease within a game and the game clock only runs down within a period.
  Reads that break this are rejected and filled from their neighbours. The check is
  global, not greedy: the longest consistent chain of reads wins, so one confident
  misread cannot poison everything after it.
- A new score counts only once it holds for two samples.
- A sample is not live when the bug is hidden, or when the clock or score sits behind
  where the game already is for a sustained stretch (a replay re-airing the bug).
- The clock is running where it decreases across neighbouring samples.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from ..sports.base import SportAdapter
from .calibration import parse_field
from .sampler import RawSamples

STATE_HIDDEN, STATE_LIVE, STATE_REPLAY = 0, 1, 2

SCORE_MIN_CONF = 0.45
CLOCK_MIN_CONF = 0.30
REPLAY_MIN_SAMPLES = 3  # a backward jump must last this long to be a replay, not a misread
STOPPED_SECONDS = 1.4  # a clock value held this long means the clock is stopped
# A stoppage that restarts a possession (whistle, ball handed in) lasts seconds. Requiring
# this much keeps one repeated clock misread from looking like a stop-and-start.
RESTART_MIN_STOP = 2.4
# an operator correcting a shot clock reset (24, then 14) does it within this long
CORRECTION_SECONDS = 1.5
FILL_GAP_SECONDS = 6.0


def lnds_mask(values: np.ndarray) -> np.ndarray:
    """Mask of a longest non-decreasing subsequence (ties allowed). O(n log n)."""
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    tails: list[float] = []
    tail_idx: list[int] = []
    prev = np.full(n, -1, dtype=np.int64)
    for i, v in enumerate(values):
        pos = bisect.bisect_right(tails, v)
        if pos == len(tails):
            tails.append(v)
            tail_idx.append(i)
        else:
            tails[pos] = v
            tail_idx[pos] = i
        prev[i] = tail_idx[pos - 1] if pos > 0 else -1
    i = tail_idx[-1]
    while i >= 0:
        mask[i] = True
        i = prev[i]
    return mask


def weighted_lnds(values: list[float], weights: list[float]) -> list[bool]:
    """Keep-mask of the heaviest non-decreasing subsequence. O(n^2); n is the number of runs."""
    n = len(values)
    if n == 0:
        return []
    best = list(weights)
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if values[j] <= values[i] and best[j] + weights[i] > best[i]:
                best[i] = best[j] + weights[i]
                prev[i] = j
    i = max(range(n), key=lambda k: best[k])
    keep = [False] * n
    while i >= 0:
        keep[i] = True
        i = prev[i]
    return keep


@dataclass
class ShotReset:
    t: float  # when the shot clock jumped up
    t_start: float  # when it began counting down again (the possession really starting)
    value: float
    index: int


@dataclass
class ClockStart:
    t: float  # estimated moment the game clock started running after a stoppage
    stopped_for: float
    index: int


@dataclass
class Timeline:
    fps: float
    t: np.ndarray
    visible: np.ndarray
    state: np.ndarray  # STATE_*
    period: np.ndarray  # float, nan = unknown
    clock: np.ndarray  # seconds remaining, nan = unknown
    shot_clock: np.ndarray  # nan = off or unknown
    score_away: np.ndarray
    score_home: np.ndarray
    clock_running: np.ndarray
    confidence: np.ndarray
    # True where the value was read directly at this sample (not filled from neighbours)
    clock_read: np.ndarray
    away_read: np.ndarray
    home_read: np.ndarray
    notes: dict = field(default_factory=dict)
    shot_reset_values: tuple[float, ...] = ()
    # Text read for bug fields that have no column of their own (down and distance, count,
    # outs, runners...), one entry per sample, "" where not live or unreadable.
    extra: dict[str, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def live(self) -> np.ndarray:
        return self.state == STATE_LIVE

    def score(self, side: str) -> np.ndarray:
        return self.score_home if side == "home" else self.score_away

    def score_read(self, side: str) -> np.ndarray:
        return self.home_read if side == "home" else self.away_read

    def index_at(self, t: float) -> int:
        """Index of the last sample at or before ``t`` (clamped)."""
        i = int(np.searchsorted(self.t, t + 1e-9, side="right")) - 1
        return max(0, min(len(self.t) - 1, i))

    def value_near(self, arr: np.ndarray, i: int, reach: int = 3) -> float:
        """arr[i], or the nearest known value within ``reach`` samples, or nan."""
        n = len(arr)
        for d in range(reach + 1):
            for j in (i - d, i + d):
                if 0 <= j < n and not np.isnan(arr[j]):
                    return float(arr[j])
        return float("nan")

    # -- intervals ---------------------------------------------------------
    @cached_property
    def not_live_intervals(self) -> list[tuple[float, float]]:
        """Stretches to keep out of any clip.

        Conservative on purpose: a not-live run is widened to the neighbouring live
        samples, because the cut to a replay or commercial happened somewhere between
        two samples and no frame of it may leak into an export.
        """
        out: list[tuple[float, float]] = []
        live = self.live
        n = len(live)
        i = 0
        while i < n:
            if live[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and not live[j + 1]:
                j += 1
            start = self.t[i - 1] if i > 0 else self.t[i] - self.dt
            end = self.t[j + 1] if j + 1 < n else self.t[j] + self.dt
            out.append((float(start), float(end)))
            i = j + 1
        return out

    def intervals_of(self, state: int) -> list[tuple[float, float]]:
        out = []
        sel = self.state == state
        n = len(sel)
        i = 0
        while i < n:
            if not sel[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and sel[j + 1]:
                j += 1
            out.append((float(self.t[i]), float(self.t[j])))
            i = j + 1
        return out

    # -- possession signals ----------------------------------------------------
    @cached_property
    def shot_resets(self) -> list[ShotReset]:
        """Moments the shot clock jumped up, with when it started counting down again."""
        out: list[ShotReset] = []
        sc, live, n = self.shot_clock, self.live, len(self.t)
        last_known = -1
        for i in range(n):
            if not live[i] or np.isnan(sc[i]):
                continue
            found = None
            if last_known >= 0 and sc[i] >= sc[last_known] + 1.0:
                found = self._reset_at(i, float(sc[last_known]))
            if found is not None:
                value, settled = found
                gap = self.t[i] - self.t[last_known]
                t_reset = self.t[i] - self.dt / 2 if gap <= self.dt * 1.5 else self.t[i]
                # find the first decrement: the shot clock shows its reset value for one
                # full second after it starts, so it started 1 s before the change
                t_start = t_reset
                k = settled
                while k < n and self.t[k] - self.t[i] < 40.0:
                    if live[k] and not np.isnan(sc[k]):
                        if sc[k] < value:
                            t_start = max(t_reset, self.t[k] - self.dt / 2 - 1.0)
                            break
                        if sc[k] > value:
                            break
                    k += 1
                out.append(ShotReset(float(t_reset), float(t_start), value, i))
            last_known = i
        # The shot clock is switched off when a possession starts with less game clock than
        # shot clock. Going from a number to blank while play continues marks that change.
        limit = max(self.shot_reset_values) if self.shot_reset_values else (float(np.nanmax(sc)) if not np.isnan(sc).all() else 24.0)
        for i in range(1, n - 3):
            if not (live[i] and np.isnan(sc[i]) and live[i - 1] and not np.isnan(sc[i - 1])):
                continue
            clock = self.value_near(self.clock, i, 2)
            if np.isnan(clock) or clock > limit + 0.5:
                continue
            ahead = slice(i, min(n, i + 4))
            if live[ahead].all() and np.isnan(sc[ahead]).all() and self.visible[ahead].all():
                t_off = float(self.t[i] - self.dt / 2)
                out.append(ShotReset(t_off, t_off, float("nan"), i))
        out.sort(key=lambda r: r.t)
        return out

    def text_changes(self, name: str, hold: int = 2) -> list[tuple[float, str, str]]:
        """Changes in an extra text field: (video time, old text, new text).

        A new value counts once it has been read on ``hold`` samples in a row, the same
        rule scores follow, so one misread does not register as a change.
        """
        track = self.extra.get(name)
        if not track:
            return []
        out: list[tuple[float, str, str]] = []
        current = ""
        i, n = 0, len(track)
        while i < n:
            text = track[i]
            if not text or text == current:
                i += 1
                continue
            j = i
            while j + 1 < n and track[j + 1] == text:
                j += 1
            if j - i + 1 >= hold:
                if current:
                    out.append((float(self.t[i]), current, text))
                current = text
            i = j + 1
        return out

    def _reset_at(self, i: int, previous: float) -> tuple[float, int] | None:
        """If the jump up at sample i is a shot clock reset: the value it was reset to and
        the sample from which that value counts down. None for a misread.

        A reset lands on a reset value and the next read continues from it; a misread digit
        does neither. Two things real broadcasts do are allowed for: the graphic can skip
        the reset value and first show the second after it (12, then 23), and the operator
        can correct a reset within a moment (24, then 14 for an offensive rebound).
        """
        sc = self.shot_clock
        landed = float(sc[i])
        ahead = [
            k for k in range(i + 1, min(len(sc), i + int(3 * self.fps) + 1))
            if self.live[k] and not np.isnan(sc[k])
        ]

        def continues(k: int) -> bool:
            return landed - (self.t[k] - self.t[i]) - 1.0 <= sc[k] <= landed

        targets = self.shot_reset_values
        if not targets:
            return (landed, i) if not ahead or continues(ahead[0]) else None
        if any(abs(landed - r) < 0.01 for r in targets):
            if not ahead or continues(ahead[0]):
                return landed, i
            k = ahead[0]
            corrected = float(sc[k])
            if (
                previous < corrected < landed
                and any(abs(corrected - r) < 0.01 for r in targets)
                and self.t[k] - self.t[i] <= CORRECTION_SECONDS
            ):
                return corrected, k
            return None
        if landed >= previous + 2.0 and len(ahead) >= 2:
            a, b = ahead[0], ahead[1]
            for r in targets:
                if abs(landed - (r - 1.0)) < 0.01 and continues(a) and continues(b) and sc[b] <= sc[a]:
                    return float(r), i
        return None

    def score_change_times(self, side: str) -> list[float]:
        """Video times at which this side's cleaned score went up."""
        cache = self.__dict__.setdefault("_score_changes", {})
        if side not in cache:
            score = self.score(side)
            times = []
            prev = np.nan
            for i in range(len(score)):
                v = score[i]
                if np.isnan(v):
                    continue
                if not np.isnan(prev) and v > prev:
                    times.append(float(self.t[i]))
                prev = v
            cache[side] = times
        return cache[side]

    @cached_property
    def clock_starts(self) -> list[ClockStart]:
        """Moments the game clock started running after being stopped."""
        out: list[ClockStart] = []
        clock, live, n = self.clock, self.live, len(self.t)
        known = [i for i in range(n) if live[i] and not np.isnan(clock[i])]
        a = 0
        while a < len(known):
            b = a
            while b + 1 < len(known) and abs(clock[known[b + 1]] - clock[known[a]]) < 0.05:
                b += 1
            first, last = known[a], known[b]
            held = self.t[last] - self.t[first]
            # At the start of the file, or coming back from a break, the stoppage began
            # before we could see it, so a shorter observed hold is enough.
            unseen_before = a == 0 or self.t[first] - self.t[known[a - 1]] > 2.0
            need = STOPPED_SECONDS if unseen_before else RESTART_MIN_STOP
            if held >= need and b + 1 < len(known):
                nxt = known[b + 1]
                if clock[nxt] < clock[last] and self.t[nxt] - self.t[last] <= 2.5:
                    tenths = clock[last] < 60.0
                    upper = self.t[nxt]
                    lower = self.t[nxt] - self.dt - (0.1 if tenths else 1.0)
                    # the shot clock starts with the game clock; its first tick narrows the window
                    sc = self.shot_clock
                    if not np.isnan(sc[last]):
                        # a shot clock sitting on its reset value is exact: it shows that
                        # value for one full second after it starts
                        fresh = sc[last] in (24.0, 14.0)
                        for k in range(last + 1, min(n, last + int(3 * self.fps) + 1)):
                            if live[k] and not np.isnan(sc[k]) and sc[k] < sc[last]:
                                upper = min(upper, self.t[k] - 1.0 if fresh else self.t[k])
                                lower = max(lower, self.t[k] - self.dt - 1.0)
                                break
                    est = (lower + upper) / 2 if lower < upper else upper - 0.5
                    est = max(est, self.t[last])
                    out.append(ClockStart(float(est), float(held), nxt))
            a = b + 1
        return out

    def stopped_duration(self, i: int, reach: float = 180.0) -> float:
        """Seconds the game clock had already shown its current value at sample i. A replay
        or a break in between does not interrupt it: a stopped clock stays stopped while
        the bug is away. A running whole-second clock gives up to a second."""
        now = self.value_near(self.clock, i, 2)
        if np.isnan(now):
            return 0.0
        first = i
        j = i - 1
        while j >= 0 and self.t[i] - self.t[j] <= reach:
            if self.live[j] and not np.isnan(self.clock[j]):
                if abs(float(self.clock[j]) - now) >= 0.05:
                    break
                first = j
            j -= 1
        return float(self.t[i] - self.t[first])

    def last_clock_stop(self, t: float, within: float) -> float | None:
        """Video time the game clock last went from running to stopped in the ``within``
        seconds before t. None if it kept running, or was already stopped before that."""
        end = self.index_at(t)
        begin = max(0, self.index_at(t - within - 3.0))
        known = [k for k in range(begin, end + 1) if self.live[k] and not np.isnan(self.clock[k])]
        found = None
        a = 0
        while a < len(known):
            b = a
            while b + 1 < len(known) and abs(self.clock[known[b + 1]] - self.clock[known[a]]) < 0.05:
                b += 1
            # a running clock repeats a value too: twice when it shows whole seconds
            need = 0.5 if self.clock[known[a]] < 60.0 else 1.5
            if a > 0 and self.t[known[b]] - self.t[known[a]] >= need and self.t[known[a]] >= t - within:
                found = float(self.t[known[a]] - self.dt / 2)
            a = b + 1
        return found

    def stopped_since(self, i: int, seconds: float) -> bool:
        """Has the game clock shown the same value for at least ``seconds`` up to sample i?"""
        now = self.value_near(self.clock, i, 2)
        if np.isnan(now):
            return False
        # the last clock value we could read at least ``seconds`` ago; a replay or a
        # break may sit in between, during which a stopped clock stays stopped
        j = self.index_at(self.t[i] - seconds)
        floor = self.t[i] - 180.0
        while j >= 0 and self.t[j] >= floor and (np.isnan(self.clock[j]) or not self.live[j]):
            j -= 1
        if j < 0 or self.t[j] < floor:
            return False
        return abs(now - float(self.clock[j])) < 0.05

    # -- storage ---------------------------------------------------------------
    def to_frame(self) -> pl.DataFrame:
        def nullable_int(arr: np.ndarray) -> pl.Series:
            return pl.Series([None if np.isnan(v) else int(v) for v in arr], dtype=pl.Int32)

        def nullable_float(arr: np.ndarray) -> pl.Series:
            return pl.Series([None if np.isnan(v) else float(v) for v in arr], dtype=pl.Float64)

        return pl.DataFrame(
            {
                "t_video": self.t,
                "bug_visible": self.visible,
                "period": nullable_int(self.period),
                "clock": nullable_float(self.clock),
                "shot_clock": nullable_float(self.shot_clock),
                "score_home": nullable_int(self.score_home),
                "score_away": nullable_int(self.score_away),
                "confidence": self.confidence.astype(np.float32),
                "live": self.live,
                "clock_running": self.clock_running,
                "state": self.state.astype(np.int8),
                "clock_read": self.clock_read,
                "home_read": self.home_read,
                "away_read": self.away_read,
            }
        )

    def save(self, path: Path) -> None:
        self.to_frame().write_parquet(path)

    @classmethod
    def load(cls, path: Path, fps: float) -> Timeline:
        df = pl.read_parquet(path)

        def arr(name: str) -> np.ndarray:
            return df[name].cast(pl.Float64).fill_null(float("nan")).to_numpy().astype(np.float64)

        return cls(
            fps=fps,
            t=df["t_video"].to_numpy().astype(np.float64),
            visible=df["bug_visible"].to_numpy().astype(bool),
            state=df["state"].to_numpy().astype(np.int8),
            period=arr("period"),
            clock=arr("clock"),
            shot_clock=arr("shot_clock"),
            score_away=arr("score_away"),
            score_home=arr("score_home"),
            clock_running=df["clock_running"].to_numpy().astype(bool),
            confidence=df["confidence"].to_numpy().astype(np.float32),
            clock_read=df["clock_read"].to_numpy().astype(bool),
            away_read=df["away_read"].to_numpy().astype(bool),
            home_read=df["home_read"].to_numpy().astype(bool),
        )


# -- building ------------------------------------------------------------------------


def _parse_column(raw: RawSamples, name: str, min_conf: float, parse=parse_field) -> np.ndarray:
    out = np.full(len(raw), np.nan)
    texts = raw.texts.get(name)
    if texts is None:
        return out
    confs = raw.confs[name]
    for i, text in enumerate(texts):
        if not raw.visible[i] or not text or confs[i] < min_conf:
            continue
        value = parse(name, text)
        if value is not None:
            try:
                out[i] = float(value)
            except (TypeError, ValueError):
                continue
    return out


CORE_FIELDS = {"period", "clock", "shot_clock", "away_score", "home_score", "away_label", "home_label"}


def _ffill(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _clean_period(period_raw: np.ndarray) -> np.ndarray:
    """Periods only go up. Keep the longest consistent set of reads and fill the rest."""
    idx = np.flatnonzero(~np.isnan(period_raw))
    out = np.full(len(period_raw), np.nan)
    if len(idx) == 0:
        return out
    keep = lnds_mask(period_raw[idx])
    out[idx[keep]] = period_raw[idx[keep]]
    out = _ffill(out)
    first = idx[keep][0]
    out[:first] = out[first]
    return out


def _infer_period(clock_raw: np.ndarray, period_seconds: float) -> np.ndarray:
    """No period field: count periods from the clock jumping back up to the top."""
    out = np.full(len(clock_raw), np.nan)
    idx = np.flatnonzero(~np.isnan(clock_raw))
    if len(idx) == 0:
        return out
    period = 1
    low = clock_raw[idx[0]]
    for pos, i in enumerate(idx):
        c = clock_raw[i]
        if c > low + 0.4 * period_seconds:
            # sustained: most of the next few reads are also up here
            ahead = clock_raw[idx[pos : pos + 8]]
            if (ahead > low + 0.4 * period_seconds).mean() >= 0.7:
                period += 1
                low = c
        low = min(low, c)
        out[i] = period
    out = _ffill(out)
    out[: idx[0]] = out[idx[0]]
    return out


def _runs_of(mask: np.ndarray, max_gap: int = 1) -> list[tuple[int, int, int]]:
    """Runs of True in ``mask`` allowing gaps of up to ``max_gap`` False: (first, last, count)."""
    out = []
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    start = prev = idx[0]
    count = 1
    for i in idx[1:]:
        if i - prev <= max_gap + 1:
            count += 1
        else:
            out.append((int(start), int(prev), count))
            start, count = i, 1
        prev = i
    out.append((int(start), int(prev), count))
    return out


def _clean_scores(
    raw_values: np.ndarray, eligible: np.ndarray, replay: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Stable, non-decreasing score track.

    ``replay`` marks samples already known to be a replay from the clock going backward.
    Returns (score per sample, read mask, sustained backward runs as (first, last) indexes).
    """
    n = len(raw_values)
    idx = [i for i in range(n) if eligible[i] and not np.isnan(raw_values[i])]
    runs: list[list] = []  # [first, last, value, count]
    for i in idx:
        v = raw_values[i]
        if runs and runs[-1][2] == v:
            runs[-1][1] = i
            runs[-1][3] += 1
        else:
            runs.append([i, i, v, 1])
    # A value counts only once it holds for 2+ samples. A later run showing the same value
    # supports an earlier one when everything in between is old news: a new score shows
    # briefly, a replay re-airs the previous score, then the new score returns. The first
    # showing was real and the stretch in between was the replay (scores never go back
    # down). A lone misread gets no such credit: between it and any later true run of the
    # same value the game moves through scores that had not been seen yet.
    support = []
    seen_max = -np.inf  # highest value that has held for 2+ samples so far
    for k, r in enumerate(runs):
        sup = 0
        for o in runs[k + 1 :]:
            if o[2] == r[2]:
                # A single-sample showing needs hard evidence that a replay came between:
                # the clock was seen going backward. Otherwise it is just a misread that
                # happens to equal the next score.
                if r[3] >= 2 or (replay is not None and replay[r[1] + 1 : o[0]].any()):
                    sup += o[3]
            elif o[2] > seen_max or o[2] > r[2]:
                break
        support.append(sup)
        if r[3] >= 2:
            seen_max = max(seen_max, r[2])
    picked = [(r, sup) for r, sup in zip(runs, support, strict=True) if r[3] >= 2 or (r[3] + sup >= 3 and sup > 0)]
    stable = [r for r, _ in picked]
    keep = weighted_lnds([r[2] for r in stable], [r[3] + sup for r, sup in picked])
    score = np.full(n, np.nan)
    read = np.zeros(n, dtype=bool)
    backward: list[tuple[int, int]] = []
    current = np.nan
    accepted: list[list] = []
    for r, k in zip(stable, keep, strict=True):
        if k:
            accepted.append(r)
            current = r[2]
        elif r[3] >= REPLAY_MIN_SAMPLES and not np.isnan(current) and r[2] < current:
            backward.append((r[0], r[1]))
    for r in accepted:
        first, last, value, _ = r
        score[first:] = value
        for i in range(first, last + 1):
            if eligible[i] and raw_values[i] == value:
                read[i] = True
    if accepted:
        score[: accepted[0][0]] = accepted[0][2]
    return score, read, backward


def _drop_shot_outliers(shot: np.ndarray, t: np.ndarray, resets: tuple[float, ...], reach: float = 3.0) -> None:
    """Blank shot clock reads that are physically impossible.

    A shot clock only counts down in real time, holds, or jumps to a reset value. The
    pass follows a trusted chain of reads: a read that continues the chain or is a
    legitimate reset extends it; anything else (a drop of ten seconds between samples, a
    jump to a value that is not a reset) is blanked. If the "impossible" reads keep
    agreeing with each other for two seconds, the chain was the thing that was wrong and
    the pass switches over to them.
    """

    def continues(prev: float, cur: float, dt: float) -> bool:
        return prev - dt - 1.0 <= cur <= prev + 0.01

    def is_reset(v: float) -> bool:
        return any(abs(v - r) < 0.01 for r in resets)

    trusted_i = -1
    pending: list[int] = []
    for i in np.flatnonzero(~np.isnan(shot)):
        b = shot[i]
        if trusted_i < 0 or t[i] - t[trusted_i] > reach:
            trusted_i, pending = i, []
            continue
        a = shot[trusted_i]
        dt = t[i] - t[trusted_i]
        if continues(a, b, dt) or (is_reset(b) and b >= a - dt - 1.0):
            trusted_i, pending = i, []
            continue
        # does it at least agree with the other rejected reads?
        if pending and not continues(shot[pending[-1]], b, t[i] - t[pending[-1]]):
            pending = []
        pending.append(i)
        if len(pending) >= 4 and t[pending[-1]] - t[pending[0]] >= 1.5:
            trusted_i, pending = i, []  # they were right; keep them
            continue
    # blank whatever is still off the chain: redo the walk and drop rejected reads
    trusted_i = -1
    pending = []
    for i in np.flatnonzero(~np.isnan(shot)):
        b = shot[i]
        if trusted_i < 0 or t[i] - t[trusted_i] > reach:
            for j in pending:
                shot[j] = np.nan
            trusted_i, pending = i, []
            continue
        a = shot[trusted_i]
        dt = t[i] - t[trusted_i]
        if continues(a, b, dt) or (is_reset(b) and b >= a - dt - 1.0):
            for j in pending:
                shot[j] = np.nan
            trusted_i, pending = i, []
            continue
        if pending and not continues(shot[pending[-1]], b, t[i] - t[pending[-1]]):
            for j in pending:
                shot[j] = np.nan
            pending = []
        pending.append(i)
        if len(pending) >= 4 and t[pending[-1]] - t[pending[0]] >= 1.5:
            trusted_i, pending = i, []
    for j in pending:
        shot[j] = np.nan


def build_timeline(raw: RawSamples, adapter: SportAdapter) -> Timeline:
    n = len(raw)
    t = raw.t
    visible = raw.visible.copy()
    dt = 1.0 / raw.fps

    parse = adapter.parse_field
    period_raw = _parse_column(raw, "period", CLOCK_MIN_CONF, parse)
    clock_raw = _parse_column(raw, "clock", CLOCK_MIN_CONF, parse) if adapter.has_clock else np.full(n, np.nan)
    shot_raw = _parse_column(raw, "shot_clock", CLOCK_MIN_CONF, parse)
    away_raw = _parse_column(raw, "away_score", SCORE_MIN_CONF, parse)
    home_raw = _parse_column(raw, "home_score", SCORE_MIN_CONF, parse)
    notes: dict = {}

    # a clock above the period length is a misread
    clock_raw[clock_raw > adapter.period_seconds + 0.5] = np.nan

    # -- period
    if np.isnan(period_raw).all():
        period = _infer_period(clock_raw, adapter.period_seconds)
        notes["period_inferred"] = bool(adapter.has_clock)
    else:
        period = _clean_period(period_raw)

    # -- game time must never go backward: keep the longest consistent chain of clock reads
    elapsed = np.full(n, np.nan)
    has_clock = ~np.isnan(clock_raw) & ~np.isnan(period)
    for i in np.flatnonzero(has_clock):
        elapsed[i] = adapter.elapsed(int(period[i]), float(clock_raw[i]))
    clock_idx = np.flatnonzero(has_clock)
    on_chain = np.zeros(n, dtype=bool)
    if len(clock_idx):
        on_chain[clock_idx[lnds_mask(elapsed[clock_idx])]] = True
    off_chain = has_clock & ~on_chain

    # sustained off-chain stretches are replays; isolated ones are misreads
    state = np.where(visible, STATE_LIVE, STATE_HIDDEN).astype(np.int8)
    replay_runs = [(a, b) for a, b, count in _runs_of(off_chain, max_gap=1) if count >= REPLAY_MIN_SAMPLES]
    for a, b in replay_runs:
        state[a : b + 1] = np.where(visible[a : b + 1], STATE_REPLAY, STATE_HIDDEN)
    notes["clock_misreads"] = int(off_chain.sum() - sum((state[a : b + 1] == STATE_REPLAY).sum() for a, b in replay_runs))

    # -- scores
    eligible = state == STATE_LIVE
    clock_replay = state == STATE_REPLAY
    score_away, away_read, back_a = _clean_scores(away_raw, eligible, clock_replay)
    score_home, home_read, back_h = _clean_scores(home_raw, eligible, clock_replay)
    for a, b in back_a + back_h:
        # the score sat behind the game for a while: a replay whose clock was not readable
        state[a : b + 1] = np.where(visible[a : b + 1], STATE_REPLAY, STATE_HIDDEN)
    if back_a or back_h:
        eligible = state == STATE_LIVE
        score_away, away_read, _ = _clean_scores(away_raw, eligible, clock_replay)
        score_home, home_read, _ = _clean_scores(home_raw, eligible, clock_replay)
    live = state == STATE_LIVE

    # -- clock: direct reads on the chain, short gaps interpolated
    clock = np.full(n, np.nan)
    clock_read = on_chain & live
    clock[clock_read] = clock_raw[clock_read]
    known = np.flatnonzero(clock_read)
    for a, b in zip(known[:-1], known[1:], strict=False):
        if b - a <= 1 or t[b] - t[a] > FILL_GAP_SECONDS or period[a] != period[b]:
            continue
        for i in range(a + 1, b):
            if live[i]:
                frac = (t[i] - t[a]) / (t[b] - t[a])
                clock[i] = clock[a] + (clock[b] - clock[a]) * frac

    # -- shot clock: drop impossible values and one-sample blips
    shot = np.where(live, shot_raw, np.nan)
    limit = getattr(adapter, "shot_clock_max", 24.0)
    shot[shot > limit + 0.5] = np.nan
    resets = tuple(getattr(adapter, "shot_clock_resets", ()) or ())
    _drop_shot_outliers(shot, t, resets)

    # -- clock running: the value differs from a neighbour within a second
    running = np.zeros(n, dtype=bool)
    reach = max(1, int(round(1.0 * raw.fps)))
    for i in np.flatnonzero(~np.isnan(clock) & live):
        lo, hi = max(0, i - reach), min(n - 1, i + reach)
        for j in range(lo, hi + 1):
            if j != i and live[j] and not np.isnan(clock[j]) and abs(clock[j] - clock[i]) > 0.05:
                if (j < i and clock[j] > clock[i]) or (j > i and clock[j] < clock[i]):
                    running[i] = True
                    break

    # -- confidence: the weakest of the reads that matter at this sample
    conf = np.zeros(n, dtype=np.float32)
    parts = [raw.confs.get(k) for k in ("away_score", "home_score", "clock")]
    parts = [p for p in parts if p is not None]
    if parts:
        conf = np.minimum.reduce(parts).astype(np.float32)
    conf[~live] = 0.0

    vis_live = int(live.sum())
    notes.update(
        samples=n,
        visible=int(visible.sum()),
        live=vis_live,
        replay_samples=int((state == STATE_REPLAY).sum()),
        hidden_samples=int((state == STATE_HIDDEN).sum()),
        clock_unreadable=round(float((np.isnan(clock) & live).sum() / max(1, vis_live)), 4),
        score_unreadable=round(float(((~away_read | ~home_read) & live).sum() / max(1, vis_live)), 4),
        shot_clock_seen=round(float((~np.isnan(shot)).sum() / max(1, vis_live)), 4),
        sample_seconds=dt,
    )
    return Timeline(
        fps=raw.fps, t=t, visible=visible, state=state, period=period, clock=clock, shot_clock=shot,
        score_away=score_away, score_home=score_home, clock_running=running, confidence=conf,
        clock_read=clock_read, away_read=away_read, home_read=home_read, notes=notes,
        shot_reset_values=resets,
        extra={
            name: [
                texts[i].strip() if live[i] and raw.confs[name][i] >= CLOCK_MIN_CONF else "" for i in range(n)
            ]
            for name, texts in raw.texts.items() if name not in CORE_FIELDS
        },
    )
