"""Stage 6b: keep the edges of each clip on the game camera.

The bug says play is live. It does not say what the director is showing: a possession can
begin under a crowd shot or a player's close-up while the clock already runs, and a make
is followed within seconds by a close-up of the scorer. Neither is a replay, so nothing
on the bug gives it away.

The pictures do. A game camera shows the playing surface, and one colour (hardwood,
grass, ice) fills a large, steady share of the frame. A crowd shot or a close-up shows
almost none of it. This stage learns that colour from the clips themselves and moves the
start and end of each scoring clip in to the first and last frames that show the surface.

It only ever trims edges, and only when the picture is unambiguous. A whole possession
shown from another camera is left as it is.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import ffmpeg_bin
from .clips import ClipDraft
from .frames import square_pixel_filter
from .probe import Probe

log = logging.getLogger(__name__)

CAMERA_FPS = 2.0
FRAME_W, FRAME_H = 96, 54
# rows looked at: below the top edge, above the lower-third graphics
ROWS = slice(int(FRAME_H * 0.12), int(FRAME_H * 0.82))
HUE_BINS, SAT_BINS = 30, 8
DARK, GREY = 90, 40  # value below DARK is shadow and crowd; saturation below GREY has no hue
BRIGHT_GREY = 170  # ...unless it is bright: ice, a white floor
SURFACE_PEAK = 0.25  # colours at least this common, relative to the commonest, are the surface
MIN_TYPICAL = 0.06  # a "surface" filling less of the frame than this is not one
CUTAWAY_RATIO = 0.15  # a frame showing under this fraction of the usual surface is not the game camera
MIN_RUN = 2  # samples; a single odd frame is a camera flash
LEAD_START = 2.5  # a cutaway at the top of a clip begins within this many seconds of its start
MAX_LEAD_TRIM = 12.0
MAX_TAIL_TRIM = 2.0  # a longer run at the end means the play itself ended on another camera
EDGE_MARGIN = 0.1


@dataclass
class SurfaceModel:
    """Which colours are the playing surface, and how much of a frame they usually fill."""

    bins: np.ndarray  # bool, HUE_BINS * SAT_BINS + 1 (the last is bright grey: ice)
    typical: float


def _colour_index(frames: np.ndarray) -> np.ndarray:
    """Per pixel: the colour bin it falls in, or -1 for dark and dull pixels."""
    flat = frames[:, ROWS].reshape(-1, FRAME_W, 3)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV).reshape(len(frames), -1, 3).astype(np.int32)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    index = np.minimum(hue * HUE_BINS // 180, HUE_BINS - 1) * SAT_BINS + np.minimum(sat * SAT_BINS // 256, SAT_BINS - 1)
    index = np.where(sat < GREY, np.where(val >= BRIGHT_GREY, HUE_BINS * SAT_BINS, -1), index)
    return np.where(val < DARK, -1, index)


def surface_share(frames: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """For each frame, the share of its pixels that are surface-coloured."""
    if len(frames) == 0:
        return np.zeros(0, dtype=np.float32)
    index = _colour_index(frames)
    lookup = np.append(bins, False)  # index -1 lands on the appended False
    return lookup[index].mean(axis=1).astype(np.float32)


def learn_surface(frames: np.ndarray) -> SurfaceModel | None:
    """The surface colour of this game, from frames that mostly show the game camera.
    None when no colour stands out enough to judge by."""
    if len(frames) < 8:
        return None
    index = _colour_index(frames)
    counts = np.bincount(index[index >= 0], minlength=HUE_BINS * SAT_BINS + 1).astype(np.float64)
    if counts.sum() == 0:
        return None
    bins = counts >= SURFACE_PEAK * counts.max()
    typical = float(np.median(surface_share(frames, bins)))
    if typical < MIN_TYPICAL:
        return None
    return SurfaceModel(bins=bins, typical=typical)


def cutaway_flags(frames: np.ndarray, model: SurfaceModel) -> np.ndarray:
    """True for frames that are not the game camera."""
    return surface_share(frames, model.bins) < CUTAWAY_RATIO * model.typical


def trim_edges(flags: np.ndarray, duration: float, min_keep: float, fps: float = CAMERA_FPS) -> tuple[float, float]:
    """Seconds to take off the start and the end of a clip, given which of its samples
    (taken every 1/fps from its first frame) are cutaways."""
    n = len(flags)
    lead = tail = 0.0
    if n == 0:
        return lead, tail

    # the end: cutaway samples running up to the last one
    m = n
    while m > 0 and flags[m - 1]:
        m -= 1
    if 0 < m < n:
        cut = duration - ((m - 1) / fps + EDGE_MARGIN)
        if 0 < cut <= MAX_TAIL_TRIM:
            tail = cut

    # the start: a run of cutaway samples beginning in the first moments
    first = next((i for i in range(n) if flags[i]), None)
    if first is not None and first / fps <= LEAD_START:
        j = first
        while j < n and flags[j]:
            j += 1
        if j - first >= MIN_RUN and j < n and j / fps <= MAX_LEAD_TRIM:
            lead = j / fps

    if duration - lead - tail < min_keep:
        lead = 0.0
        if duration - tail < min_keep:
            tail = 0.0
    return round(lead, 3), round(tail, 3)


def sample_segment(probe: Probe, start: float, end: float, fps: float = CAMERA_FPS) -> np.ndarray:
    """Small frames of ``[start, end)``, the first one at ``start``. Shape (n, FRAME_H, FRAME_W, 3), BGR."""
    filters = [f for f in [square_pixel_filter(probe)] if f]
    filters += [f"fps={fps}:start_time=0:round=up", f"scale={FRAME_W}:{FRAME_H}:flags=area"]
    cmd = [
        ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{max(0.0, end - start):.3f}",
        "-i", probe.path, "-an", "-sn", "-dn", "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=False).stdout
    size = FRAME_W * FRAME_H * 3
    n = len(raw) // size
    return np.frombuffer(raw[: n * size], dtype=np.uint8).reshape(n, FRAME_H, FRAME_W, 3)


def _play_segments(clip: ClipDraft) -> tuple[int, int | None]:
    """Index of the segment a clip's play starts in, and of the one it scores in (None
    when the score first showed after a cutaway, where the clip already ends)."""
    scored_at = clip.changes[0].t if clip.changes else None
    for k, (a, b) in enumerate(clip.segments):
        if scored_at is not None and a <= scored_at <= b + 0.01:
            return 0, k
    return 0, None


def refine_clips(
    probe: Probe,
    clips: list[ClipDraft],
    min_keep: float,
    progress: Callable[[float], None] | None = None,
) -> dict:
    """Trim crowd shots and close-ups off the edges of scoring-play clips, in place."""
    plays = [c for c in clips if c.kind != "free_throws" and c.segments]
    if not plays:
        return {"checked": 0, "trimmed": 0, "seconds_removed": 0.0}

    wanted: dict[tuple[int, int], tuple[float, float]] = {}
    for n, clip in enumerate(plays):
        first, last = _play_segments(clip)
        for k in {first, last} - {None}:
            wanted[(n, k)] = (clip.segments[k][0], clip.segments[k][1])

    done = 0

    def fetch(item: tuple[tuple[int, int], tuple[float, float]]) -> tuple[tuple[int, int], np.ndarray]:
        nonlocal done
        key, (a, b) = item
        frames = sample_segment(probe, a, b)
        done += 1
        if progress:
            progress(done / len(wanted))
        return key, frames

    with ThreadPoolExecutor(max_workers=4) as pool:
        sampled = dict(pool.map(fetch, wanted.items()))

    stack = [f for f in sampled.values() if len(f)]
    model = learn_surface(np.concatenate(stack)) if stack else None
    if model is None:
        return {"checked": len(plays), "trimmed": 0, "seconds_removed": 0.0, "skipped": "no playing-surface colour stands out"}

    trimmed, removed = 0, 0.0
    for n, clip in enumerate(plays):
        first, last = _play_segments(clip)
        before = clip.duration
        frames = sampled.get((n, first))
        if frames is not None and len(frames):
            a, b = clip.segments[first]
            lead, tail = trim_edges(cutaway_flags(frames, model), b - a, min_keep)
            if lead:
                clip.segments[first][0] = round(a + lead, 3)
            if tail and last == first:
                clip.segments[first][1] = round(b - tail, 3)
        if last is not None and last != first:
            frames = sampled.get((n, last))
            if frames is not None and len(frames):
                a, b = clip.segments[last]
                _, tail = trim_edges(cutaway_flags(frames, model), b - a, min(min_keep, b - a))
                if tail:
                    clip.segments[last][1] = round(b - tail, 3)
        cut = before - clip.duration
        if cut > 1e-6:
            trimmed += 1
            removed += cut
            if "camera" not in clip.start_cause:
                clip.start_cause += "+camera"
    return {
        "checked": len(plays), "trimmed": trimmed, "seconds_removed": round(removed, 2),
        "surface_share": round(model.typical, 3),
    }
