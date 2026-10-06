"""Moving a clip's edges: the one rule the API and the worker both apply.

A clip keeps at least 0.2 s of its first and last segment, never starts before the file
and never ends after it.
"""

from __future__ import annotations

MIN_KEEP = 0.2


def move_edges(
    segments: list[list[float]] | list[tuple[float, float]],
    duration: float,
    src_in: float | None = None,
    src_out: float | None = None,
) -> list[list[float]]:
    out = [list(seg) for seg in segments]
    if src_in is not None:
        out[0][0] = round(max(0.0, min(float(src_in), out[0][1] - MIN_KEEP)), 3)
    if src_out is not None:
        out[-1][1] = round(min(duration, max(float(src_out), out[-1][0] + MIN_KEEP)), 3)
    return out


def edit_key(pbp_event_id: str | None, team: str, period: int | None, clock: float | None, kind: str, score_after: int | None) -> tuple:
    """What identifies the same play across two analyses of one game: the play-by-play
    event when there is one, else where and what it was."""
    if pbp_event_id:
        return ("pbp", pbp_event_id)
    return ("bug", team, period, None if clock is None else round(float(clock), 1), kind, score_after)
