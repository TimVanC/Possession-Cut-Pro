"""Small interval helpers shared by clip building and export."""

from __future__ import annotations

Interval = tuple[float, float]


def merge(spans: list[Interval], gap: float = 0.0) -> list[Interval]:
    """Union of intervals; spans closer than ``gap`` are joined."""
    out: list[Interval] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(spans: list[Interval], holes: list[Interval]) -> list[Interval]:
    """``spans`` with every part covered by ``holes`` removed."""
    out: list[Interval] = []
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


def total(spans: list[Interval] | list[list[float]]) -> float:
    return sum(b - a for a, b in spans)


def clamp(spans: list[Interval], lo: float, hi: float) -> list[Interval]:
    out = []
    for a, b in spans:
        a, b = max(lo, a), min(hi, b)
        if b > a:
            out.append((a, b))
    return out
