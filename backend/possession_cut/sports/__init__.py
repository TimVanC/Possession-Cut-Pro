"""Sport adapters. ``get_adapter('nba')`` is the only entry point the pipeline uses."""

from __future__ import annotations

from functools import lru_cache

from .base import (
    FieldSpec,
    Game,
    PlayByPlayUnavailable,
    ScoreChange,
    ScoringEvent,
    SportAdapter,
)

__all__ = [
    "FieldSpec",
    "Game",
    "PlayByPlayUnavailable",
    "ScoreChange",
    "ScoringEvent",
    "SportAdapter",
    "available_sports",
    "get_adapter",
]


def _registry() -> dict[str, type[SportAdapter]]:
    from .mlb import MLBAdapter
    from .nba import NBAAdapter
    from .nfl import NFLAdapter
    from .nhl import NHLAdapter

    return {"nba": NBAAdapter, "nfl": NFLAdapter, "nhl": NHLAdapter, "mlb": MLBAdapter}


def available_sports() -> list[dict[str, str]]:
    return [{"key": key, "name": cls.name} for key, cls in _registry().items()]


@lru_cache
def get_adapter(sport: str) -> SportAdapter:
    try:
        return _registry()[sport.lower()]()
    except KeyError:
        raise ValueError(f"Unknown sport {sport!r}. Known: {', '.join(_registry())}") from None
