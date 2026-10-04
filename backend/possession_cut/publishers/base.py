from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class PublishRequest:
    video_path: Path
    caption: str
    title: str = ""
    scheduled_at: str | None = None  # ISO 8601; None = publish now
    extra: dict = field(default_factory=dict)


@dataclass
class PublishResult:
    ok: bool
    url: str | None = None
    remote_id: str | None = None
    error: str | None = None


@runtime_checkable
class Publisher(Protocol):
    """One destination platform. Implementations live next to this file."""

    name: str

    def is_configured(self) -> bool: ...

    def publish(self, request: PublishRequest) -> PublishResult: ...


_REGISTRY: dict[str, Publisher] = {}


def register_publisher(publisher: Publisher) -> None:
    _REGISTRY[publisher.name] = publisher


def get_publisher(name: str) -> Publisher:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise LookupError(f"No publisher named {name!r} is registered.") from None
