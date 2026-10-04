"""HTTP and disk-cache helpers shared by the sport adapters.

League endpoints are free and keyless but flaky: they throttle, time out, and some want
browser-like headers. Every call retries with backoff, and anything that is final (a
finished game's play-by-play) is cached on disk per game ID so it is fetched once.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import get_settings
from .base import PlayByPlayUnavailable

log = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 3,
    timeout: float = 15.0,
    backoff: float = 1.5,
) -> Any:
    """GET a JSON document, retrying timeouts, 429s and 5xx with exponential backoff."""
    last = "no attempt made"
    for attempt in range(attempts):
        try:
            response = httpx.get(
                url, params=params, headers={**BROWSER_HEADERS, **(headers or {})},
                timeout=timeout, follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}"
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    last = "response was not JSON"
            elif response.status_code in (403, 404):
                # not there (old game on a recent-games CDN, wrong id): retrying will not help
                raise PlayByPlayUnavailable(f"{url} returned {response.status_code}")
            else:
                last = f"HTTP {response.status_code}"
        if attempt < attempts - 1:
            time.sleep(backoff * (2**attempt))
    raise PlayByPlayUnavailable(f"{url} failed after {attempts} attempts ({last})")


def cache_dir(sport: str) -> Path:
    path = get_settings().cache_path / sport
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_cache(sport: str, name: str) -> Any | None:
    path = cache_dir(sport) / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_cache(sport: str, name: str, data: Any) -> None:
    path = cache_dir(sport) / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def parse_iso_clock(text: str | None) -> float | None:
    """'PT11M42.00S' -> 702.0 seconds. Also accepts '11:42' and '42.3'."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("PT"):
        minutes, seconds = 0.0, 0.0
        body = text[2:]
        if "M" in body:
            m, body = body.split("M", 1)
            minutes = float(m or 0)
        if body.endswith("S"):
            seconds = float(body[:-1] or 0)
        return minutes * 60 + seconds
    if ":" in text:
        m, s = text.split(":", 1)
        try:
            return int(m) * 60 + float(s)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None
