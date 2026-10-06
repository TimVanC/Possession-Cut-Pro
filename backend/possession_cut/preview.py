"""The preview copy the review screen plays.

A browser cannot play every upload (MKV, TS, AC-3 audio), and on a hosted copy even a
playable file is a poor fit: every clip jump seeks a multi-gigabyte source over the
internet and then waits for the next keyframe, often five seconds of video away. So the
worker makes a small copy with a keyframe every second, at most 480 tall, and the page
plays that instead. It is built in the background from the moment a file is probed, so it
is usually ready before the review starts; until then the page plays the original.

The files live next to the job's other big temporaries (``SCRATCH_DIR``):

    proxy.mp4       the finished copy
    proxy.part.mp4  one being written
    proxy.progress  how far along, for the page
    proxy.failed    why the last attempt failed, so it is not retried forever
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import get_settings


def proxy_path(job_id: int | str) -> Path:
    return get_settings().scratch_path(job_id) / "proxy.mp4"


def part_path(job_id: int | str) -> Path:
    return proxy_path(job_id).with_name("proxy.part.mp4")


def progress_path(job_id: int | str) -> Path:
    return proxy_path(job_id).with_name("proxy.progress")


def failed_path(job_id: int | str) -> Path:
    return proxy_path(job_id).with_name("proxy.failed")


def wants_preview(probe: dict | None) -> bool:
    """Whether a job should get a preview copy: always on a server, else only when the
    browser cannot play the original."""
    if not probe:
        return False
    return get_settings().hosted or not bool(probe.get("browser_playable"))


def preview_state(job_id: int | str, probe: dict | None) -> dict:
    """What the page needs to know: which file plays now, and whether a better one is coming."""
    ready = proxy_path(job_id).exists()
    playable = bool((probe or {}).get("browser_playable"))
    building = not ready and part_path(job_id).exists()
    progress = None
    if building:
        try:
            progress = json.loads(progress_path(job_id).read_text(encoding="utf-8")).get("fraction")
        except (OSError, ValueError, AttributeError):
            progress = None
    failed = None
    if not ready and not building and failed_path(job_id).exists():
        try:
            failed = failed_path(job_id).read_text(encoding="utf-8")[:200] or "failed"
        except OSError:
            failed = "failed"
    return {
        "media_ready": ready or playable,
        "media_source": "proxy" if ready else ("source" if playable else None),
        "preview": {
            "wanted": wants_preview(probe), "ready": ready, "building": building,
            "progress": progress, "failed": failed,
        },
    }
