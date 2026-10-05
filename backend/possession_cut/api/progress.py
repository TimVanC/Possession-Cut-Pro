"""What the page shows while a job works: the steps of the task, which one it is on, and
roughly how long is left.

Everything here is derived from what the worker already writes to the job row (task,
stage, progress, when the task started), so it needs nothing new stored.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..db import Job

# Steps per task, in order. Analysis and export steps are recognised by the stage the
# worker reports; calibration reports one stage throughout, so its steps are placed by
# how far along the bar they begin.
ANALYZE_STEPS: list[tuple[str, set[str]]] = [
    ("Read the score bug", {"sampling"}),
    ("Build the game timeline", {"timeline", "ocr_fallback"}),
    ("Match play-by-play", {"pbp", "matching"}),
    ("Trim to the game camera", {"camera"}),
    ("Make thumbnails", {"thumbnails", "proxy"}),
]
EXPORT_STEPS: list[tuple[str, set[str]]] = [
    ("Render the video", {"rendering"}),
    ("Write the caption", {"caption"}),
]
CALIBRATE_STEPS: list[tuple[str, float]] = [
    ("Sample frames", 0.0),
    ("Check saved layouts", 0.2),
    ("Find the score bug", 0.3),
    ("Check the readings", 0.8),
]
MIN_PROGRESS, MIN_ELAPSED = 0.03, 4.0  # before this there is nothing to estimate from


def _active_step(job: Job) -> tuple[list[str], int]:
    """Step labels for the job's task and the index of the one in progress (-1 = not begun)."""
    if job.task == "calibrate":
        labels = [label for label, _ in CALIBRATE_STEPS]
        if job.task_started_at is None:
            return labels, -1
        done = [k for k, (_, begins) in enumerate(CALIBRATE_STEPS) if (job.progress or 0.0) >= begins]
        return labels, done[-1]
    steps = ANALYZE_STEPS if job.task == "analyze" else EXPORT_STEPS if job.task == "export" else []
    labels = [label for label, _ in steps]
    for k, (_, stages) in enumerate(steps):
        if job.stage in stages:
            return labels, k
    return labels, -1


def progress_view(job: Job, now: datetime | None = None) -> dict:
    """``steps`` (label and done/active/todo), ``eta_seconds`` and ``elapsed_seconds``.

    The estimate is the time taken so far scaled by how much of the bar is left. That is
    accurate for the long stretches (reading the bug, rendering), which advance evenly, and
    it is withheld until there is enough to go on.
    """
    if job.task is None:
        return {"steps": [], "eta_seconds": None, "elapsed_seconds": None}
    labels, active = _active_step(job)
    steps = [
        {"label": label, "state": "done" if k < active else "active" if k == active else "todo"}
        for k, label in enumerate(labels)
    ]
    started = job.task_started_at
    if started is None:
        return {"steps": steps, "eta_seconds": None, "elapsed_seconds": None}
    if started.tzinfo is None:  # SQLite hands datetimes back without their zone
        started = started.replace(tzinfo=UTC)
    elapsed = max(0.0, ((now or datetime.now(UTC)) - started).total_seconds())
    fraction = job.progress or 0.0
    eta = None
    if fraction >= MIN_PROGRESS and elapsed >= MIN_ELAPSED:
        eta = elapsed * (1.0 - fraction) / fraction
        eta = float(max(5, round(eta / 5) * 5))  # to the nearest 5 s: steady enough to read
    return {"steps": steps, "eta_seconds": eta, "elapsed_seconds": round(elapsed, 1)}
