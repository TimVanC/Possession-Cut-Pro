"""The steps and time estimate shown while a job works."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from possession_cut.api.progress import progress_view
from possession_cut.db import Job

NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)


def job(task: str | None, stage: str = "", progress: float = 0.0, running_for: float | None = None) -> Job:
    started = None if running_for is None else NOW - timedelta(seconds=running_for)
    return Job(source_path="x.mp4", task=task, stage=stage, progress=progress, task_started_at=started)


def states(view: dict) -> list[str]:
    return [s["state"] for s in view["steps"]]


def test_idle_job_has_no_steps_or_estimate():
    assert progress_view(job(None), NOW) == {"steps": [], "eta_seconds": None, "elapsed_seconds": None}


def test_queued_task_lists_its_steps_as_not_begun():
    view = progress_view(job("analyze", stage="queued"), NOW)
    assert [s["label"] for s in view["steps"]][0] == "Read the score bug"
    assert set(states(view)) == {"todo"} and view["eta_seconds"] is None and view["elapsed_seconds"] is None


def test_analysis_steps_follow_the_stage_the_worker_reports():
    reading = progress_view(job("analyze", "sampling", 0.30, running_for=120), NOW)
    assert states(reading) == ["active", "todo", "todo", "todo", "todo"]
    # Claude re-reading unclear frames is part of building the timeline
    rereading = progress_view(job("analyze", "ocr_fallback", 0.77, running_for=330), NOW)
    assert states(rereading) == ["done", "active", "todo", "todo", "todo"]
    camera = progress_view(job("analyze", "camera", 0.85, running_for=400), NOW)
    assert states(camera) == ["done", "done", "done", "active", "todo"]
    proxy = progress_view(job("analyze", "proxy", 0.95, running_for=420), NOW)
    assert states(proxy) == ["done", "done", "done", "done", "active"]


def test_calibration_steps_follow_the_bar():
    assert states(progress_view(job("calibrate", "calibrating", 0.05, running_for=2), NOW)) == ["active", "todo", "todo", "todo"]
    assert states(progress_view(job("calibrate", "calibrating", 0.25, running_for=6), NOW)) == ["done", "active", "todo", "todo"]
    assert states(progress_view(job("calibrate", "calibrating", 0.60, running_for=30), NOW)) == ["done", "done", "active", "todo"]
    assert states(progress_view(job("calibrate", "calibrating", 0.85, running_for=60), NOW)) == ["done", "done", "done", "active"]


def test_export_steps():
    rendering = progress_view(job("export", "rendering", 0.50, running_for=200), NOW)
    assert states(rendering) == ["active", "todo"] and rendering["eta_seconds"] == 200.0
    assert states(progress_view(job("export", "caption", 0.96, running_for=390), NOW)) == ["done", "active"]


def test_estimate_scales_time_so_far_by_what_is_left():
    # 2 minutes in, 30% done: 4 min 40 s to go
    view = progress_view(job("analyze", "sampling", 0.30, running_for=120), NOW)
    assert view["eta_seconds"] == 280.0 and view["elapsed_seconds"] == 120.0
    # rounded to 5 s, and never shown as zero while work remains
    assert progress_view(job("analyze", "sampling", 0.31, running_for=121), NOW)["eta_seconds"] % 5 == 0
    assert progress_view(job("export", "caption", 0.99, running_for=10), NOW)["eta_seconds"] == 5.0


def test_no_estimate_until_there_is_something_to_go_on():
    assert progress_view(job("analyze", "sampling", 0.01, running_for=30), NOW)["eta_seconds"] is None, "barely started"
    assert progress_view(job("analyze", "sampling", 0.50, running_for=2), NOW)["eta_seconds"] is None, "only just running"


def test_start_time_without_a_zone_is_read_as_utc():
    j = job("analyze", "sampling", 0.5, running_for=60)
    j.task_started_at = j.task_started_at.replace(tzinfo=None)  # what SQLite returns
    assert progress_view(j, NOW)["eta_seconds"] == 60.0
