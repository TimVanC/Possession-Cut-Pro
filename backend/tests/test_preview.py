"""The preview copy the review screen plays, and when one is made."""

from __future__ import annotations

import json
import subprocess

import pytest

from possession_cut import config, preview
from possession_cut.config import ffprobe_bin
from possession_cut.pipeline.export import make_proxy

PLAYABLE = {"browser_playable": True}
MKV = {"browser_playable": False}


def test_on_this_computer_a_playable_file_plays_as_it_is(settings):
    state = preview.preview_state(7, PLAYABLE)
    assert state["media_ready"] and state["media_source"] == "source"
    assert state["preview"] == {"wanted": False, "ready": False, "building": False, "progress": None, "failed": None}
    assert preview.wants_preview(MKV) and not preview.wants_preview(None)
    assert not preview.preview_state(8, MKV)["media_ready"]


def test_a_hosted_copy_wants_a_preview_and_the_page_follows_its_progress(settings, monkeypatch):
    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.setenv("APP_PASSWORD", "secret")
    config.get_settings.cache_clear()
    try:
        assert preview.wants_preview(PLAYABLE), "on a server even a playable file gets a copy"

        preview.part_path(7).write_bytes(b"")
        preview.progress_path(7).write_text(json.dumps({"fraction": 0.4}), encoding="utf-8")
        state = preview.preview_state(7, PLAYABLE)
        assert state["media_source"] == "source", "the original plays while the copy is made"
        assert state["preview"]["building"] and state["preview"]["progress"] == 0.4

        preview.part_path(7).replace(preview.proxy_path(7))
        state = preview.preview_state(7, PLAYABLE)
        assert state["media_source"] == "proxy" and state["preview"]["ready"] and not state["preview"]["building"]

        preview.proxy_path(7).unlink()
        preview.failed_path(7).write_text("ExportError: boom", encoding="utf-8")
        state = preview.preview_state(7, MKV)
        assert not state["media_ready"] and state["media_source"] is None
        assert state["preview"]["failed"].startswith("ExportError") and state["preview"]["wanted"]
    finally:
        config.get_settings.cache_clear()


def _ffprobe(args: list[str]) -> str:
    return subprocess.run([ffprobe_bin(), "-v", "error", *args], capture_output=True, text=True, check=True).stdout


@pytest.mark.video
def test_the_seekable_copy_has_a_keyframe_every_second_and_is_small(coverage_probe, tmp_path):
    out = make_proxy(coverage_probe, tmp_path / "proxy.mp4", seekable=True, threads=2)
    raw = _ffprobe(["-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(out)])
    keys = [float(x.strip(",")) for x in raw.split() if x.strip(",")]
    gaps = [b - a for a, b in zip(keys, keys[1:], strict=False)]
    assert len(keys) >= coverage_probe.duration - 2 and max(gaps) <= 1.1, "a jump anywhere lands within a second of video"
    height = int(_ffprobe(["-select_streams", "v:0", "-show_entries", "stream=height", "-of", "csv=p=0", str(out)]).strip().strip(","))
    assert height <= 480
    assert out.stat().st_size < coverage_probe.size_bytes


@pytest.mark.video
def test_the_worker_makes_the_copy_for_a_hosted_job_and_then_leaves_it_alone(settings, monkeypatch, coverage_probe):
    from possession_cut.db import Job, session_scope
    from possession_cut.worker import preview as worker_preview

    monkeypatch.setenv("HOSTED", "1")
    monkeypatch.setenv("APP_PASSWORD", "secret")
    config.get_settings.cache_clear()
    try:
        with session_scope() as s:
            s.add(Job(id=3, source_path=coverage_probe.path, probe=coverage_probe.to_dict()))
            s.add(Job(id=4, source_path="gone.mp4", probe=coverage_probe.to_dict()))  # its file is gone: skipped
        found = worker_preview.next_job()
        assert found is not None and found[0] == 3
        assert worker_preview.build(*found)
        state = preview.preview_state(3, coverage_probe.to_dict())
        assert state["media_source"] == "proxy" and state["preview"]["ready"] and not state["preview"]["building"]
        assert not preview.progress_path(3).exists()
        assert worker_preview.next_job() is None, "done once; the job with no file is never picked"
    finally:
        config.get_settings.cache_clear()
