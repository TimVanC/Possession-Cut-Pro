"""The whole product flow through the HTTP API, with the worker run inline:
new job -> calibrate -> analyze -> review edits -> export."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from possession_cut.config import ffmpeg_bin, ffprobe_bin

pytestmark = pytest.mark.video


@pytest.fixture()
def app_env(settings, coverage_video, monkeypatch):
    """A fresh data dir, with the folder holding the synthetic video browsable."""
    from possession_cut import config, db

    video, truth = coverage_video
    monkeypatch.setenv("ALLOWED_ROOTS", str(video.parent))
    config.get_settings.cache_clear()
    db.reset_engine()
    config.get_settings().ensure_dirs()
    from possession_cut.api.main import create_app

    client = TestClient(create_app())
    yield client, video, truth, config.get_settings()
    client.close()


def work(max_tasks: int = 5) -> list[tuple[int, str]]:
    """Run queued tasks the way the worker process would."""
    from possession_cut.worker.__main__ import claim_next, run_one

    done = []
    for _ in range(max_tasks):
        claimed = claim_next()
        if claimed is None:
            break
        run_one(*claimed)
        done.append(claimed)
    return done


def test_health_sports_and_browse(app_env):
    client, video, _, settings = app_env
    h = client.get("/api/health").json()
    assert h["ok"] and h["ffmpeg"] and h["claude"]["configured"] is False and h["worker"] is False
    sports = client.get("/api/sports").json()
    nba = next(s for s in sports if s["key"] == "nba")
    assert len(nba["teams"]) == 30 and nba["periods"] == 4
    assert {f["name"] for f in nba["fields"]} >= {"away_score", "home_score", "clock", "period", "shot_clock"}

    roots = client.get("/api/fs/browse").json()
    assert str(video.parent.resolve()) in roots["roots"]
    listing = client.get("/api/fs/browse", params={"path": str(video.parent)}).json()
    names = [e["name"] for e in listing["entries"]]
    assert video.name in names and not any(n.endswith(".json") for n in names), "only folders and video files"
    assert client.get("/api/fs/browse", params={"path": str(Path(video.anchor) / "Windows")}).status_code == 403


def test_job_creation_errors(app_env, tmp_path):
    client, video, _, _ = app_env
    assert client.post("/api/jobs", json={"source_path": str(video.parent / "nope.mp4")}).status_code == 404
    assert client.post("/api/jobs", json={"source_path": str(Path(video.anchor) / "Windows" / "x.mp4")}).status_code == 403
    small = video.parent / "small_test_clip.mp4"
    subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
                    "-pix_fmt", "yuv420p", str(small)], check=True)
    try:
        r = client.post("/api/jobs", json={"source_path": str(small)})
        assert r.status_code == 400 and "480p" in r.json()["detail"]
    finally:
        small.unlink(missing_ok=True)
    assert client.post("/api/jobs", json={"source_path": str(video), "sport": "curling"}).status_code == 400
    assert client.get("/api/jobs/999").status_code == 404


def test_full_flow(app_env):
    client, video, truth, settings = app_env

    # -- new job
    r = client.post("/api/jobs", json={
        "source_path": str(video), "sport": "nba", "team": "NY",
        "start_spec": {"mode": "start"}, "options": {"include_free_throws": True},
    })
    assert r.status_code == 201
    job = r.json()
    jid = job["id"]
    assert job["status"] == "calibrating" and job["busy"] and job["probe"]["height"] == 720
    assert client.post(f"/api/jobs/{jid}/analyze").status_code == 409, "not calibrated yet"

    # progress stream: the first event carries the current state
    events = client.get(f"/api/jobs/{jid}/events", params={"once": 1})
    assert events.headers["content-type"].startswith("text/event-stream")
    first = next(line for line in events.text.splitlines() if line.startswith("data:"))
    assert json.loads(first[5:])["status"] == "calibrating" and json.loads(first[5:])["busy"]

    # -- calibrate (worker)
    assert work() == [(jid, "calibrate")]
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "ready" and not job["busy"] and job["calibration"]["confidence"] >= 0.9
    cal = client.get(f"/api/jobs/{jid}/calibration").json()
    assert cal["source"] == "local" and len(cal["frames"]) == 12 and cal["teams"] == {"away": "SA", "home": "NY"}
    assert client.get(cal["frames"][0]["url"]).headers["content-type"] == "image/jpeg"
    frame = client.get(f"/api/jobs/{jid}/frame", params={"t": 42.0})
    assert frame.status_code == 200 and frame.content[:2] == b"\xff\xd8"

    # live read-out with the boxes as drawn, then with a deliberately wrong clock box
    good = client.post(f"/api/jobs/{jid}/calibration/preview", json={"bug": cal["bug"], "fields": cal["fields"], "frame": 0}).json()
    assert good["reads"]["clock"]["ok"] and good["reads"]["home_score"]["ok"]
    wrong = dict(cal["fields"], clock=cal["fields"]["away_label"])
    bad = client.post(f"/api/jobs/{jid}/calibration/preview", json={"bug": cal["bug"], "fields": wrong, "frame": 0}).json()
    assert not bad["reads"]["clock"]["ok"]

    # "Looks right": confirm and save as a template
    saved = client.put(f"/api/jobs/{jid}/calibration", json={
        "bug": cal["bug"], "fields": cal["fields"], "confirmed": True, "template_name": "ESPN NBA test", "broadcaster": "ESPN",
    }).json()
    assert saved["confirmed"] and saved["template_name"] == "ESPN NBA test"
    templates = client.get("/api/templates").json()
    assert [t["name"] for t in templates] == ["ESPN NBA test"]
    assert client.get(templates[0]["image"]).headers["content-type"] == "image/png"

    # -- analyze (worker)
    assert client.post(f"/api/jobs/{jid}/analyze").json()["status"] == "analyzing"
    assert work() == [(jid, "analyze")]
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "review", job["error"]
    summary = job["summary"]
    assert summary["follow_side"] == "home" and summary["clips"] == 9
    assert summary["pbp"]["available"] and summary["pbp"]["unmatched_pbp"] == 0 and summary["unmatched_pbp"] == []
    assert summary["final_score"] == {"away": 107, "home": 109}
    assert summary["suggested_title"] == "Every NY bucket vs SA"
    for name in ("probe.json", "calibration.json", "timeline.parquet", "pbp.json", "events.json", "cutlist.json"):
        assert (settings.jobs_path / str(jid) / name).exists(), name

    # -- review: clip list, toggle, nudge, persistence
    clips = client.get(f"/api/jobs/{jid}/clips").json()
    assert len(clips) == 9 and all(c["enabled"] and c["scorer"] and c["description"] for c in clips)
    truth_clips = truth["cutlists"]["home"]
    for c, t in zip(clips, truth_clips, strict=True):
        assert (c["score_before"], c["score_after"], c["points"], c["kind"]) == (t["score_before"], t["score_after"], t["points"], t["kind"])
        assert abs(c["src_out"] - t["src_out"]) <= 1.0
    assert client.get(clips[0]["thumbnail"]).headers["content-type"] == "image/jpeg"

    c1 = clips[1]
    nudged = client.patch(f"/api/clips/{c1['id']}", json={"src_in": c1["src_in"] + 0.5, "src_out": c1["src_out"] - 0.5}).json()
    assert nudged["src_in"] == pytest.approx(c1["src_in"] + 0.5) and nudged["src_out"] == pytest.approx(c1["src_out"] - 0.5)
    assert nudged["edited"] and nudged["duration"] == pytest.approx(c1["duration"] - 1.0)
    for c in clips[2:]:
        client.patch(f"/api/clips/{c['id']}", json={"enabled": False})
    reloaded = client.get(f"/api/jobs/{jid}/clips").json()
    assert [c["enabled"] for c in reloaded] == [True, True] + [False] * 7, "edits persist across reloads"
    assert reloaded[1]["src_in"] == nudged["src_in"]
    crazy = client.patch(f"/api/clips/{c1['id']}", json={"src_in": 99999}).json()
    assert crazy["src_in"] < crazy["src_out"], "an in point cannot pass the out point"
    reset = client.post(f"/api/clips/{c1['id']}/reset").json()
    assert reset["src_in"] == c1["src_in"] and reset["src_out"] == c1["src_out"] and not reset["edited"]

    # every clip at once: the clips in the cut move together, the rest stay; a reset keeps the toggles
    before = client.get(f"/api/jobs/{jid}/clips").json()
    moved = client.post(f"/api/jobs/{jid}/clips/nudge", json={"edge": "in", "delta": -0.5}).json()
    for b, m in zip(before, moved, strict=True):
        if b["enabled"]:
            assert m["src_in"] == pytest.approx(max(0.0, b["src_in"] - 0.5)) and m["edited"]
        else:
            assert m["src_in"] == b["src_in"] and not m["edited"]
    moved = client.post(f"/api/jobs/{jid}/clips/nudge", json={"edge": "out", "delta": 0.5}).json()
    assert moved[0]["src_out"] == pytest.approx(before[0]["src_out"] + 0.5)
    assert client.post(f"/api/jobs/{jid}/clips/nudge", json={"edge": "sideways", "delta": 0.5}).status_code == 422
    assert client.post(f"/api/jobs/{jid}/clips/nudge", json={"edge": "in", "delta": 60}).status_code == 422
    back = client.post(f"/api/jobs/{jid}/clips/reset").json()
    assert [c["enabled"] for c in back] == [c["enabled"] for c in before], "which clips are on is untouched"
    assert all(not c["edited"] for c in back) and back[0]["src_in"] == before[0]["src_in"]

    # several edits in one request (what undo and "turn these off" send); a stranger's id is refused
    a, b = back[0], back[1]
    bulk = client.post(
        f"/api/jobs/{jid}/clips/bulk",
        json={"updates": [{"id": a["id"], "enabled": False, "src_in": a["src_in"] + 1.0}, {"id": b["id"], "src_out": b["src_out"] - 1.0}]},
    ).json()
    assert len(bulk) == len(back)
    assert not bulk[0]["enabled"] and bulk[0]["src_in"] == pytest.approx(a["src_in"] + 1.0)
    assert bulk[1]["enabled"] and bulk[1]["src_out"] == pytest.approx(b["src_out"] - 1.0)
    assert client.post(f"/api/jobs/{jid}/clips/bulk", json={"updates": [{"id": 999999, "enabled": True}]}).status_code == 404
    client.post(f"/api/jobs/{jid}/clips/bulk", json={"updates": [{"id": a["id"], "enabled": True, "src_in": a["src_in"]}, {"id": b["id"], "src_out": b["src_out"]}]})

    # the source streams with HTTP range support
    part = client.get(f"/api/media/{jid}/source", headers={"Range": "bytes=100-299"})
    assert part.status_code == 206 and len(part.content) == 200
    assert part.headers["content-range"].startswith("bytes 100-299/")

    # -- export (worker)
    exp = client.post(f"/api/jobs/{jid}/export", json={"title": "Knicks comeback vs Spurs", "caption": "Test game"})
    assert exp.status_code == 201 and exp.json()["status"] == "pending"
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "exporting"
    assert work() == [(jid, "export")]
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "done", job["error"]
    (export,) = client.get(f"/api/jobs/{jid}/exports").json()
    assert export["status"] == "done" and export["size_bytes"] > 100_000
    out = Path(export["path"])
    assert out.parent == settings.exports_path and out.exists()
    probe = json.loads(subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_streams", str(out)],
        capture_output=True, text=True, check=True).stdout)
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert export["duration"] == pytest.approx(reloaded[0]["duration"] + c1["duration"], abs=0.2)
    assert client.get(export["url"], headers={"Range": "bytes=0-9"}).status_code == 206
    caption = client.get(f"/api/exports/{export['id']}/caption").text
    assert "NY" in caption and "#" in caption and export["settings"]["caption_source"] == "template"
    cutlist = json.loads(Path(export["cutlist_path"]).read_text(encoding="utf-8"))
    assert cutlist["title"] == "Knicks comeback vs Spurs" and len(cutlist["clips"]) == 2
    assert {"src_in", "src_out", "game_clock", "score_after", "scorer", "confidence"} <= set(cutlist["clips"][0])
    assert Path(export["caption_path"]).read_text(encoding="utf-8").strip() == caption.strip()

    # -- a second job on the same broadcast reuses the template
    j2 = client.post("/api/jobs", json={"source_path": str(video), "team": "home"}).json()
    work()
    cal2 = client.get(f"/api/jobs/{j2['id']}/calibration").json()
    assert cal2["source"] == "template" and cal2["template_name"] == "ESPN NBA test"

    # -- delete: artifacts go, the source file and the exports stay
    assert client.delete(f"/api/jobs/{jid}").json() == {"deleted": jid}
    assert not (settings.jobs_path / str(jid)).exists()
    assert video.exists() and out.exists()
    assert client.get(f"/api/jobs/{jid}").status_code == 404
    assert client.delete(f"/api/templates/{templates[0]['id']}").json()["deleted"] == templates[0]["id"]
    assert client.get("/api/templates").json() == []


def test_auto_run_start_and_reanalysis_reuses_the_samples(app_env):
    client, video, truth, settings = app_env
    # the low point per the scripted play-by-play: the last Spurs score that made the gap largest
    low, gap = None, 0
    for ev in truth["pbp"]:
        d = ev["score_away"] - ev["score_home"]
        if ev["team"] == "SA" and d > 0 and d >= gap:
            low, gap = ev, d
    assert low is not None

    preview = client.get("/api/games/run-start", params={
        "sport": "nba", "game_id": "synthetic", "team": "NY", "source_path": str(video)}).json()
    assert preview["available"] and preview["trailed"] and preview["deficit"] == gap
    assert f"Down {gap}" in preview["label"] and preview["period"] == low["period"]
    assert client.get("/api/games/run-start", params={
        "sport": "nba", "game_id": "synthetic", "team": "BOS", "source_path": str(video)}).json()["available"] is False

    jid = client.post("/api/jobs", json={"source_path": str(video), "team": "NY", "start_spec": {"mode": "auto_run"}}).json()["id"]
    work()
    client.post(f"/api/jobs/{jid}/analyze")
    work()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "review", job["error"]
    run = job["summary"]["run_start"]
    assert run["deficit"] == gap and run["score_home"] == low["score_home"]
    assert f"down {gap}" in job["summary"]["start_label"]
    clips = client.get(f"/api/jobs/{jid}/clips").json()
    assert clips and clips[0]["score_before"] == low["score_home"], "the cut starts with the first score after the low point"
    assert sum(c["points"] for c in clips) == 109 - low["score_home"], "and has every score after it"

    # change the options and re-run: the file is not sampled again
    client.patch(f"/api/jobs/{jid}", json={"start_spec": {"mode": "start"}, "options": {"include_opponent": True}})
    client.post(f"/api/jobs/{jid}/analyze")
    work()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["summary"]["sampler"] == {"cached": True, "samples": 770}
    assert job["summary"]["clips"] == 14 and {c["team"] for c in client.get(f"/api/jobs/{jid}/clips").json()} == {"away", "home"}


def test_cancel_and_failure_paths(app_env):
    client, video, _, _ = app_env
    jid = client.post("/api/jobs", json={"source_path": str(video)}).json()["id"]
    cancelled = client.post(f"/api/jobs/{jid}/cancel").json()
    assert cancelled["status"] == "draft" and not cancelled["busy"] and work() == []

    # a job whose source disappears fails cleanly instead of hanging
    copy = video.parent / "vanishing_test_copy.mp4"
    shutil.copyfile(video, copy)
    try:
        j2 = client.post("/api/jobs", json={"source_path": str(copy)}).json()["id"]
    finally:
        copy.unlink()
    work()
    failed = client.get(f"/api/jobs/{j2}").json()
    assert failed["status"] == "failed" and failed["error"] and not failed["busy"] and not failed["source_exists"]
    assert client.post(f"/api/jobs/{j2}/calibrate").status_code == 404


def test_inbox_watcher_creates_one_draft_per_finished_file(app_env):
    client, video, _, settings = app_env
    from possession_cut.worker.inbox import InboxWatcher

    target = settings.inbox_path / "dropped_game.mp4"
    subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-t", "2", "-i", str(video), "-c", "copy", str(target)], check=True)
    (settings.inbox_path / "notes.txt").write_text("not a video")
    watcher = InboxWatcher()
    assert watcher.scan() == [], "first sighting: might still be copying"
    with open(target, "ab") as f:
        f.write(b"\0" * 10)
    assert watcher.scan() == [], "size changed: still being written"
    created = watcher.scan()
    assert len(created) == 1 and watcher.scan() == [], "one draft, once"
    draft = client.get(f"/api/jobs/{created[0]}").json()
    assert draft["status"] == "draft" and draft["from_inbox"] and draft["source_name"] == "dropped_game.mp4"
    box = client.get("/api/inbox").json()
    assert box["waiting"] == 1 and box["drafts"] == created
