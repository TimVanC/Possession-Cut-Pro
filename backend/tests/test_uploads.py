"""Uploading a game file through the page: pieces in, a draft job out."""

from __future__ import annotations

import os
import subprocess
import time
from collections import namedtuple
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from possession_cut.config import ffmpeg_bin


@pytest.fixture(scope="module")
def small_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("upload_src") / "tiny.mp4"
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=10:duration=2",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture()
def client(settings):
    from possession_cut.api.main import create_app

    with TestClient(create_app()) as c:
        yield c


def send(client: TestClient, data: bytes, name: str, piece: int = 4096) -> dict:
    """Upload ``data`` the way the page does. Returns the last response's JSON."""
    started = client.post("/api/uploads", json={"name": name, "size": len(data)})
    assert started.status_code == 201, started.text
    upload = started.json()
    assert upload["received"] == 0 and upload["chunk_size"] > 0
    offset = 0
    while offset < len(data):
        r = client.put(f"/api/uploads/{upload['id']}", params={"offset": offset}, content=data[offset : offset + piece])
        assert r.status_code == 200, r.text
        offset = r.json()["received"]
    return upload


def test_upload_becomes_a_draft_job_and_goes_with_it(client, settings, small_video):
    data = small_video.read_bytes()
    upload = send(client, data, "NBA_20260610_SAS_NYK.mp4")
    r = client.post(f"/api/uploads/{upload['id']}/complete")
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["status"] == "draft" and job["uploaded"] is True and job["source_name"] == "NBA_20260610_SAS_NYK.mp4"
    assert job["probe"]["height"] == 480 and job["probe"]["size_bytes"] == len(data)
    stored = Path(job["source_path"])
    assert stored.parent == settings.uploads_path.resolve() and stored.read_bytes() == data
    assert not list(settings.uploads_path.glob("*.part")) and not list(settings.uploads_path.glob("*.json"))

    # the uploaded copy is reachable like any other source, and set up like an inbox draft
    assert client.get("/api/jobs").json()[0]["id"] == job["id"]
    patched = client.patch(f"/api/jobs/{job['id']}", json={"sport": "nba", "team": "home"})
    assert patched.status_code == 200 and patched.json()["team"] == "home"

    # a second upload of the same name does not overwrite the first
    again = send(client, data, "NBA_20260610_SAS_NYK.mp4")
    second = client.post(f"/api/uploads/{again['id']}/complete").json()
    assert Path(second["source_path"]).name == "NBA_20260610_SAS_NYK (2).mp4" and stored.exists()

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert not stored.exists(), "the app's own copy is deleted with the job"
    assert Path(second["source_path"]).exists(), "another job's upload is untouched"


def test_a_file_picked_from_disk_is_never_deleted(client, settings, small_video, tmp_path):
    mine = tmp_path / "my_game.mp4"
    mine.write_bytes(small_video.read_bytes())
    job = client.post("/api/jobs", json={"source_path": str(mine), "calibrate": False}).json()
    assert job["uploaded"] is False
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert mine.exists()


def test_upload_resumes_after_a_broken_piece(client, small_video):
    data = small_video.read_bytes()
    upload = client.post("/api/uploads", json={"name": "game.mkv", "size": len(data)}).json()
    uid = upload["id"]
    assert client.put(f"/api/uploads/{uid}", params={"offset": 0}, content=data[:5000]).json()["received"] == 5000

    # the page retries a piece it thinks was lost: the engine says where it really is
    retry = client.put(f"/api/uploads/{uid}", params={"offset": 0}, content=data[:5000])
    assert retry.status_code == 409 and retry.json()["detail"]["received"] == 5000
    assert client.get(f"/api/uploads/{uid}").json()["received"] == 5000

    # finishing early is refused, and so is sending more than was announced
    early = client.post(f"/api/uploads/{uid}/complete")
    assert early.status_code == 409 and early.json()["detail"]["received"] == 5000
    over = client.put(f"/api/uploads/{uid}", params={"offset": 5000}, content=data[5000:] + b"extra")
    assert over.status_code == 400
    assert client.get(f"/api/uploads/{uid}").json()["received"] == 5000, "the oversized piece was rolled back"

    assert client.put(f"/api/uploads/{uid}", params={"offset": 5000}, content=data[5000:]).json()["received"] == len(data)
    assert client.post(f"/api/uploads/{uid}/complete").status_code == 201


def test_upload_refuses_what_it_cannot_use(client, settings, monkeypatch):
    assert client.post("/api/uploads", json={"name": "notes.txt", "size": 10}).status_code == 400
    assert client.post("/api/uploads", json={"name": "game.mp4", "size": 0}).status_code == 422
    assert client.get("/api/uploads/not-an-id").status_code == 404
    assert client.get("/api/uploads/" + "0" * 32).status_code == 404

    # a file with a video name that is not a video is removed again
    junk = b"this is not a video" * 100
    upload = send(client, junk, "broken.mp4")
    r = client.post(f"/api/uploads/{upload['id']}/complete")
    assert r.status_code == 400
    assert not list(settings.uploads_path.iterdir()), "nothing is left behind"
    assert client.get("/api/jobs").json() == []

    # no room on the disk: said before a single byte is sent
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("possession_cut.api.uploads.shutil.disk_usage", lambda _p: usage(100, 99, 3 * 1024**3))
    full = client.post("/api/uploads", json={"name": "game.mp4", "size": 5 * 1024**3})
    assert full.status_code == 507 and "disk space" in full.json()["detail"]


def test_cancelled_and_abandoned_uploads_are_cleaned_up(client, settings):
    from possession_cut.api.uploads import drop_stale

    upload = client.post("/api/uploads", json={"name": "game.mp4", "size": 1000}).json()
    client.put(f"/api/uploads/{upload['id']}", params={"offset": 0}, content=b"x" * 400)
    assert client.delete(f"/api/uploads/{upload['id']}").status_code == 200
    assert not list(settings.uploads_path.iterdir())
    assert client.put(f"/api/uploads/{upload['id']}", params={"offset": 0}, content=b"x").status_code == 404

    old = client.post("/api/uploads", json={"name": "old.mp4", "size": 1000}).json()
    fresh = client.post("/api/uploads", json={"name": "fresh.mp4", "size": 1000}).json()
    part = settings.uploads_path / f"{old['id']}.part"
    two_days_ago = time.time() - 2 * 24 * 3600
    os.utime(part, (two_days_ago, two_days_ago))
    assert drop_stale(settings.uploads_path) == 1
    assert client.get(f"/api/uploads/{old['id']}").status_code == 404
    assert client.get(f"/api/uploads/{fresh['id']}").status_code == 200


def test_upload_names_are_made_safe():
    from possession_cut.api.uploads import safe_name, unique_path

    assert safe_name("C:\\Users\\me\\Downloads\\Game 7.MP4") == "Game 7.mp4"
    assert safe_name("../../etc/passwd.mkv") == "passwd.mkv"
    assert safe_name('what<>:"|?*.mp4') == "what_______.mp4"
    assert safe_name(".mp4") == "game.mp4" or safe_name(".mp4").endswith("mp4")
    assert len(safe_name("x" * 400 + ".mp4")) <= 160
    assert unique_path(Path("."), "definitely-not-here.mp4").name == "definitely-not-here.mp4"


def test_uploads_stay_out_of_cloud_synced_folders(tmp_path, monkeypatch):
    from possession_cut import config

    assert config._is_synced(Path("C:/Users/me/OneDrive/Desktop/app/data"))
    assert config._is_synced(Path("/Users/me/Dropbox/app/data"))
    assert config._is_synced(Path("C:/Users/me/OneDrive - Contoso/app/data"))
    assert not config._is_synced(tmp_path / "data") or "onedrive" in str(tmp_path).lower()

    monkeypatch.setenv("UPLOADS_DIR", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "OneDrive" / "app" / "data"))
    config.get_settings.cache_clear()
    try:
        synced = config.get_settings()
        assert synced.uploads_path == config._local_app_dir() / "uploads"
        monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "elsewhere"))
        config.get_settings.cache_clear()
        assert config.get_settings().uploads_path == tmp_path / "elsewhere", "an explicit folder always wins"
    finally:
        config.get_settings.cache_clear()


def test_a_briefly_locked_file_is_waited_out(client, monkeypatch):
    """Windows antivirus and indexers hold a growing file for a moment; that must not
    fail the piece."""
    upload = client.post("/api/uploads", json={"name": "game.mp4", "size": 2000}).json()
    uid = upload["id"]
    real_open = Path.open
    refusals = {"left": 3}

    def flaky_open(self, mode="r", *args, **kwargs):
        if mode == "ab" and refusals["left"] > 0:
            refusals["left"] -= 1
            raise PermissionError(13, "Permission denied")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    r = client.put(f"/api/uploads/{uid}", params={"offset": 0}, content=b"x" * 1000)
    assert r.status_code == 200 and r.json()["received"] == 1000 and refusals["left"] == 0

    # still locked after the wait: the page is told to try again, nothing is lost
    monkeypatch.setattr("possession_cut.api.uploads.LOCK_WAIT_SECONDS", 0.2)
    refusals["left"] = 10_000
    busy = client.put(f"/api/uploads/{uid}", params={"offset": 1000}, content=b"x" * 1000)
    assert busy.status_code == 503
    refusals["left"] = 0
    assert client.get(f"/api/uploads/{uid}").json()["received"] == 1000
    assert client.put(f"/api/uploads/{uid}", params={"offset": 1000}, content=b"x" * 1000).json()["received"] == 2000
