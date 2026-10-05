"""A copy running on a server: one password in front of everything, no view of the
server's disk, and a disk that does not fill up."""

from __future__ import annotations

import os
import subprocess
import time
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from possession_cut.config import ffmpeg_bin

PASSWORD = "correct horse battery staple"


def make_client(monkeypatch, **env) -> TestClient:
    from possession_cut import config, db
    from possession_cut.api import auth
    from possession_cut.api.main import create_app

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.get_settings.cache_clear()
    db.reset_engine()
    auth._failures.clear()
    return TestClient(create_app())


@pytest.fixture()
def hosted(settings, monkeypatch):
    with make_client(monkeypatch, HOSTED="1", APP_PASSWORD=PASSWORD) as c:
        yield c


@pytest.fixture(scope="module")
def small_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("hosted_src") / "tiny.mp4"
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=10:duration=2",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


# -- signing in ------------------------------------------------------------------------------


def test_nothing_is_served_without_signing_in(hosted):
    health = hosted.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["auth"] == {"required": True, "authenticated": False, "configured": True} and body["hosted"] is True
    assert set(body) == {"ok", "version", "hosted", "auth"}, "nothing about the machine before sign-in"

    for method, path in (
        ("get", "/api/jobs"), ("get", "/api/sports"), ("get", "/api/templates"), ("post", "/api/uploads"),
        ("get", "/api/media/1/source"), ("get", "/api/exports/1/file"), ("get", "/api/jobs/1/events"),
        ("delete", "/api/jobs/1"), ("get", "/api/fs/browse"),
    ):
        r = getattr(hosted, method)(path)
        assert r.status_code == 401, (method, path, r.status_code)


def test_right_password_opens_the_app_and_signing_out_closes_it(hosted):
    wrong = hosted.post("/api/login", json={"password": "guess"})
    assert wrong.status_code == 401 and "pc_session" not in wrong.cookies
    ok = hosted.post("/api/login", json={"password": PASSWORD})
    assert ok.status_code == 200 and ok.json() == {"authenticated": True}
    cookie = ok.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "max-age=" in cookie

    assert hosted.get("/api/jobs").status_code == 200
    body = hosted.get("/api/health").json()
    assert body["auth"]["authenticated"] is True and "ffmpeg" in body and "uploads_dir" in body

    hosted.post("/api/logout")
    assert hosted.get("/api/jobs").status_code == 401


def test_session_cookie_is_secure_behind_an_https_proxy(hosted):
    r = hosted.post("/api/login", json={"password": PASSWORD}, headers={"x-forwarded-proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_forged_expired_and_stale_sessions_are_refused(settings, monkeypatch):
    from possession_cut import config
    from possession_cut.api import auth

    with make_client(monkeypatch, HOSTED="1", APP_PASSWORD=PASSWORD) as c:
        s = config.get_settings()
        token = auth.make_token(s)
        assert auth.valid_token(s, token)
        expires, _, signature = token.partition(".")
        assert not auth.valid_token(s, f"{int(expires) + 999}.{signature}"), "expiry is signed"
        assert not auth.valid_token(s, f"{expires}.{'0' * 64}")
        assert not auth.valid_token(s, "nonsense") and not auth.valid_token(s, "") and not auth.valid_token(s, None)
        assert not auth.valid_token(s, auth.make_token(s, now=time.time() - auth.SESSION_SECONDS - 5)), "expired"
        c.cookies.set("pc_session", f"{expires}.{'0' * 64}")
        assert c.get("/api/jobs").status_code == 401

    # changing the password signs everyone out
    with make_client(monkeypatch, APP_PASSWORD="a new password") as c2:
        c2.cookies.set("pc_session", token)
        assert c2.get("/api/jobs").status_code == 401


def test_guessing_is_slowed_down(hosted):
    for _ in range(8):
        assert hosted.post("/api/login", json={"password": "nope"}).status_code == 401
    blocked = hosted.post("/api/login", json={"password": PASSWORD})
    assert blocked.status_code == 429, "even the right password waits once the limit is hit"
    other = hosted.post("/api/login", json={"password": PASSWORD}, headers={"x-forwarded-for": "203.0.113.9"})
    assert other.status_code == 200, "someone else is not locked out"


def test_hosted_copy_without_a_password_serves_nothing(settings, monkeypatch):
    with make_client(monkeypatch, HOSTED="1", APP_PASSWORD="") as c:
        body = c.get("/api/health").json()
        assert body["auth"] == {"required": True, "authenticated": False, "configured": False}
        blocked = c.get("/api/jobs")
        assert blocked.status_code == 503 and "APP_PASSWORD" in blocked.json()["detail"]
        assert c.post("/api/login", json={"password": "anything"}).status_code == 503


def test_local_copy_is_unchanged(settings, monkeypatch):
    with make_client(monkeypatch) as c:
        assert c.get("/api/jobs").status_code == 200
        body = c.get("/api/health").json()
        assert body["hosted"] is False and body["auth"] == {"required": False, "authenticated": True, "configured": True}
        assert c.get("/api/fs/browse").status_code == 200

    # a password can also be put on a local copy
    with make_client(monkeypatch, APP_PASSWORD=PASSWORD) as c:
        assert c.get("/api/jobs").status_code == 401
        c.post("/api/login", json={"password": PASSWORD})
        assert c.get("/api/fs/browse").status_code == 200, "still local: the file picker works"


# -- no view of the server's disk -----------------------------------------------------------


def test_hosted_copy_takes_uploads_only(hosted, small_video, tmp_path):
    hosted.post("/api/login", json={"password": PASSWORD})
    assert hosted.get("/api/fs/browse").status_code == 403
    assert hosted.get("/api/fs/browse", params={"path": str(tmp_path)}).status_code == 403

    # a file that happens to be on the server cannot be opened by path
    on_disk = tmp_path / "someone_elses.mp4"
    on_disk.write_bytes(small_video.read_bytes())
    assert hosted.post("/api/jobs", json={"source_path": str(on_disk), "calibrate": False}).status_code == 403
    assert hosted.post("/api/jobs", json={"source_path": "/etc/passwd"}).status_code in (403, 404)

    data = small_video.read_bytes()
    upload = hosted.post("/api/uploads", json={"name": "game.mp4", "size": len(data)}).json()
    hosted.put(f"/api/uploads/{upload['id']}", params={"offset": 0}, content=data)
    job = hosted.post(f"/api/uploads/{upload['id']}/complete").json()
    assert job["uploaded"] is True and job["status"] == "draft"
    assert hosted.get(f"/api/media/{job['id']}/source").status_code == 200
    assert hosted.post("/api/exports/1/reveal").status_code == 403


def test_big_temporary_files_go_to_the_scratch_folder(settings, monkeypatch, tmp_path):
    from possession_cut import config

    monkeypatch.setenv("SCRATCH_DIR", str(tmp_path / "scratch"))
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.scratch_path(7) == tmp_path / "scratch" / "7" and s.scratch_path(7).is_dir()
    monkeypatch.setenv("SCRATCH_DIR", "")
    config.get_settings.cache_clear()
    assert config.get_settings().scratch_path(7) == config.get_settings().job_dir(7), "default: with the job"


# -- the disk -----------------------------------------------------------------------------------


def _export(session, job_id: int, path: Path, age_days: float, size: int = 1000):
    from possession_cut.db import Export, Job

    if session.get(Job, job_id) is None:  # an export belongs to a job
        session.add(Job(id=job_id, source_path="game.mp4"))
        session.flush()
    path.write_bytes(b"x" * size)
    path.with_name(path.stem + ".cutlist.json").write_text("{}")
    path.with_name(path.stem + ".caption.txt").write_text("caption")
    export = Export(job_id=job_id, status="done", path=str(path), created_at=datetime.now(UTC) - timedelta(days=age_days))
    session.add(export)
    session.flush()
    return export.id


def test_uploads_and_exports_past_their_shelf_life_are_removed(settings, monkeypatch):
    from possession_cut import config, db
    from possession_cut.db import Job, session_scope
    from possession_cut.worker import janitor

    monkeypatch.setenv("UPLOAD_RETENTION_DAYS", "7")
    monkeypatch.setenv("EXPORT_RETENTION_DAYS", "30")
    config.get_settings.cache_clear()
    db.reset_engine()
    s = config.get_settings()
    s.ensure_dirs()
    old_upload, fresh_upload, busy_upload = (s.uploads_path / n for n in ("old.mp4", "fresh.mp4", "busy.mp4"))
    orphan, unfinished = s.uploads_path / "orphan.mp4", s.uploads_path / ("a" * 32 + ".part")
    for f in (old_upload, fresh_upload, busy_upload, orphan, unfinished):
        f.write_bytes(b"video")
    long_ago = time.time() - 20 * 86400
    os.utime(orphan, (long_ago, long_ago))
    os.utime(unfinished, (long_ago, long_ago))
    with session_scope() as session:
        stale = datetime.now(UTC) - timedelta(days=10)
        session.add(Job(source_path=str(old_upload.resolve()), updated_at=stale))
        session.add(Job(source_path=str(fresh_upload.resolve()), updated_at=datetime.now(UTC)))
        session.add(Job(source_path=str(busy_upload.resolve()), updated_at=stale, task="analyze"))
        session.flush()
        _export(session, 1, s.exports_path / "old_cut.mp4", age_days=45)
        _export(session, 1, s.exports_path / "new_cut.mp4", age_days=2)

    assert janitor.sweep() == {"uploads": 2, "exports": 1}
    assert not old_upload.exists() and not orphan.exists()
    assert fresh_upload.exists() and busy_upload.exists(), "in use, or a job is working on it"
    assert unfinished.exists(), "unfinished uploads have their own clean-up"
    assert not (s.exports_path / "old_cut.mp4").exists() and not (s.exports_path / "old_cut.caption.txt").exists()
    assert (s.exports_path / "new_cut.mp4").exists()
    assert janitor.sweep() == {"uploads": 0, "exports": 0}


def test_nothing_is_removed_unless_a_shelf_life_is_set(settings):
    from possession_cut.db import session_scope
    from possession_cut.worker import janitor

    upload = settings.uploads_path / "keep.mp4"
    upload.write_bytes(b"video")
    long_ago = time.time() - 400 * 86400
    os.utime(upload, (long_ago, long_ago))
    with session_scope() as session:
        _export(session, 1, settings.exports_path / "ancient.mp4", age_days=400)
    assert janitor.sweep() == {"uploads": 0, "exports": 0}
    assert upload.exists() and (settings.exports_path / "ancient.mp4").exists()


def test_an_export_makes_room_by_removing_the_oldest_first(settings, monkeypatch):
    from possession_cut.db import session_scope
    from possession_cut.worker import janitor

    with session_scope() as session:
        _export(session, 1, settings.exports_path / "oldest.mp4", age_days=9)
        _export(session, 1, settings.exports_path / "older.mp4", age_days=5)
        _export(session, 1, settings.exports_path / "just_made.mp4", age_days=0)

    usage = namedtuple("usage", "total used free")

    def disk(_path):
        left = sum(1 for n in ("oldest.mp4", "older.mp4") if (settings.exports_path / n).exists())
        return usage(10_000, 0, 1_000 + (2 - left) * 1_000)  # each removal frees 1,000

    monkeypatch.setattr(janitor.shutil, "disk_usage", disk)
    assert janitor.make_room(500) == 0, "already room"
    assert janitor.make_room(1_500) == 1
    assert not (settings.exports_path / "oldest.mp4").exists() and (settings.exports_path / "older.mp4").exists()
    assert janitor.make_room(99_999) == 1, "takes what it may, never the export made in the last hour"
    assert (settings.exports_path / "just_made.mp4").exists()


def test_a_removed_export_says_so(hosted, settings):
    from possession_cut.db import Job, session_scope

    hosted.post("/api/login", json={"password": PASSWORD})
    with session_scope() as session:
        job = Job(source_path="gone.mp4")
        session.add(job)
        session.flush()
        export_id = _export(session, job.id, settings.exports_path / "cut.mp4", age_days=1)
        job_id = job.id
    listed = hosted.get(f"/api/jobs/{job_id}/exports").json()[0]
    assert listed["url"] and listed["file_removed"] is False
    (settings.exports_path / "cut.mp4").unlink()
    listed = hosted.get(f"/api/jobs/{job_id}/exports").json()[0]
    assert listed["url"] is None and listed["file_removed"] is True
    assert hosted.get(f"/api/exports/{export_id}/file").status_code == 404
