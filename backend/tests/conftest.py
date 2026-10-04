"""Shared fixtures. The synthetic broadcast is rendered once and cached on disk."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _cache_dir() -> Path:
    # Outside the repo on purpose: rendered videos are big and the repo may live in a synced folder.
    root = Path(os.environ.get("POSSESSION_CUT_TEST_CACHE", Path(tempfile.gettempdir()) / "possession-cut-tests"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _synth_version() -> str:
    h = hashlib.sha256()
    for name in ("script.py", "render.py"):
        h.update((BACKEND / "possession_cut" / "synth" / name).read_bytes())
    return h.hexdigest()[:12]


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    return _cache_dir()


@pytest.fixture(scope="session")
def coverage_script():
    from possession_cut.synth.script import coverage_game

    return coverage_game()


@pytest.fixture(scope="session")
def coverage_video(coverage_script) -> tuple[Path, dict]:
    """(video path, ground truth) for the short fixed game. Rendered once per generator version."""
    from possession_cut.synth.render import render_video

    path = _cache_dir() / f"coverage_{_synth_version()}.mp4"
    truth_path = Path(str(path) + ".truth.json")
    if not (path.exists() and truth_path.exists()):
        for stale in _cache_dir().glob("coverage_*"):
            stale.unlink(missing_ok=True)
        render_video(coverage_script, path)
    return path, json.loads(truth_path.read_text(encoding="utf-8"))


def _calibration_version() -> str:
    h = hashlib.sha256(_synth_version().encode())
    for name in ("calibration.py", "calib_local.py", "bugreader.py", "ocr.py", "geometry.py", "frames.py"):
        h.update((BACKEND / "possession_cut" / "pipeline" / name).read_bytes())
    return h.hexdigest()[:12]


@pytest.fixture(scope="session")
def coverage_probe(coverage_video):
    from possession_cut.pipeline.probe import probe_file

    return probe_file(coverage_video[0])


@pytest.fixture(scope="session")
def ocr_engine():
    from possession_cut.pipeline.ocr import get_engine

    return get_engine(4)


@pytest.fixture(scope="session")
def coverage_calibration(coverage_video, coverage_probe, ocr_engine):
    """Local (no Claude) calibration of the coverage video: (calibration, reference, mask, job_dir).

    Cached on disk per version of the generator and of the calibration code.
    """
    import cv2

    from possession_cut.pipeline.calibration import Calibration, calibrate
    from possession_cut.sports import get_adapter

    job_dir = _cache_dir() / f"calib_{_calibration_version()}"
    cal_path = job_dir / "calibration.json"
    if cal_path.exists():
        cal = Calibration.load(cal_path)
        return cal, cv2.imread(str(job_dir / "ref.png")), cv2.imread(str(job_dir / "mask.png"), cv2.IMREAD_GRAYSCALE), job_dir
    for stale in _cache_dir().glob("calib_*"):
        import shutil

        shutil.rmtree(stale, ignore_errors=True)
    job_dir.mkdir(parents=True)
    cal, ref, mask = calibrate(coverage_probe, job_dir, get_adapter("nba"), ocr_engine, claude=None)
    cv2.imwrite(str(job_dir / "ref.png"), ref)
    cv2.imwrite(str(job_dir / "mask.png"), mask)
    cal.save(cal_path)
    return cal, ref, mask, job_dir


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    """Isolated Settings pointing every folder at a temp dir, with no API key."""
    from possession_cut import config, db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setenv("EXPORTS_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")
    config.get_settings.cache_clear()
    db.reset_engine()
    s = config.get_settings()
    s.ensure_dirs()
    yield s
    db.reset_engine()
    config.get_settings.cache_clear()
