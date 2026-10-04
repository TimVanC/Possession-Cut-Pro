"""Frame access through ffmpeg: single frames for calibration, ROI streams for OCR."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..config import ffmpeg_bin, get_settings
from .probe import Probe


class FrameError(RuntimeError):
    pass


def square_pixel_filter(probe: Probe) -> str | None:
    """Filter that makes pixels square, so normalized boxes mean the same thing everywhere."""
    if probe.anamorphic:
        return f"scale={probe.display_width}:{probe.height},setsar=1"
    return None


def _hwaccel_args() -> list[str]:
    mode = (get_settings().hwaccel or "none").lower()
    return ["-hwaccel", "auto"] if mode == "auto" else []


def extract_frame(probe: Probe, t: float, max_height: int | None = None) -> np.ndarray:
    """One BGR frame at time ``t`` (seconds), at display resolution."""
    t = min(max(0.0, t), max(0.0, probe.duration - 0.05))
    w, h = probe.display_width, probe.height
    filters = [f for f in [square_pixel_filter(probe)] if f]
    if max_height and h > max_height:
        w, h = int(round(w * max_height / h / 2) * 2), max_height
        filters.append(f"scale={w}:{h}")
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{t:.3f}", "-i", probe.path, "-frames:v", "1"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) < w * h * 3:
        raise FrameError(
            f"Could not read a frame at {t:.2f}s: {proc.stderr.decode('utf-8', 'replace').strip()[:300]}"
        )
    return np.frombuffer(proc.stdout[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()


def extract_frames(probe: Probe, times: list[float], max_height: int | None = None) -> list[np.ndarray | None]:
    """Several frames in parallel. A frame that fails to decode comes back as None."""

    def one(t: float) -> np.ndarray | None:
        try:
            return extract_frame(probe, t, max_height)
        except FrameError:
            return None

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(times)))) as pool:
        return list(pool.map(one, times))


def save_jpeg(frame: np.ndarray, path: Path, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise FrameError(f"Could not encode {path}")
    path.write_bytes(buf.tobytes())


def stream_roi(
    probe: Probe,
    roi: tuple[int, int, int, int],
    fps: float,
    start: float = 0.0,
    duration: float | None = None,
    threads: int | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Decode ``[start, start+duration)`` and yield (time, BGR crop) at ``fps`` samples per second.

    ``roi`` is (x, y, w, h) in display pixels. Only the crop is converted to BGR, which is
    what keeps a 2.5 hour broadcast inside the analysis budget.
    """
    x, y, w, h = roi
    # ffmpeg crops chroma-subsampled video on even pixels only: an odd offset or size is
    # silently rounded, and the raw frames coming out are then not the size we asked for.
    # So crop an even-aligned box around the ROI and take the exact ROI out of it here.
    ex, ey = x - x % 2, y - y % 2
    ew = min(probe.display_width - ex, (x + w - ex) + (x + w - ex) % 2)
    eh = min(probe.height - ey, (y + h - ey) + (y + h - ey) % 2)
    ew, eh = ew - ew % 2, eh - eh % 2
    ox, oy = x - ex, y - ey
    w, h = min(w, ew - ox), min(h, eh - oy)
    filters = [f for f in [square_pixel_filter(probe)] if f]
    filters.append(f"crop={ew}:{eh}:{ex}:{ey}")
    filters.append(f"fps={fps}:start_time=0:round=up")
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", *_hwaccel_args()]
    if threads:
        cmd += ["-threads", str(threads)]
    cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", probe.path, "-an", "-sn", "-dn", "-vf", ",".join(filters),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = ew * eh * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size * 4)
    assert proc.stdout is not None
    expected = None if duration is None else int(round(duration * fps))
    try:
        k = 0
        while expected is None or k < expected:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(eh, ew, 3)
            yield start + k / fps, frame[oy : oy + h, ox : ox + w]
            k += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        proc.wait()
    if k == 0 and err.strip():
        raise FrameError(f"ffmpeg produced no samples from {start:.1f}s: {err.strip()[:300]}")


def default_decode_threads(workers: int) -> int:
    return max(2, (os.cpu_count() or 4) // max(1, workers))
