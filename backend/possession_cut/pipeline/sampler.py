"""Stage 3: read the score bug across the whole file.

The file is split into chunks that are decoded in parallel (one ffmpeg per chunk, input
seeking makes the start of each chunk cheap). Only the bug region is converted out of
YUV, and a field is re-read only when its pixels changed, which is what brings a 2.5 hour
broadcast in under the 15 minute budget on a laptop CPU.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ..sports.base import SportAdapter
from .bugreader import BugReader, ReaderState
from .calibration import Calibration
from .frames import default_decode_threads, stream_roi
from .ocr import OcrEngine
from .probe import Probe

CHUNK_SECONDS = 150.0


@dataclass
class RawSamples:
    """What OCR saw, before any cleaning. One entry per sample."""

    fps: float
    t: np.ndarray  # float64 seconds
    visible: np.ndarray  # bool
    similarity: np.ndarray  # float32
    texts: dict[str, list[str]] = field(default_factory=dict)
    confs: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    def to_frame(self) -> pl.DataFrame:
        cols: dict[str, object] = {
            "t_video": self.t,
            "bug_visible": self.visible,
            "similarity": self.similarity.astype(np.float32),
        }
        for name in self.texts:
            cols[f"{name}_text"] = self.texts[name]
            cols[f"{name}_conf"] = self.confs[name].astype(np.float32)
        return pl.DataFrame(cols)

    def save(self, path: Path) -> None:
        self.to_frame().write_parquet(path)

    @classmethod
    def load(cls, path: Path, fps: float) -> RawSamples:
        df = pl.read_parquet(path)
        names = [c[: -len("_text")] for c in df.columns if c.endswith("_text")]
        return cls(
            fps=fps,
            t=df["t_video"].to_numpy().astype(np.float64),
            visible=df["bug_visible"].to_numpy().astype(bool),
            similarity=df["similarity"].to_numpy().astype(np.float32),
            texts={n: df[f"{n}_text"].to_list() for n in names},
            confs={n: df[f"{n}_conf"].to_numpy().astype(np.float32) for n in names},
        )


def chunk_plan(start: float, end: float, fps: float, chunk_seconds: float = CHUNK_SECONDS) -> list[tuple[float, float]]:
    """(start, duration) chunks on the sample grid covering [start, end)."""
    step = 1.0 / fps
    first = int(np.ceil(start * fps - 1e-9))
    last = int(np.floor(end * fps - 1e-9))  # last sample index (inclusive)
    per_chunk = max(1, int(round(chunk_seconds * fps)))
    out = []
    k = first
    while k <= last:
        n = min(per_chunk, last - k + 1)
        out.append((k * step, n * step))
        k += n
    return out


def sample_bug(
    probe: Probe,
    cal: Calibration,
    adapter: SportAdapter,
    reference: np.ndarray | None,
    mask: np.ndarray | None,
    fps: float = 2.0,
    workers: int = 4,
    start: float = 0.0,
    end: float | None = None,
    progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RawSamples:
    end = probe.duration if end is None else min(end, probe.duration)
    chunks = chunk_plan(start, end, fps)
    total = sum(int(round(d * fps)) for _, d in chunks)
    names = [s.name for s in adapter.bug_fields if s.name in cal.fields]
    fw, fh = probe.display_width, probe.height

    local = threading.local()
    done = 0
    lock = threading.Lock()
    t_start = time.time()
    totals = {"ocr_calls": 0, "cache_hits": 0}

    def reader() -> BugReader:
        r = getattr(local, "reader", None)
        if r is None:
            r = BugReader(
                tuple(cal.bug), {k: tuple(v) for k, v in cal.fields.items()}, adapter.bug_fields,
                fw, fh, OcrEngine(threads=1), reference, mask,
            )
            local.reader = r
        return r

    def run_chunk(chunk: tuple[float, float]):
        nonlocal done
        c_start, c_dur = chunk
        r = reader()
        state = ReaderState()
        rows = []
        for t, crop in stream_roi(probe, r.roi, fps, c_start, c_dur, threads=default_decode_threads(workers)):
            if should_stop and should_stop():
                break
            read = r.read(crop, state)
            rows.append((t, read))
            with lock:
                done += 1
                if progress and done % 50 == 0:
                    rate = done / max(1e-6, time.time() - t_start)
                    progress(done / max(1, total), f"Reading the score bug ({done}/{total} samples, {rate / fps:.0f}x real time)")
        with lock:
            totals["ocr_calls"] += state.ocr_calls
            totals["cache_hits"] += state.cache_hits
        return rows

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(run_chunk, chunks))

    rows = [row for chunk_rows in results for row in chunk_rows]
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    raw = RawSamples(
        fps=fps,
        t=np.array([r[0] for r in rows], dtype=np.float64),
        visible=np.array([r[1].visible for r in rows], dtype=bool),
        similarity=np.array([r[1].similarity for r in rows], dtype=np.float32),
        texts={name: [""] * n for name in names},
        confs={name: np.zeros(n, dtype=np.float32) for name in names},
    )
    for i, (_, read) in enumerate(rows):
        for name, fr in read.fields.items():
            if name in raw.texts:
                raw.texts[name][i] = fr.text
                raw.confs[name][i] = fr.conf
    elapsed = time.time() - t_start
    raw.stats = {
        "samples": n,
        "seconds": round(elapsed, 2),
        "speed_x_realtime": round((end - start) / max(elapsed, 1e-6), 1),
        "ocr_calls": totals["ocr_calls"],
        "cache_hits": totals["cache_hits"],
        "workers": workers,
    }
    if progress:
        progress(1.0, f"Read {n} samples in {elapsed:.0f}s")
    return raw
