"""Read a calibrated score bug: is it on screen, and what does each field say.

Used three ways with the same code path: calibration validation, the calibration
screen's live preview, and the analysis sampler (two reads per second of video).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..sports.base import FieldSpec
from .geometry import Box, to_px
from .ocr import CHARSETS, OcrEngine, field_signature, prepare_field, same_signature

VISIBLE_THRESHOLD = 0.72
N_SLICES = 6


@dataclass
class FieldRead:
    text: str = ""
    conf: float = 0.0
    cached: bool = False


@dataclass
class BugRead:
    visible: bool
    similarity: float
    fields: dict[str, FieldRead] = field(default_factory=dict)


@dataclass
class ReaderState:
    """Per-stream memory: the last crop signature and read of each field."""

    signatures: dict[str, np.ndarray] = field(default_factory=dict)
    reads: dict[str, FieldRead] = field(default_factory=dict)
    ocr_calls: int = 0
    cache_hits: int = 0


def build_reference(crops: list[np.ndarray], field_boxes: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Reference image and static mask for a bug from several crops of it.

    The reference is the per-pixel median. The mask keeps pixels that stay put across
    the crops and sit outside the changing fields (scores, clocks).
    """
    stack = np.stack([c.astype(np.float32) for c in crops])
    ref = np.median(stack, axis=0).astype(np.uint8)
    if len(crops) >= 3:
        deviation = np.abs(stack - ref[None].astype(np.float32)).max(axis=3)  # N,H,W
        # tolerate one outlier frame (a score animation, a transition)
        dev = np.sort(deviation, axis=0)[max(0, len(crops) - 2)]
        mask = (dev < 28).astype(np.uint8) * 255
    else:
        mask = np.full(ref.shape[:2], 255, dtype=np.uint8)
    for x0, y0, x1, y1 in field_boxes:
        mask[max(0, y0) : y1, max(0, x0) : x1] = 0
    if mask.mean() < 255 * 0.08:
        # nearly everything looked dynamic; fall back to "everything outside the fields"
        mask[:] = 255
        for x0, y0, x1, y1 in field_boxes:
            mask[max(0, y0) : y1, max(0, x0) : x1] = 0
    return ref, mask


class BugReader:
    def __init__(
        self,
        bug: Box,
        fields: dict[str, Box],
        specs: tuple[FieldSpec, ...],
        frame_w: int,
        frame_h: int,
        engine: OcrEngine | None = None,
        reference: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        self.engine = engine
        self.specs = {s.name: s for s in specs}
        x0, y0, x1, y1 = to_px(bug, frame_w, frame_h)
        self.roi = (x0, y0, x1 - x0, y1 - y0)  # x, y, w, h in display pixels
        self.field_px: dict[str, tuple[int, int, int, int]] = {}
        for name, box in fields.items():
            if box is None or name not in self.specs:
                continue
            fx0, fy0, fx1, fy1 = to_px(tuple(box), frame_w, frame_h)
            # relative to the ROI, clipped to it
            rx0, ry0 = max(0, fx0 - x0), max(0, fy0 - y0)
            rx1, ry1 = min(self.roi[2], fx1 - x0), min(self.roi[3], fy1 - y0)
            if rx1 - rx0 >= 4 and ry1 - ry0 >= 4:
                self.field_px[name] = (rx0, ry0, rx1, ry1)
        self._ref_gray: np.ndarray | None = None
        self._slices: list[tuple[np.ndarray, np.ndarray, float]] = []
        if reference is not None:
            self.set_reference(reference, mask)

    # -- visibility --------------------------------------------------------
    def set_reference(self, reference: np.ndarray, mask: np.ndarray | None) -> None:
        w, h = self.roi[2], self.roi[3]
        ref = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if mask is None:
            m = np.ones((h, w), dtype=bool)
        else:
            m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        self._ref_gray = gray
        self._slices = []
        edges = np.linspace(0, w, N_SLICES + 1).astype(int)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            sel = np.zeros((h, w), dtype=bool)
            sel[:, a:b] = m[:, a:b]
            if sel.sum() < 40:
                continue
            ref_vals = gray[sel]
            centered = ref_vals - ref_vals.mean()
            self._slices.append((sel, centered, float(np.sqrt((centered**2).sum()))))

    def similarity(self, crop: np.ndarray) -> float:
        """0..1: how much this crop looks like the calibrated bug."""
        if self._ref_gray is None or not self._slices:
            return 1.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gray.shape != self._ref_gray.shape:
            gray = cv2.resize(gray, (self._ref_gray.shape[1], self._ref_gray.shape[0]))
        scores = []
        for sel, ref_centered, ref_norm in self._slices:
            vals = gray[sel]
            if ref_norm < 3.0 * np.sqrt(len(vals)):
                # flat panel: no texture to correlate, compare brightness instead
                mad = float(np.abs(vals - (self._ref_gray[sel])).mean())
                scores.append(max(0.0, 1.0 - mad / 40.0))
                continue
            centered = vals - vals.mean()
            norm = float(np.sqrt((centered**2).sum()))
            if norm < 1e-3:
                scores.append(0.0)
                continue
            ncc = float((centered * ref_centered).sum() / (norm * ref_norm))
            mad = float(np.abs(vals - self._ref_gray[sel]).mean())
            scores.append(max(0.0, ncc) * max(0.0, 1.0 - mad / 120.0))
        # A score animation can cover up to half the bug; judge by the better half.
        scores.sort(reverse=True)
        top = scores[: max(1, (len(scores) + 1) // 2)]
        return float(np.mean(top))

    # -- reading -------------------------------------------------------------
    def field_crop(self, crop: np.ndarray, name: str) -> np.ndarray:
        x0, y0, x1, y1 = self.field_px[name]
        return crop[y0:y1, x0:x1]

    def read(self, crop: np.ndarray, state: ReaderState | None = None, check_visible: bool = True) -> BugRead:
        """Read every field of one bug crop. ``state`` enables skip-if-unchanged caching."""
        sim = self.similarity(crop) if check_visible else 1.0
        if check_visible and sim < VISIBLE_THRESHOLD:
            if state is not None:
                state.signatures.clear()
                state.reads.clear()
            return BugRead(visible=False, similarity=sim)
        assert self.engine is not None, "an OCR engine is required to read fields"
        out: dict[str, FieldRead] = {}
        todo_names: list[str] = []
        todo_imgs: list[np.ndarray] = []
        todo_sigs: list[np.ndarray] = []
        for name in self.field_px:
            sub = self.field_crop(crop, name)
            sig = field_signature(sub)
            if state is not None and same_signature(state.signatures.get(name), sig):
                prev = state.reads[name]
                out[name] = FieldRead(prev.text, prev.conf, cached=True)
                state.cache_hits += 1
                continue
            todo_names.append(name)
            todo_imgs.append(prepare_field(sub))
            todo_sigs.append(sig)
        if todo_imgs:
            charsets = [CHARSETS[self.specs[n].charset] for n in todo_names]
            results = self.engine.recognize(todo_imgs, charsets)
            for name, sig, (text, conf) in zip(todo_names, todo_sigs, results, strict=True):
                read = FieldRead(text.strip(), conf)
                out[name] = read
                if state is not None:
                    state.signatures[name] = sig
                    state.reads[name] = read
                    state.ocr_calls += 1
        return BugRead(visible=True, similarity=sim, fields=out)
