"""OCR for score bug fields.

Uses the PP-OCRv4 models bundled with rapidocr-onnxruntime (ONNX, CPU). Field reads
skip text detection (the boxes are known from calibration) and decode with a
restricted character set: the recognizer's per-step probabilities are masked to the
characters a field can contain before CTC decoding, so a score can never come back
as letters.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DIGITS = "0123456789"
CHARSETS = {
    "score": DIGITS,
    "clock": DIGITS + ":.",
    "shot_clock": DIGITS + ".",
    "period": DIGITS + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
    "count": DIGITS + "-",
    "text": None,  # unrestricted
}

REC_HEIGHT = 48


@dataclass
class TextBox:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the image passed in
    text: str
    conf: float


def _models_dir() -> Path:
    import rapidocr_onnxruntime

    return Path(rapidocr_onnxruntime.__file__).parent / "models"


class OcrEngine:
    """One recognizer session (and, lazily, one full detector) per instance. Thread-safe to call."""

    def __init__(self, threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        model = _models_dir() / "ch_PP-OCRv4_rec_infer.onnx"
        self._rec = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input = self._rec.get_inputs()[0].name
        meta = self._rec.get_modelmeta().custom_metadata_map
        chars = meta["character"].splitlines()
        # same layout the bundled CTC decoder uses: blank first, space last
        self.chars: list[str] = ["blank", *chars, " "]
        self._index = {c: i for i, c in enumerate(self.chars)}
        self._allowed_cache: dict[str, np.ndarray] = {}
        self._full = None
        self._strip = None
        self._full_lock = threading.Lock()
        self._threads = threads
        # Measured on a laptop CPU: single-threaded sessions are fastest one crop at a
        # time; batching only pays off when the session has several threads.
        self._batch = 8 if threads >= 4 else 1

    # -- restricted recognition ------------------------------------------
    def _allowed(self, charset: str) -> np.ndarray:
        cached = self._allowed_cache.get(charset)
        if cached is None:
            idx = sorted({0} | {self._index[c] for c in charset if c in self._index})
            cached = np.array(idx, dtype=np.int64)
            self._allowed_cache[charset] = cached
        return cached

    @staticmethod
    def _to_input(img: np.ndarray, width: int) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        new_w = min(width, max(1, int(np.ceil(REC_HEIGHT * w / h))))
        resized = cv2.resize(img, (new_w, REC_HEIGHT), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        resized = (resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        out = np.zeros((3, REC_HEIGHT, width), dtype=np.float32)
        out[:, :, :new_w] = resized
        return out

    def recognize(self, images: list[np.ndarray], charsets: list[str | None]) -> list[tuple[str, float]]:
        """Recognize pre-cropped text lines. Returns (text, confidence 0-1) per image."""
        results: list[tuple[str, float]] = [("", 0.0)] * len(images)
        if not images:
            return results
        ratios = [img.shape[1] / img.shape[0] for img in images]
        order = np.argsort(ratios)
        for b0 in range(0, len(order), self._batch):
            batch = order[b0 : b0 + self._batch]
            width = int(np.ceil(REC_HEIGHT * max(max(ratios[i] for i in batch), 1.0) / 8) * 8)
            width = max(48, min(width, 640))
            tensor = np.stack([self._to_input(images[i], width) for i in batch])
            probs = self._rec.run(None, {self._input: tensor})[0]
            for row, i in enumerate(batch):
                results[i] = self._decode(probs[row], charsets[i])
        return results

    def _decode(self, probs: np.ndarray, charset: str | None) -> tuple[str, float]:
        if charset is None:
            idx = probs.argmax(axis=1)
            conf = probs.max(axis=1)
        else:
            allowed = self._allowed(charset)
            sub = probs[:, allowed]
            k = sub.argmax(axis=1)
            idx = allowed[k]
            # confidence stays the raw probability: if the model wanted a character
            # outside the set, the read is flagged as weak instead of looking certain
            conf = sub[np.arange(len(k)), k]
        keep = idx != 0
        keep[1:] &= idx[1:] != idx[:-1]
        if not keep.any():
            return "", 0.0
        text = "".join(self.chars[i] for i in idx[keep])
        return text, float(conf[keep].min())

    # -- full detection + recognition (calibration only) ------------------
    def read_text(self, image: np.ndarray, min_conf: float = 0.3, strip: bool = False) -> list[TextBox]:
        """Find and read every text line in ``image``.

        ``strip=True`` is for wide, short crops (a zoomed score bug): the detector keeps the
        image at the size given instead of blowing its short side up to 736 px.
        """
        from rapidocr_onnxruntime import RapidOCR

        with self._full_lock:
            if strip:
                if self._strip is None:
                    self._strip = RapidOCR(
                        intra_op_num_threads=max(1, self._threads),
                        det_limit_type="max", det_limit_side_len=2400, width_height_ratio=-1,
                    )
                ocr = self._strip
            else:
                if self._full is None:
                    self._full = RapidOCR(intra_op_num_threads=max(1, self._threads))
                ocr = self._full
            if strip:
                result, _ = ocr(image, use_cls=False, text_score=min_conf, unclip_ratio=1.3, box_thresh=0.5)
            else:
                result, _ = ocr(image, use_cls=False, text_score=min_conf)
        boxes = []
        for quad, text, conf in result or []:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            boxes.append(TextBox((min(xs), min(ys), max(xs), max(ys)), str(text).strip(), float(conf)))
        return boxes


_engines: dict[int, OcrEngine] = {}
_engines_lock = threading.Lock()


def get_engine(threads: int = 1) -> OcrEngine:
    """Process-wide engine cache keyed by thread count."""
    with _engines_lock:
        engine = _engines.get(threads)
        if engine is None:
            engine = _engines[threads] = OcrEngine(threads)
        return engine


# -- image preparation ---------------------------------------------------------


def drop_rules(binary: np.ndarray) -> np.ndarray:
    """Blank out line-like ink: a bug's border, a row of timeout dashes, an underline.

    A box that reaches a hair past its text picks these up, and the recognizer then sees
    a long bar with small digits beside it and reads nothing. Lines are far wider than
    tall, or hug a side of the box as a thin vertical; glyphs are neither (a colon's dots
    are square, a "1" is a good deal wider than a hairline).
    """
    ink = (binary < 128).astype(np.uint8)
    if not ink.any():
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    height, width = binary.shape[:2]
    out = binary.copy()
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i][:4])
        flat = w >= 3 * h and h <= 0.25 * height
        side_bar = (x == 0 or x + w == width) and w <= 0.12 * h
        if flat or side_bar:
            out[labels == i] = 255
    return out


def prepare_field(crop: np.ndarray, scale: int = 3, binarize: bool = True) -> np.ndarray:
    """Upscale 3x, normalize polarity to dark text on a light ground, binarize, pad.

    Returns a BGR image ready for the recognizer.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # The border of a field is background. If it is darker than the middle, text is light: invert.
    b = max(2, gray.shape[0] // 10)
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    bg = float(np.median(border))
    if bg < 128:
        gray = 255 - gray
    if binarize:
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # a blank field has no real foreground; Otsu would split noise. Detect via contrast.
        if float(gray.max()) - float(gray.min()) < 40:
            out = np.full_like(gray, 255)
        out = drop_rules(out)
        out = cv2.GaussianBlur(out, (3, 3), 0)
    else:
        lo, hi = np.percentile(gray, (2, 98))
        out = np.clip((gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)
        if hi - lo < 40:
            out = np.full_like(gray, 255)
    # Trim to the ink so a short value in a wide box (a "7" in a three-digit score field)
    # fills the recognizer's input the same way a full one does.
    ink = out < 140
    cols = np.flatnonzero(ink.any(axis=0))
    rows = np.flatnonzero(ink.any(axis=1))
    if len(cols) and len(rows):
        out = out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    pad = max(6, out.shape[0] // 3)
    out = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def field_signature(crop: np.ndarray) -> np.ndarray:
    """Small binary thumbnail used to tell whether a field changed since the last sample."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    small = cv2.resize(gray, (48, 16), interpolation=cv2.INTER_AREA)
    return small


def same_signature(a: np.ndarray | None, b: np.ndarray, tol: float = 6.0) -> bool:
    if a is None:
        return False
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) < tol and (
        float(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) < 60
    )


# -- parsing ---------------------------------------------------------------------

_CLOCK_MMSS = re.compile(r"^(\d{1,2}):(\d{2})$")
_CLOCK_TENTHS = re.compile(r"^(\d{1,2})\.(\d)$")


def parse_score(text: str, max_value: int = 250) -> int | None:
    text = text.strip()
    if not text.isdigit() or len(text) > 3:
        return None
    value = int(text)
    return value if value <= max_value else None


def parse_clock(text: str, max_minutes: int = 20) -> float | None:
    """Seconds remaining from 'M:SS', 'MM:SS' or 'SS.t'. None when it is not a clock."""
    text = text.strip().replace(" ", "")
    m = _CLOCK_MMSS.match(text)
    if m:
        minutes, seconds = int(m.group(1)), int(m.group(2))
        if seconds < 60 and minutes <= max_minutes:
            return float(minutes * 60 + seconds)
        return None
    m = _CLOCK_TENTHS.match(text)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 10.0
    # OCR sometimes drops the colon: 1022 -> 10:22, 122 -> 1:22
    if text.isdigit() and len(text) in (3, 4):
        minutes, seconds = int(text[:-2]), int(text[-2:])
        if seconds < 60 and minutes <= max_minutes:
            return float(minutes * 60 + seconds)
    return None


def parse_shot_clock(text: str, max_value: int = 40) -> float | None:
    text = text.strip()
    m = _CLOCK_TENTHS.match(text)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 10.0
    if text.isdigit() and len(text) <= 2 and int(text) <= max_value:
        return float(int(text))
    return None


_PERIOD_WORDS = {
    "1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4,
    "1": 1, "2": 2, "3": 3, "4": 4,
    "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4,
    "1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4,
    "OT": 5, "OT1": 5, "1OT": 5, "2OT": 6, "OT2": 6, "3OT": 7, "OT3": 7,
    "1STQTR": 1, "2NDQTR": 2, "3RDQTR": 3, "4THQTR": 4,
    "P1": 1, "P2": 2, "P3": 3,
}
# common confusions in small bold type
_PERIOD_FIXES = str.maketrans({"I": "1", "L": "1", "S": "5", "O": "0", "Z": "2", "B": "8"})


def parse_period(text: str) -> int | None:
    """Period number from '3rd', 'Q4', 'OT'... None for 'HALF', 'FINAL' and unreadable text."""
    key = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not key:
        return None
    if key in _PERIOD_WORDS:
        return _PERIOD_WORDS[key]
    # leading digit plus a mangled ordinal suffix: '3RO', '15T', '4TN'
    if key[0] in "1234" and len(key) <= 3 and not key[1:].isdigit():
        return int(key[0])
    if key[0] in "ILSZ" and len(key) == 3:
        fixed = key[0].translate(_PERIOD_FIXES)
        if fixed in "1234":
            return int(fixed)
    return None
