"""Render a ``GameScript`` to an MP4 that looks like a broadcast with an ESPN-style bug.

Every frame also carries a small barcode with its own frame index, top-centre above
the action. Tests decode it from exported frames to prove, frame by frame, which
source moments made it into a cut. The pipeline never reads the barcode.
"""

from __future__ import annotations

import json
import math
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import ffmpeg_bin
from ..fonts import load_font
from .script import AWAY, HOME, BugState, GameScript

BARCODE_BITS = 20
BARCODE_CELLS = BARCODE_BITS + 4 + 2  # data + checksum + two guard cells


# -- geometry -----------------------------------------------------------------


def bug_layout(width: int, height: int, offset_frac: float = 0.02) -> dict:
    """Pixel rectangles (x0, y0, x1, y1) for the bug, its fields, the barcode and the crop."""
    bug_w = int(round(width * 0.519 / 2) * 2)
    bug_h = int(round(height * 0.064 / 2) * 2)
    cx = width / 2 + offset_frac * width
    x0 = int(round(cx - bug_w / 2))
    y1 = height - int(round(height * 0.047))
    y0 = y1 - bug_h
    inset = max(2, int(round(bug_h * 0.07)))

    def cell(a: float, b: float) -> list[int]:
        return [x0 + int(round(a * bug_w)), y0 + inset, x0 + int(round(b * bug_w)), y1 - inset]

    def block(a: float, b: float) -> list[int]:
        return [x0 + int(round(a * bug_w)), y0, x0 + int(round(b * bug_w)), y1]

    fields = {
        "away_label": cell(0.078, 0.200),
        "away_score": cell(0.215, 0.350),
        "home_label": cell(0.378, 0.500),
        "home_score": cell(0.515, 0.650),
        "period": cell(0.735, 0.822),
        "clock": cell(0.826, 0.928),
        "shot_clock": cell(0.938, 0.990),
    }
    blocks = {
        AWAY: block(0.0, 0.36),
        HOME: block(0.36, 0.72),
        "info": block(0.72, 1.0),
        "away_logo": block(0.0, 0.07),
        "home_logo": block(0.655, 0.72),
        "shot_box": block(0.932, 0.995),
    }

    # PRD crop rule: full height, width = bug width / 0.78, centred on the bug, clamped.
    crop_w = min(width, int(round(bug_w / 0.78 / 2) * 2))
    crop_x = int(round((x0 + bug_w / 2) - crop_w / 2))
    crop_x = max(0, min(width - crop_w, crop_x))

    cell_px = max(8, int(round(width / 80)))
    bar_x0 = int(round(x0 + bug_w / 2 - BARCODE_CELLS * cell_px / 2))
    bar_y0 = int(round(height * 0.035))
    return {
        "width": width,
        "height": height,
        "bug": [x0, y0, x0 + bug_w, y1],
        "fields": fields,
        "blocks": blocks,
        "crop": [crop_x, 0, crop_w, height],
        "barcode": {"x0": bar_x0, "y0": bar_y0, "cell": cell_px, "cells": BARCODE_CELLS},
    }


def normalized(box: list[int], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box
    return [round(x0 / width, 5), round(y0 / height, 5), round(x1 / width, 5), round(y1 / height, 5)]


def layout_truth(layout: dict) -> dict:
    w, h = layout["width"], layout["height"]
    cx, cy, cw, ch = layout["crop"]
    return {
        "bug_px": layout["bug"],
        "bug": normalized(layout["bug"], w, h),
        "fields_px": layout["fields"],
        "fields": {k: normalized(v, w, h) for k, v in layout["fields"].items()},
        "crop_px": layout["crop"],
        "crop": [round(cx / w, 5), round(cy / h, 5), round(cw / w, 5), round(ch / h, 5)],
        "barcode": layout["barcode"],
    }


# -- barcode --------------------------------------------------------------------


def _barcode_bits(n: int) -> list[int]:
    data = [(n >> (BARCODE_BITS - 1 - i)) & 1 for i in range(BARCODE_BITS)]
    check = sum((n >> (4 * i)) & 0xF for i in range(BARCODE_BITS // 4)) & 0xF
    chk = [(check >> (3 - i)) & 1 for i in range(4)]
    return [1] + data + chk + [0]


def draw_barcode(frame: np.ndarray, n: int, layout: dict) -> None:
    bc = layout["barcode"]
    c, x0, y0 = bc["cell"], bc["x0"], bc["y0"]
    frame[y0 - c // 2 : y0 + c + c // 2, x0 - c : x0 + (BARCODE_CELLS + 1) * c] = 128
    for i, bit in enumerate(_barcode_bits(n)):
        frame[y0 : y0 + c, x0 + i * c : x0 + (i + 1) * c] = 255 if bit else 0


def barcode_centers(layout: dict) -> list[tuple[float, float]]:
    """Centre of each barcode cell in source pixels."""
    bc = layout["barcode"]
    c = bc["cell"]
    return [(bc["x0"] + (i + 0.5) * c, bc["y0"] + 0.5 * c) for i in range(BARCODE_CELLS)]


def decode_barcode(samples: list[float]) -> int | None:
    """Turn per-cell brightness (0-255) back into a frame index. None if it does not validate."""
    if len(samples) != BARCODE_CELLS:
        return None
    bits = [1 if s > 128 else 0 for s in samples]
    if bits[0] != 1 or bits[-1] != 0:
        return None
    n = 0
    for b in bits[1 : 1 + BARCODE_BITS]:
        n = (n << 1) | b
    check = 0
    for b in bits[1 + BARCODE_BITS : 1 + BARCODE_BITS + 4]:
        check = (check << 1) | b
    expected = sum((n >> (4 * i)) & 0xF for i in range(BARCODE_BITS // 4)) & 0xF
    return n if check == expected else None


def read_barcode(frame: np.ndarray, layout: dict) -> int | None:
    """Decode the barcode from a full, unscaled source frame."""
    r = max(1, layout["barcode"]["cell"] // 4)
    samples = []
    for cx, cy in barcode_centers(layout):
        x, y = int(cx), int(cy)
        samples.append(float(frame[y - r : y + r + 1, x - r : x + r + 1].mean()))
    return decode_barcode(samples)


# -- drawing helpers ----------------------------------------------------------


@lru_cache(maxsize=4096)
def _sprite(text: str, size: int, bold: bool = True) -> np.ndarray:
    font = load_font(size, bold)
    left, top, right, bottom = font.getbbox(text)
    w, h = max(1, right - left + 2), max(1, bottom - top + 2)
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((1 - left, 1 - top), text, fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0


def _blit(frame: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int], cx: float, cy: float) -> None:
    h, w = alpha.shape
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, fw), min(y + h, fh)
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y : y1 - y, x0 - x : x1 - x, None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    roi = roi * (1.0 - a) + np.array(color, dtype=np.float32) * a
    frame[y0:y1, x0:x1] = roi.astype(np.uint8)


def _center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


# colours are BGR
WHITE = (255, 255, 255)
AWAY_BG, AWAY_ACCENT = (38, 38, 38), (196, 196, 196)
HOME_BG, HOME_ACCENT = (165, 61, 0), (33, 132, 245)
INFO_BG = (18, 18, 18)
SHOT_BG, SHOT_FG = (20, 20, 120), (80, 220, 255)


class SyntheticRenderer:
    def __init__(
        self,
        script: GameScript,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        bug_offset: float = 0.02,
    ) -> None:
        self.script = script
        self.width, self.height, self.fps = width, height, fps
        self.layout = bug_layout(width, height, bug_offset)
        self.n_frames = int(math.floor(script.duration * fps))
        self._rng = np.random.default_rng(script.seed)
        self._court = self._make_court()
        self._crowd = self._make_crowd()
        self._bug_base = self._make_bug_base()
        bx0, by0, bx1, by1 = self.layout["bug"]
        self._bug_h = by1 - by0
        self._players = [
            (
                self._rng.uniform(0.15, 0.85),
                self._rng.uniform(0.35, 0.8),
                self._rng.uniform(0.05, 0.3),
                self._rng.uniform(0.04, 0.14),
                self._rng.uniform(0.2, 0.9),
                self._rng.uniform(0, 6.28),
                i % 2,
            )
            for i in range(10)
        ]

    # -- static art ------------------------------------------------------
    def _make_court(self) -> np.ndarray:
        h, w = int(self.height * 1.3), int(self.width * 1.5)
        low = self._rng.normal(0, 1, (h // 24, w // 24, 3)).astype(np.float32)
        tex = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        base = np.array([74, 132, 188], dtype=np.float32)
        court = np.clip(base + tex * 22, 0, 255).astype(np.uint8)
        for x in range(0, w, w // 9):
            cv2.line(court, (x, 0), (x + h // 5, h), (225, 235, 240), max(2, self.height // 240))
        cv2.circle(court, (w // 2, h // 2), h // 4, (225, 235, 240), max(2, self.height // 200))
        cv2.rectangle(court, (w // 5, h // 3), (w // 3, h * 2 // 3), (60, 60, 190), max(3, self.height // 160))
        return court

    def _make_crowd(self) -> list[np.ndarray]:
        band_h = int(self.height * 0.2)
        out = []
        for _ in range(8):
            low = self._rng.integers(30, 150, (band_h // 6, self.width // 6, 3), dtype=np.uint8)
            out.append(cv2.resize(low, (self.width, band_h), interpolation=cv2.INTER_NEAREST))
        return out

    def _make_bug_base(self) -> np.ndarray:
        bx0, by0, bx1, by1 = self.layout["bug"]
        w, h = bx1 - bx0, by1 - by0
        base = np.zeros((h, w, 3), dtype=np.uint8)

        def local(box: list[int]) -> tuple[int, int, int, int]:
            return box[0] - bx0, box[1] - by0, box[2] - bx0, box[3] - by0

        blocks = self.layout["blocks"]
        for key, bg in ((AWAY, AWAY_BG), (HOME, HOME_BG), ("info", INFO_BG)):
            x0, y0, x1, y1 = local(blocks[key])
            base[y0:y1, x0:x1] = bg
        for key, accent in (("away_logo", AWAY_ACCENT), ("home_logo", HOME_ACCENT)):
            x0, y0, x1, y1 = local(blocks[key])
            cv2.circle(base, ((x0 + x1) // 2, (y0 + y1) // 2), int(h * 0.36), accent, -1, cv2.LINE_AA)
            cv2.circle(base, ((x0 + x1) // 2, (y0 + y1) // 2), int(h * 0.2), (20, 20, 20), -1, cv2.LINE_AA)
        x0, y0, x1, y1 = local(blocks["shot_box"])
        pad = max(2, h // 9)
        base[y0 + pad : y1 - pad, x0:x1] = SHOT_BG
        # accent strip along the top and a dark frame, like a real bug
        base[0 : max(2, h // 16), :] = (230, 230, 230)
        cv2.rectangle(base, (0, 0), (w - 1, h - 1), (8, 8, 8), max(1, h // 24))
        teams = self.script.teams
        label_size = int(h * 0.56)
        for key, team in (("away_label", AWAY), ("home_label", HOME)):
            cx, cy = _center(list(local(self.layout["fields"][key])))
            _blit(base, _sprite(teams[team]["abbr"], label_size), WHITE, cx, cy)
        return base

    # -- per-frame -----------------------------------------------------------
    def _draw_background(self, frame: np.ndarray, t: float, zoomed: bool = False) -> None:
        ch, cw = self._court.shape[:2]
        max_x, max_y = cw - self.width, ch - self.height
        phase = 0.31 if zoomed else 0.0
        ox = int((0.5 + 0.5 * math.sin(t * 0.21 + phase * 9)) * max_x)
        oy = int((0.5 + 0.5 * math.sin(t * 0.13 + 1.0 + phase * 5)) * max_y)
        frame[:] = self._court[oy : oy + self.height, ox : ox + self.width]
        band = self._crowd[int(t * 10) % len(self._crowd)]
        frame[: band.shape[0]] = band
        r = int(self.height / (20 if zoomed else 30))
        for px, py, ax, ay, speed, ph, side in self._players:
            x = int((px + ax * math.sin(t * speed + ph)) * self.width)
            y = int((py + ay * math.cos(t * speed * 0.8 + ph)) * self.height)
            color = (235, 235, 235) if side == 0 else (170, 70, 10)
            cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y - r), r // 2, (150, 190, 225), -1, cv2.LINE_AA)
        bx = int((0.5 + 0.4 * math.sin(t * 1.3)) * self.width)
        by = int((0.55 + 0.2 * math.sin(t * 2.1)) * self.height)
        cv2.circle(frame, (bx, by), max(4, r // 3), (30, 110, 230), -1, cv2.LINE_AA)

    def _draw_cutaway(self, frame: np.ndarray, t: float) -> None:
        """A crowd shot with a player in close-up: no playing surface in sight."""
        band = self._crowd[int(t * 4) % len(self._crowd)]
        frame[:] = cv2.resize(band, (self.width, self.height), interpolation=cv2.INTER_NEAREST) // 2
        cx = int(self.width * (0.5 + 0.03 * math.sin(t * 1.7)))
        cv2.rectangle(frame, (cx - self.width // 7, int(self.height * 0.55)), (cx + self.width // 7, self.height),
                      (40, 40, 40), -1)
        cv2.circle(frame, (cx, int(self.height * 0.4)), self.height // 8, (150, 190, 225), -1, cv2.LINE_AA)

    def _draw_commercial(self, frame: np.ndarray, t: float) -> None:
        hue = int((t * 12) % 180)
        hsv = np.empty((1, 1, 3), dtype=np.uint8)
        hsv[0, 0] = (hue, 190, 215)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.int32)
        ramp = np.linspace(0.55, 1.0, self.width, dtype=np.float32)[None, :, None]
        frame[:] = np.clip(color[None, None, :] * ramp, 0, 255).astype(np.uint8)
        s = self.height / 720
        stripe = int((t * 160 * s) % (self.width // 2))
        for k in range(-2, 6):
            x = stripe + k * self.width // 4
            cv2.line(frame, (x, 0), (x - int(200 * s), self.height), (245, 245, 245), int(18 * s))
        cv2.putText(frame, "SPONSOR BREAK", (int(190 * s), int(330 * s)), cv2.FONT_HERSHEY_DUPLEX,
                    2.6 * s, (20, 20, 20), int(6 * s), cv2.LINE_AA)
        # decoy numbers sitting exactly where the score bug normally is
        bx0, by0, bx1, by1 = self.layout["bug"]
        decoy = f"CALL 1-800-555-01{int(t) % 90 + 10}   $19.99   24:00"
        cv2.putText(frame, decoy, (bx0, by1 - int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.95 * s,
                    (250, 250, 250), int(2 * s), cv2.LINE_AA)

    def _draw_bug(self, frame: np.ndarray, state: BugState) -> None:
        bx0, by0, bx1, by1 = self.layout["bug"]
        frame[by0:by1, bx0:bx1] = self._bug_base
        h = self._bug_h
        fields = self.layout["fields"]
        score_size, clock_size = int(h * 0.74), int(h * 0.56)
        _blit(frame, _sprite(str(state.away_score), score_size), WHITE, *_center(fields["away_score"]))
        _blit(frame, _sprite(str(state.home_score), score_size), WHITE, *_center(fields["home_score"]))
        if state.period_text:
            _blit(frame, _sprite(state.period_text, int(h * 0.44)), (215, 215, 215), *_center(fields["period"]))
        if state.clock_text:
            _blit(frame, _sprite(state.clock_text, clock_size), WHITE, *_center(fields["clock"]))
        if state.shot_text:
            _blit(frame, _sprite(state.shot_text, clock_size), SHOT_FG, *_center(fields["shot_clock"]))
        if state.anim_team:
            # the scoring team's wordmark sweeps across its block and hides the score
            x0, y0, x1, y1 = self.layout["blocks"][state.anim_team]
            accent = AWAY_ACCENT if state.anim_team == AWAY else HOME_ACCENT
            frame[y0:y1, x0:x1] = accent
            word = _sprite(self.script.teams[state.anim_team]["name"].upper(), int(h * 0.6))
            span = max(1, (x1 - x0) - word.shape[1] - 8)
            cx = x0 + 4 + word.shape[1] / 2 + span * state.anim_progress
            _blit(frame, word, (15, 15, 15), cx, (y0 + y1) / 2)

    def frame(self, n: int) -> np.ndarray:
        t = n / self.fps
        state = self.script.state_at(t)
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        s = self.height / 720
        if state.scene == "commercial":
            self._draw_commercial(frame, t)
        elif state.scene == "replay_hidden":
            self._draw_background(frame, t * 0.5, zoomed=True)
            cv2.putText(frame, "REPLAY", (int(40 * s), int(200 * s)), cv2.FONT_HERSHEY_DUPLEX, 1.2 * s,
                        (255, 255, 255), int(3 * s), cv2.LINE_AA)
        else:
            if state.scene == "live" and self.script.in_cutaway(t):
                self._draw_cutaway(frame, t)
            else:
                self._draw_background(frame, state.source_time)
            self._draw_bug(frame, state)
            if state.scene == "replay_bug":
                cv2.putText(frame, "REPLAY", (int(40 * s), int(200 * s)), cv2.FONT_HERSHEY_DUPLEX, 1.2 * s,
                            (255, 255, 255), int(3 * s), cv2.LINE_AA)
        draw_barcode(frame, n, self.layout)
        tag = f"t={t:7.2f} f={n} {state.scene}"
        cv2.putText(frame, tag, (int(16 * s), int(170 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s,
                    (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
        return frame


# -- audio ------------------------------------------------------------------------


def write_audio(script: GameScript, path: Path, sample_rate: int = 48000) -> None:
    """Crowd noise, a hum under commercials, and a beep at every make."""
    rng = np.random.default_rng(script.seed + 1)
    total = int(script.duration * sample_rate)
    chunk = sample_rate * 10
    beeps = [(ev.make_time, 660.0 if ev.kind == "ft" else 880.0) for ev in script.events]
    kernel = np.ones(24, dtype=np.float32) / 24
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for start in range(0, total, chunk):
            n = min(chunk, total - start)
            tt = (start + np.arange(n, dtype=np.float64)) / sample_rate
            noise = np.convolve(rng.normal(0, 1, n + 23).astype(np.float32), kernel, mode="valid")
            sig = noise * (0.16 + 0.05 * np.sin(tt * 0.7))
            for h0, h1, kind in script.hidden:
                if kind == "commercial" and h1 > tt[0] and h0 < tt[-1]:
                    mask = (tt >= h0) & (tt < h1)
                    sig[mask] = 0.12 * np.sin(2 * np.pi * 330.0 * tt[mask])
            for bt, freq in beeps:
                if bt + 0.15 > tt[0] and bt < tt[-1]:
                    mask = (tt >= bt) & (tt < bt + 0.15)
                    sig[mask] += 0.3 * np.sin(2 * np.pi * freq * (tt[mask] - bt))
            wav.writeframes((np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes())


# -- encode -----------------------------------------------------------------------


def render_video(
    script: GameScript,
    out_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    bug_offset: float = 0.02,
    crf: int = 21,
    preset: str = "veryfast",
    progress=None,
) -> dict:
    """Render the script to ``out_path`` and write ``<out>.truth.json`` and ``<out>.pbp.json``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = SyntheticRenderer(script, width, height, fps, bug_offset)
    audio_path = out_path.with_suffix(".tmp.wav")
    write_audio(script, audio_path)

    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-g", str(fps * 2), "-c:a", "aac", "-b:a", "128k", "-shortest",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for n in range(renderer.n_frames):
            proc.stdin.write(renderer.frame(n).tobytes())
            if progress and n % 300 == 0:
                progress(n, renderer.n_frames)
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while rendering the synthetic game:\n{err}")
    finally:
        if proc.poll() is None:
            proc.kill()
        audio_path.unlink(missing_ok=True)

    truth = script.to_truth(fps=fps, width=width, height=height, layout=layout_truth(renderer.layout))
    truth["n_frames"] = renderer.n_frames
    truth_path = Path(str(out_path) + ".truth.json")
    truth_path.write_text(json.dumps(truth, indent=1), encoding="utf-8")
    pbp_path = Path(str(out_path) + ".pbp.json")
    pbp_path.write_text(
        json.dumps(
            {
                "source": "synthetic",
                "away": script.teams[AWAY]["abbr"],
                "home": script.teams[HOME]["abbr"],
                "events": truth["pbp"],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return truth
