"""Stage 1: ffprobe the source file."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from ..config import ffprobe_bin

MIN_HEIGHT = 480
MAX_EXPORT_FPS = 60

# What a browser <video> element can play straight from the file.
_BROWSER_CONTAINERS = ("mp4", "mov", "m4v", "webm")
_BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
_BROWSER_AUDIO = {"aac", "mp3", "opus", "vorbis", "flac"}


class ProbeError(ValueError):
    """The file cannot be used as a source."""


@dataclass
class Probe:
    path: str
    container: str
    duration: float
    size_bytes: int
    bit_rate: int
    width: int  # coded width
    height: int
    display_width: int  # width once pixels are square
    sar_num: int
    sar_den: int
    fps: float
    fps_num: int
    fps_den: int
    video_codec: str
    pix_fmt: str
    interlaced: bool
    rotation: int
    start_time: float
    has_audio: bool
    audio_codec: str | None
    audio_channels: int
    audio_sample_rate: int
    browser_playable: bool

    @property
    def anamorphic(self) -> bool:
        return self.display_width != self.width

    @property
    def export_fps(self) -> Fraction:
        """Source frame rate, capped at 60."""
        rate = Fraction(self.fps_num, self.fps_den)
        return min(rate, Fraction(MAX_EXPORT_FPS))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Probe:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


def _rate(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        rate = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def run_ffprobe(path: str | Path) -> dict:
    cmd = [
        ffprobe_bin(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe could not read {path}: {proc.stderr.strip()[:400]}")
    return json.loads(proc.stdout or "{}")


def parse_probe(path: str | Path, raw: dict) -> Probe:
    streams = raw.get("streams", [])
    fmt = raw.get("format", {})
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise ProbeError("No video stream found in this file.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(side["rotation"]) % 360
    if "rotate" in (video.get("tags") or {}):
        rotation = int(video["tags"]["rotate"]) % 360

    sar = _rate((video.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or Fraction(1)
    display_width = int(round(width * sar / 2) * 2) if sar != 1 else width

    if min(height, display_width) < MIN_HEIGHT or height < MIN_HEIGHT:
        raise ProbeError(f"Video is {display_width}x{height}; files under 480p are rejected.")

    rate = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    if rate is None:
        raise ProbeError("Could not determine the frame rate.")
    # limit_denominator keeps 30000/1001-style rates exact without absurd fractions
    rate = rate.limit_denominator(1001)

    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise ProbeError("Could not determine the duration.")

    container = (fmt.get("format_name") or "").lower()
    vcodec = (video.get("codec_name") or "").lower()
    acodec = (audio.get("codec_name") or "").lower() if audio else None
    playable = (
        any(c in container.split(",") for c in _BROWSER_CONTAINERS)
        and vcodec in _BROWSER_VIDEO
        and (acodec is None or acodec in _BROWSER_AUDIO)
        and (video.get("pix_fmt") or "yuv420p") in ("yuv420p", "yuvj420p")
        and rotation == 0
    )

    return Probe(
        path=str(path),
        container=container,
        duration=duration,
        size_bytes=int(fmt.get("size") or 0),
        bit_rate=int(fmt.get("bit_rate") or 0),
        width=width,
        height=height,
        display_width=display_width,
        sar_num=sar.numerator,
        sar_den=sar.denominator,
        fps=float(rate),
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        video_codec=vcodec,
        pix_fmt=video.get("pix_fmt") or "",
        interlaced=(video.get("field_order") or "progressive") not in ("progressive", "unknown"),
        rotation=rotation,
        start_time=float(fmt.get("start_time") or 0.0),
        has_audio=audio is not None,
        audio_codec=acodec,
        audio_channels=int(audio.get("channels", 0)) if audio else 0,
        audio_sample_rate=int(audio.get("sample_rate", 0)) if audio else 0,
        browser_playable=playable,
    )


def probe_file(path: str | Path, out_json: Path | None = None) -> Probe:
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"File not found: {path}")
    probe = parse_probe(path, run_ffprobe(path))
    if out_json is not None:
        out_json.write_text(json.dumps(probe.to_dict(), indent=1), encoding="utf-8")
    return probe
