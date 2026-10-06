"""Stage 8: render the cut.

Output is a 1080x1920 MP4: the crop from calibration scaled to 1080 wide and centred on a
black canvas, a title in the top bar, an optional caption in the bottom bar, nothing
drawn over the video. Hard video cuts; the broadcast audio is crossfaded across each cut.

One ffmpeg run, one encode: every segment is its own seeked input (cheap, and frame
accurate when transcoding), trimmed by a filter graph, concatenated, cropped and padded.
Seeking each input is what keeps a five minute cut from a 2.5 hour file from decoding
the whole 2.5 hours.

Audio stays in sync by construction: each segment's audio is taken half a crossfade longer
on both sides, and each crossfade then removes exactly what was added.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from ..config import ffmpeg_bin
from ..fonts import load_font
from . import intervals as iv
from .geometry import OUTPUT_H, OUTPUT_W, crop_from_norm, output_placement
from .probe import MAX_EXPORT_FPS, Probe

MAX_INPUTS = 48  # segments per ffmpeg run; longer cuts are rendered in batches
CRF = 18
AUDIO_BITRATE = "192k"
DEFAULT_CROSSFADE_MS = 80
AUDIO_CHOICES = ("broadcast", "levelled", "none")
# about what Instagram, TikTok and YouTube normalise to; applied once over the whole cut
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


class ExportError(RuntimeError):
    pass


@dataclass
class PlannedSegment:
    src_in: float  # frame-aligned source time
    n_frames: int
    first_frame: int  # source frame index of the first frame

    def duration(self, fps: Fraction) -> float:
        return float(self.n_frames / fps)


@dataclass
class ExportPlan:
    fps: Fraction  # source frame rate, used for cutting
    out_fps: Fraction
    segments: list[PlannedSegment]
    crop: tuple[int, int, int, int]  # x, y, w, h in display pixels
    placement: tuple[int, int, int]  # scaled_w, scaled_h, y offset on the canvas
    half: float  # half the audio crossfade, seconds (0 = hard audio cuts)
    notes: list[str] = field(default_factory=list)
    # "broadcast": the game's sound as it is; "levelled": loudness evened out for the
    # platforms; "none": a silent track, for music laid over the cut
    audio: str = "broadcast"

    @property
    def duration(self) -> float:
        return float(sum(s.n_frames for s in self.segments) / self.fps)

    @property
    def crossfade(self) -> float:
        return 2 * self.half

    def to_dict(self) -> dict:
        return {
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "out_fps": f"{self.out_fps.numerator}/{self.out_fps.denominator}",
            "duration": round(self.duration, 3),
            "crop": list(self.crop),
            "canvas": [OUTPUT_W, OUTPUT_H],
            "video_on_canvas": {"w": self.placement[0], "h": self.placement[1], "y": self.placement[2]},
            "audio_crossfade_ms": round(self.crossfade * 1000, 1),
            "audio": self.audio,
            "segments": [
                {"src_in": round(s.src_in, 4), "src_out": round(s.src_in + s.duration(self.fps), 4), "frames": s.n_frames}
                for s in self.segments
            ],
        }


def plan_export(
    probe: Probe,
    segments: list[tuple[float, float]] | list[list[float]],
    crop_norm: list[float],
    crossfade: bool = True,
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
    audio: str = "broadcast",
) -> ExportPlan:
    """Snap segments to source frames and work out the geometry."""
    if audio not in AUDIO_CHOICES:
        raise ValueError(f"audio must be one of {', '.join(AUDIO_CHOICES)}")
    fps = Fraction(probe.fps_num, probe.fps_den)
    frame = float(1 / fps)
    total_frames = int(probe.duration * fps)
    spans = iv.merge([(float(a), float(b)) for a, b in segments], gap=1.5 * frame)
    planned: list[PlannedSegment] = []
    for a, b in spans:
        k_in = max(0, int(round(a * fps)))
        k_out = min(total_frames, int(round(b * fps)))
        if k_out - k_in < 2:
            continue
        planned.append(PlannedSegment(float(k_in / fps), k_out - k_in, k_in))
    if not planned:
        raise ExportError("Nothing to export: no enabled clips.")

    half = 0.0
    if crossfade and len(planned) > 1:
        half_frames = max(1, int(round(crossfade_ms / 2000 * float(fps))))
        half_frames = min(half_frames, max(1, min(s.n_frames for s in planned) // 4))
        half = float(half_frames / fps)

    crop = crop_from_norm(crop_norm, probe.display_width, probe.height)
    return ExportPlan(
        fps=fps,
        out_fps=min(fps, Fraction(MAX_EXPORT_FPS)),
        segments=planned,
        crop=crop,
        placement=output_placement(crop[2], crop[3]),
        half=half,
        audio=audio,
    )


# -- title / caption overlay -----------------------------------------------------------


def _wrap(text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if font.getlength(trial) <= max_width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return [ln for ln in lines if ln]


def render_overlay(title: str, caption: str, placement: tuple[int, int, int], path: Path) -> Path | None:
    """Transparent 1080x1920 PNG with the title in the top bar and the caption in the bottom bar.

    Text is rendered here rather than with ffmpeg's drawtext so that any title (quotes,
    colons, percent signs) works without filter escaping. Returns None when there is no text.
    """
    title, caption = (title or "").strip(), (caption or "").strip()
    if not title and not caption:
        return None
    _, scaled_h, y0 = placement
    top_bar, bottom_bar = y0, OUTPUT_H - (y0 + scaled_h)
    img = Image.new("RGBA", (OUTPUT_W, OUTPUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 64
    max_w = OUTPUT_W - 2 * margin

    if title and top_bar >= 90:
        size, lines, font = 78, [title], load_font(78, bold=True)
        for size in (78, 70, 62, 54, 48, 42, 36):
            font = load_font(size, bold=True)
            lines = _wrap(title, font, max_w)
            height = int(len(lines) * size * 1.18)
            if len(lines) <= 2 and height <= top_bar - 60:
                break
        lines = lines[:3]
        line_h = int(size * 1.18)
        block_h = line_h * len(lines)
        # sit just above the video; phone UI covers the very top of the screen
        y = max(20, top_bar - block_h - max(36, int(top_bar * 0.09)))
        for i, line in enumerate(lines):
            w = font.getlength(line)
            draw.text(((OUTPUT_W - w) / 2, y + i * line_h), line, font=font, fill=(255, 255, 255, 255))

    if caption and bottom_bar >= 70:
        size = 40
        font = load_font(size, bold=False)
        lines = _wrap(caption, font, max_w)[:2]
        y = y0 + scaled_h + max(28, int(bottom_bar * 0.07))
        for i, line in enumerate(lines):
            w = font.getlength(line)
            draw.text(((OUTPUT_W - w) / 2, y + i * int(size * 1.25)), line, font=font, fill=(220, 220, 220, 255))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# -- ffmpeg ------------------------------------------------------------------------------


@lru_cache
def ffmpeg_major_version() -> int | None:
    try:
        out = subprocess.run([ffmpeg_bin(), "-version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"ffmpeg version n?(\d+)\.", out)
    return int(m.group(1)) if m else None


def _graph_args(graph: str, workdir: Path, name: str) -> list[str]:
    """Pass the filter graph from a file: long cuts overflow the Windows command line otherwise."""
    path = workdir / name
    path.write_text(graph, encoding="utf-8")
    major = ffmpeg_major_version()
    if major is not None and major < 7:
        return ["-filter_complex_script", str(path)]
    return ["-/filter_complex", str(path)]


def _fnum(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def build_batch(
    plan: ExportPlan,
    probe: Probe,
    first: int,
    last: int,
    overlay: Path | None,
    final: bool,
) -> tuple[list[str], str, list[str]]:
    """Input args, filter graph and output args for segments [first, last] in one ffmpeg run.

    ``final`` is False for an intermediate batch (lossless audio, joined afterwards).
    """
    fps = plan.fps
    frame = float(1 / fps)
    segs = plan.segments[first : last + 1]
    n = len(segs)
    inputs: list[str] = []
    v_lines: list[str] = []
    a_lines: list[str] = []
    decode_threads = ["-threads", "2"] if n > 12 else []
    pre = ("bwdif=mode=send_frame:deint=all," if probe.interlaced else "")

    for j, seg in enumerate(segs):
        h_before = plan.half if j > 0 else 0.0
        h_after = plan.half if j < n - 1 else 0.0
        seg_dur = seg.duration(fps)
        h_after = min(h_after, max(0.0, probe.duration - (seg.src_in + seg_dur)))
        h_before = min(h_before, seg.src_in)
        seek = max(0.0, seg.src_in - h_before - frame / 2)
        bias = seg.src_in - h_before - seek  # stream time of the first audio sample we want
        read = bias + h_before + seg_dur + h_after + 0.5
        inputs += [*decode_threads, "-ss", _fnum(seek), "-t", _fnum(read), "-i", probe.path]
        # the wanted first frame sits at bias + h_before; cut half a frame before it
        v_start = max(0.0, bias + h_before - frame / 2)
        v_lines.append(
            f"[{j}:v]{pre}trim=start={_fnum(v_start)}:end={_fnum(v_start + seg_dur)},setpts=PTS-STARTPTS[v{j}]"
        )
        if probe.has_audio:
            a_dur = h_before + seg_dur + h_after
            a_lines.append(
                f"[{j}:a]atrim=start={_fnum(bias)}:duration={_fnum(a_dur)},asetpts=PTS-STARTPTS,"
                f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{j}]"
            )

    cx, cy, cw, ch = plan.crop
    scaled_w, scaled_h, pad_y = plan.placement
    chain = []
    if probe.anamorphic:
        chain.append(f"scale={probe.display_width}:{probe.height}")
    chain += [
        f"crop={cw}:{ch}:{cx}:{cy}",
        f"scale={scaled_w}:{scaled_h}:flags=lanczos",
        "setsar=1",
    ]
    if plan.out_fps != fps:
        chain.append(f"fps={plan.out_fps.numerator}/{plan.out_fps.denominator}")
    chain.append(f"pad={OUTPUT_W}:{OUTPUT_H}:0:{pad_y}:color=black")
    graph = list(v_lines)
    graph.append("".join(f"[v{j}]" for j in range(n)) + f"concat=n={n}:v=1:a=0[vcat]")
    next_input = n
    if overlay is not None:
        inputs += ["-i", str(overlay)]
        graph.append(f"[vcat]{','.join(chain)}[vbase]")
        graph.append(f"[vbase][{next_input}:v]overlay=0:0:format=auto,format=yuv420p[vout]")
        next_input += 1
    else:
        graph.append(f"[vcat]{','.join(chain)},format=yuv420p[vout]")

    total = sum(s.duration(fps) for s in segs)
    if probe.has_audio and plan.audio != "none":
        graph += a_lines
        prev = "a0"
        for j in range(1, n):
            out = f"x{j}"
            if plan.half > 0:
                graph.append(f"[{prev}][a{j}]acrossfade=d={_fnum(plan.crossfade)}:c1=tri:c2=tri[{out}]")
            else:
                graph.append(f"[{prev}][a{j}]concat=n=2:v=0:a=1[{out}]")
            prev = out
        tail = []
        if first > 0:
            tail.append("afade=t=in:st=0:d=0.02")
        if last < len(plan.segments) - 1:
            tail.append(f"afade=t=out:st={_fnum(max(0.0, total - 0.02))}:d=0.02")
        if plan.audio == "levelled" and final:
            tail.append(LOUDNORM)  # a batched render levels in the final join instead
        graph.append(f"[{prev}]{','.join(tail) if tail else 'anull'}[aout]")
        audio_map = ["-map", "[aout]"]
    else:
        inputs += ["-f", "lavfi", "-t", _fnum(total), "-i", "anullsrc=r=48000:cl=stereo"]
        audio_map = ["-map", f"{next_input}:a"]

    rate = f"{plan.out_fps.numerator}/{plan.out_fps.denominator}"
    out_args = [
        "-map", "[vout]", *audio_map,
        "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", str(CRF),
        "-pix_fmt", "yuv420p", "-r", rate, "-fps_mode", "cfr",
    ]
    if final:
        out_args += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2", "-movflags", "+faststart"]
    else:
        out_args += ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]
    return inputs, ";\n".join(graph), out_args


def _run(cmd: list[str], total_seconds: float, on_progress: Callable[[float], None] | None,
         should_stop: Callable[[], bool] | None = None) -> None:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if should_stop and should_stop():
                proc.kill()
                raise InterruptedError("export cancelled")  # a cancel, not a failure
            if line.startswith("out_time_us=") and on_progress:
                try:
                    done = int(line.split("=", 1)[1]) / 1e6
                except ValueError:
                    continue
                on_progress(min(1.0, done / max(total_seconds, 1e-6)))
        err = proc.stderr.read() if proc.stderr else ""
        if proc.wait() != 0:
            raise ExportError(f"ffmpeg failed while rendering:\n{err.strip()[-1500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()


def render(
    plan: ExportPlan,
    probe: Probe,
    out_path: Path,
    overlay: Path | None = None,
    workdir: Path | None = None,
    progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    chapters: list[tuple[float, float, str]] | None = None,
) -> Path:
    """Render the plan to ``out_path``. ``chapters``: (start, end, title) in output
    seconds, written into the MP4 so players and YouTube can list the clips."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = workdir or out_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    base = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    report = ["-progress", "pipe:1", "-nostats"]
    n = len(plan.segments)
    total = plan.duration
    t0 = time.time()
    meta_path = write_chapters(workdir / "export_chapters.txt", chapters) if chapters else None

    def with_chapters(inputs: list[str]) -> tuple[list[str], list[str]]:
        """The metadata file as one more input, mapped onto the output."""
        if meta_path is None:
            return inputs, []
        return [*inputs, "-i", str(meta_path)], ["-map_metadata", str(inputs.count("-i"))]

    best = [0.0]

    def tell(frac: float, done_before: float, batch_total: float) -> None:
        if progress:
            # ffmpeg reports the slower of its streams, which can step back a little
            overall = max(best[0], (done_before + frac * batch_total) / max(total, 1e-6))
            best[0] = overall
            progress(min(0.99, overall), f"Rendering {overall * total:.0f}s of {total:.0f}s")

    if n <= MAX_INPUTS:
        inputs, graph, out_args = build_batch(plan, probe, 0, n - 1, overlay, final=True)
        inputs, meta_map = with_chapters(inputs)
        cmd = [*base, *inputs, *_graph_args(graph, workdir, "export_graph.txt"), *out_args, *meta_map, *report, str(out_path)]
        _run(cmd, total, lambda f: tell(f, 0.0, total), should_stop)
    else:
        parts: list[Path] = []
        done = 0.0
        for b, first in enumerate(range(0, n, MAX_INPUTS)):
            last = min(n, first + MAX_INPUTS) - 1
            part = workdir / f"export_part_{b:02d}.mkv"
            inputs, graph, out_args = build_batch(plan, probe, first, last, overlay, final=False)
            batch_total = sum(s.duration(plan.fps) for s in plan.segments[first : last + 1])
            cmd = [*base, *inputs, *_graph_args(graph, workdir, f"export_graph_{b:02d}.txt"), *out_args, *report, str(part)]
            _run(cmd, batch_total, lambda f, d=done, bt=batch_total: tell(f, d, bt), should_stop)
            parts.append(part)
            done += batch_total
        listing = workdir / "export_parts.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        # video is copied; audio is encoded once over the whole cut, so there are no AAC seams
        inputs, meta_map = with_chapters(["-f", "concat", "-safe", "0", "-i", str(listing)])
        cmd = [
            *base, *inputs,
            "-c:v", "copy", *(["-af", LOUDNORM] if plan.audio == "levelled" else []),
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
            *meta_map, "-movflags", "+faststart", str(out_path),
        ]
        _run(cmd, total, None, should_stop)
        for p in parts:
            p.unlink(missing_ok=True)
        listing.unlink(missing_ok=True)
    if progress:
        progress(1.0, f"Rendered {total:.0f}s in {time.time() - t0:.0f}s")
    return out_path


# -- sidecars -----------------------------------------------------------------------------


def cutlist_rows(clips: list[dict]) -> list[dict]:
    """Each clip with where it lands in the output (``out_start``, ``out_end``)."""
    out_t = 0.0
    rows = []
    for clip in clips:
        dur = iv.total([tuple(s) for s in clip["segments"]])
        rows.append({**clip, "out_start": round(out_t, 3), "out_end": round(out_t + dur, 3)})
        out_t += dur
    return rows


def write_cutlist(path: Path, meta: dict, clips: list[dict], plan: ExportPlan) -> list[dict]:
    """cutlist.json: every clip with source in/out, game time, score, scorer, confidence.
    Returns the rows it wrote."""
    rows = cutlist_rows(clips)
    doc = {**meta, "render": plan.to_dict(), "clips": rows}
    path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return rows


def _mmss(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _game_clock(clock: float | None) -> str:
    if clock is None:
        return ""
    return f"{int(clock // 60)}:{int(clock % 60):02d}" if clock >= 60 else f"{clock:.1f}"


def clip_title(row: dict, format_period: Callable[[int | None], str]) -> str:
    """One line for a chapter or a timestamp: "Q3 9:40 · J. Brunson +3 · 55-81"."""
    when = f"{format_period(row.get('period'))} {_game_clock(row.get('game_clock', row.get('clock')))}".strip()
    who = row.get("scorer") or (row.get("description") or "").split(" (")[0].strip()
    what = f"+{row['points']}" if row.get("points") else {"block": "block", "steal": "steal"}.get(row.get("kind", ""), "")
    play = " ".join(x for x in (who, what) if x)
    score = ""
    if row.get("score_away") is not None and row.get("score_home") is not None:
        score = f"{row['score_away']}-{row['score_home']}"
    return " · ".join(x for x in (when, play, score) if x) or "Clip"


def chapters_for(rows: list[dict], format_period: Callable[[int | None], str]) -> list[tuple[float, float, str]]:
    return [(float(r["out_start"]), float(r["out_end"]), clip_title(r, format_period)) for r in rows]


def timestamps_block(rows: list[dict], format_period: Callable[[int | None], str]) -> str:
    """The YouTube-style list: a time and a line per clip, the first at 0:00."""
    return "\n".join(f"{_mmss(r['out_start'])} {clip_title(r, format_period)}" for r in rows)


def _meta_escape(text: str) -> str:
    return "".join("\\" + ch if ch in "=;#\\" else ch for ch in text.replace("\n", " "))


def write_chapters(path: Path, chapters: list[tuple[float, float, str]]) -> Path:
    """An ffmetadata file with one chapter per clip."""
    lines = [";FFMETADATA1"]
    for start, end, title in chapters:
        if end - start < 0.01:
            continue
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round(start * 1000))}", f"END={int(round(end * 1000))}", f"title={_meta_escape(title)}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- helpers used around export -------------------------------------------------------------


def thumbnail(probe: Probe, t: float, crop_norm: list[float], path: Path, width: int = 240) -> None:
    """Small JPEG of the cropped picture at time ``t`` (review list thumbnails)."""
    x, y, w, h = crop_from_norm(crop_norm, probe.display_width, probe.height)
    height = int(round(width * h / w / 2) * 2)
    filters = []
    if probe.anamorphic:
        filters.append(f"scale={probe.display_width}:{probe.height}")
    filters += [f"crop={w}:{h}:{x}:{y}", f"scale={width}:{height}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-y", "-v", "error", "-ss", _fnum(max(0.0, t)), "-i", probe.path,
        "-frames:v", "1", "-vf", ",".join(filters), "-q:v", "4", str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=False)


def make_proxy(
    probe: Probe, out_path: Path, progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None, seekable: bool = False, threads: int | None = None,
) -> Path:
    """Browser-playable copy of the source for the review screen.

    ``seekable``: re-encode for playing over a network, at most 480 tall with a keyframe
    every second, so a jump to any clip starts at once and costs little bandwidth.
    Otherwise (a source the browser cannot play: MKV, TS, AC-3 audio...) H.264 video is
    copied as is and anything else is transcoded to 540p. Audio becomes AAC. The timeline
    is unchanged, so clip times apply to the proxy directly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    copy_video = (
        not seekable and probe.video_codec == "h264" and probe.pix_fmt in ("yuv420p", "yuvj420p") and not probe.interlaced
    )
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", probe.path, "-map", "0:v:0"]
    if probe.has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "96k" if seekable else "128k", "-ac", "2"]
    if copy_video:
        cmd += ["-c:v", "copy"]
    else:
        height = min(480 if seekable else 540, max(2, probe.height)) // 2 * 2
        vf = ("bwdif=mode=send_frame," if probe.interlaced else "") + f"scale=-2:{height}"
        preset, crf = ("superfast", "27") if seekable else ("veryfast", "26")
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p"]
        if seekable:
            gop = max(1, int(round(probe.fps or 30.0)))
            cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
        if threads:
            cmd += ["-threads", str(threads)]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(out_path)]
    _run(cmd, probe.duration, (lambda f: progress(f, "Preparing a preview copy")) if progress else None, should_stop)
    return out_path
