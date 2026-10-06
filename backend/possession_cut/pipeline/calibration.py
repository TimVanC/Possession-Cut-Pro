"""Stage 2: locate the score bug and its fields, validate, and save a broadcaster template.

Order of attempts on a new job:

1. Existing templates: if a saved bug matches these frames, reuse it (no network).
2. Claude vision for where the bug is and which field is which, snapped to locally
   detected text for pixel accuracy.
3. Fully local detection when Claude is unavailable or fails.

Every route ends in the same validation: the bug region must be static across frames
and the fields must actually read as scores and a clock.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..ai.claude import ClaudeClient, ClaudeError
from ..sports.base import SportAdapter, same_team_code
from .bugreader import CHARSETS, VISIBLE_THRESHOLD, BugReader, build_reference
from .calib_claude import detect_with_claude
from .calib_local import LocalDetector, separate_fields, size_field_boxes, snap_to_text
from .frames import extract_frames, save_jpeg
from .geometry import Box, compute_crop, crop_to_norm, iou, norm_box, to_px
from .ocr import OcrEngine, parse_clock, parse_period, parse_score, parse_shot_clock, prepare_field
from .probe import Probe

log = logging.getLogger(__name__)

N_FRAMES = 12


class CalibrationError(RuntimeError):
    pass


@dataclass
class Calibration:
    bug: list[float]
    fields: dict[str, list[float]]
    crop: list[float]  # normalized x, y, w, h
    frame_width: int
    frame_height: int
    source: str = "local"  # template | claude | local | manual
    confidence: float = 0.0
    teams: dict[str, str] = field(default_factory=dict)
    broadcaster: str = ""
    template_id: int | None = None
    template_name: str = ""
    frames: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    confirmed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Calibration:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def sample_times(duration: float, n: int = N_FRAMES) -> list[float]:
    """Evenly spread sample points, skipping the very start and end of the file."""
    return [round(duration * (0.04 + 0.92 * (i + 0.5) / n), 2) for i in range(n)]


def parse_field(name: str, text: str):
    """Typed value of one field read, or None when the text is not valid for that field."""
    if name.endswith("_score"):
        return parse_score(text)
    if name == "clock":
        return parse_clock(text)
    if name == "shot_clock":
        return parse_shot_clock(text)
    if name == "period":
        return parse_period(text)
    return text.strip() or None


def make_reader(
    cal: Calibration,
    adapter: SportAdapter,
    frame_w: int,
    frame_h: int,
    engine: OcrEngine | None,
    reference: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> BugReader:
    return BugReader(
        tuple(cal.bug), {k: tuple(v) for k, v in cal.fields.items()}, adapter.bug_fields,
        frame_w, frame_h, engine, reference, mask,
    )


def read_frame(reader: BugReader, frame: np.ndarray, check_visible: bool = True) -> dict:
    """Bug reads for one full frame, shaped for the calibration screen."""
    x, y, w, h = reader.roi
    result = reader.read(frame[y : y + h, x : x + w], check_visible=check_visible)
    reads = {}
    for name, fr in result.fields.items():
        value = parse_field(name, fr.text)
        reads[name] = {"text": fr.text, "conf": round(fr.conf, 3), "value": value, "ok": value is not None}
    return {"visible": result.visible, "similarity": round(result.similarity, 3), "reads": reads}


def reference_from_frames(
    frames: list[np.ndarray | None], visible: list[int], bug: Box, fields: dict[str, Box]
) -> tuple[np.ndarray, np.ndarray] | None:
    crops = []
    field_px: list[tuple[int, int, int, int]] = []
    for i in visible:
        frame = frames[i]
        if frame is None:
            continue
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = to_px(bug, w, h)
        crops.append(frame[y0:y1, x0:x1])
        if not field_px:
            for box in fields.values():
                fx0, fy0, fx1, fy1 = to_px(tuple(box), w, h)
                field_px.append((fx0 - x0, fy0 - y0, fx1 - x0, fy1 - y0))
    if not crops:
        return None
    size = crops[0].shape[:2]
    crops = [c if c.shape[:2] == size else cv2.resize(c, (size[1], size[0])) for c in crops]
    return build_reference(crops, field_px)


def typical_bug_image(frames: list[np.ndarray | None], visible: list[int], bug: Box) -> np.ndarray | None:
    """Per-pixel median of the bug across the frames that show it."""
    crops = []
    for i in visible:
        frame = frames[i]
        if frame is None:
            continue
        x0, y0, x1, y1 = to_px(bug, frame.shape[1], frame.shape[0])
        crops.append(frame[y0:y1, x0:x1])
    if not crops:
        return None
    size = crops[0].shape[:2]
    crops = [c if c.shape[:2] == size else cv2.resize(c, (size[1], size[0])) for c in crops]
    return np.median(np.stack(crops), axis=0).astype(np.uint8)


def validate(
    cal: Calibration,
    adapter: SportAdapter,
    frames: list[np.ndarray | None],
    times: list[float],
    files: list[str],
    engine: OcrEngine,
    reference: np.ndarray | None,
    mask: np.ndarray | None,
) -> None:
    """Read every sampled frame with the calibration and fill in frames/checks/confidence."""
    cal.frames = []
    seen = clock_ok = score_ok = 0
    reader: BugReader | None = None
    for i, frame in enumerate(frames):
        entry = {"index": i, "time": times[i], "file": files[i], "visible": False, "similarity": 0.0, "reads": {}}
        if frame is not None:
            if reader is None:
                reader = make_reader(cal, adapter, frame.shape[1], frame.shape[0], engine, reference, mask)
            entry.update(read_frame(reader, frame))
            if entry["visible"]:
                seen += 1
                reads = entry["reads"]
                clock_ok += bool(reads.get("clock", {}).get("ok")) or not adapter.has_clock
                score_ok += bool(reads.get("away_score", {}).get("ok") and reads.get("home_score", {}).get("ok"))
        cal.frames.append(entry)
    cal.checks.update(
        frames_sampled=len(frames),
        frames_with_bug=seen,
        clock_read_rate=round(clock_ok / seen, 3) if seen else 0.0,
        score_read_rate=round(score_ok / seen, 3) if seen else 0.0,
    )
    required = [s.name for s in adapter.bug_fields if s.required]
    missing = [name for name in required if name not in cal.fields]
    if missing:
        cal.warnings.append("Fields not located: " + ", ".join(missing) + ". Draw them on the calibration screen.")
    if seen == 0:
        cal.confidence = 0.0
        cal.warnings.append("The score bug was not seen in any sampled frame.")
        return
    static_ok = 1.0
    inside, outside = cal.checks.get("static_inside"), cal.checks.get("static_outside")
    if inside is not None and outside is not None:
        # scores, clocks and status lines change, so a real bug is far from fully static;
        # what matters is that it is clearly steadier than the picture around it
        static_ok = 1.0 if (inside >= 0.35 and inside - outside >= 0.2) else 0.6
        if static_ok < 1.0:
            cal.warnings.append("The bug region does not stand out as static against its surroundings; check the box.")
    cal.confidence = round(
        min(1.0, seen / 3) * cal.checks["clock_read_rate"] * cal.checks["score_read_rate"] * static_ok
        * (0.5 if missing else 1.0),
        3,
    )
    if cal.checks["score_read_rate"] < 0.7:
        cal.warnings.append("Scores read cleanly in under 70% of frames; adjust the score boxes.")
    if adapter.has_clock and cal.checks["clock_read_rate"] < 0.7:
        cal.warnings.append("The game clock read cleanly in under 70% of frames; adjust the clock box.")


def calibrate(
    probe: Probe,
    job_dir: Path,
    adapter: SportAdapter,
    engine: OcrEngine,
    claude: ClaudeClient | None = None,
    templates: list | None = None,
    expected_teams: dict[str, str] | None = None,
    progress=None,
    should_stop=None,
) -> tuple[Calibration, np.ndarray | None, np.ndarray | None]:
    """Run calibration for one source file.

    Returns the calibration plus the bug reference image and static mask (to be stored
    with the template). Raises CalibrationError only when no frames can be decoded;
    a bug that cannot be found comes back as a low-confidence calibration with warnings
    so the user can draw the box by hand. ``should_stop`` is asked at every step, and a
    yes raises InterruptedError (a cancel from the page).
    """
    from . import templates as tmpl

    def step(fraction: float, message: str) -> None:
        if should_stop and should_stop():
            raise InterruptedError("calibration cancelled")
        if progress:
            progress(fraction, message)

    calib_dir = job_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    times = sample_times(probe.duration)
    step(0.05, "Sampling frames")
    frames = extract_frames(probe, times)
    if all(f is None for f in frames):
        raise CalibrationError("Could not decode any frames from this file.")
    files = []
    for i, frame in enumerate(frames):
        name = f"frame_{i:02d}.jpg"
        if frame is not None:
            save_jpeg(frame, calib_dir / name)
        files.append(name)
    fw, fh = probe.display_width, probe.height
    field_names = [s.name for s in adapter.bug_fields]

    # 1. a saved template that matches these frames
    step(0.2, "Checking saved templates")
    for template in templates or []:
        match = tmpl.match_template(template, frames, adapter)
        if match is None:
            continue
        ref, mask = match
        cal = Calibration(
            bug=list(template.bug),
            fields={k: list(v) for k, v in template.fields.items()},
            crop=list(template.crop) or crop_to_norm(compute_crop(tuple(template.bug), fw, fh), fw, fh),
            frame_width=fw, frame_height=fh, source="template",
            broadcaster=template.broadcaster, template_id=template.id, template_name=template.name,
        )
        validate(cal, adapter, frames, times, files, engine, ref, mask)
        if cal.confidence >= 0.5:
            step(1.0, f"Matched template {template.name}")
            return cal, ref, mask
        log.info("template %s matched visually but read poorly (%.2f); detecting afresh", template.name, cal.confidence)

    # 2/3. detect
    warnings: list[str] = []
    detector = LocalDetector(frames, engine)
    step(0.3, "Reading on-screen text")
    clock_cluster, local_visible = detector.find_clock_row()

    bug: Box | None = None
    roles: dict[str, Box] = {}
    teams: dict[str, str] = {}
    broadcaster = ""
    source = "local"
    visible: list[int] = local_visible
    static = None
    clusters: list = []

    if claude is not None and claude.available:
        step(0.45, "Asking Claude to locate the score bug")
        try:
            found = detect_with_claude(claude, frames, adapter.bug_fields, adapter.name, prefer=local_visible)
        except ClaudeError as exc:
            warnings.append(f"Claude vision was not used: {exc}")
            found = None
        if found is not None:
            source, teams, broadcaster = "claude", found.teams, found.broadcaster
            visible = found.visible
            bug = found.bug
            # tighten to the static region when the two agree
            seed = clock_cluster.box if clock_cluster is not None else found.roles.get("clock") or bug
            static_box, static = detector.static_region(visible, seed)
            if static_box is not None and iou(static_box, bug) >= 0.45:
                bug = static_box
            text_height = (clock_cluster.box[3] - clock_cluster.box[1]) if clock_cluster is not None else None
            observations = detector.fine_text(visible, bug, text_height=text_height)
            from .calib_local import _cluster

            clusters = _cluster(observations)
            roles = snap_to_text(found.roles, clusters)
            row = clock_cluster.box if clock_cluster is not None else roles.get("clock")
            if row is not None:
                roles = separate_fields(detector.snap_to_ink(roles, detector.ink_runs(visible, bug, row)), roles)
            warnings.extend(found.notes)
    elif claude is not None and claude.unavailable_reason:
        warnings.append(f"Claude vision was not used: {claude.unavailable_reason}")

    if bug is None or not roles:
        step(0.6, "Detecting the score bug locally")
        local = detector.detect(field_names, expected_teams)
        if local is None:
            # nothing found: hand the user a sensible starting box to drag
            cal = Calibration(
                bug=[0.25, 0.86, 0.75, 0.95], fields={}, crop=[0.0, 0.0, 1.0, 1.0],
                frame_width=fw, frame_height=fh, source="manual", confidence=0.0,
                warnings=[*warnings, "No score bug was found automatically. Draw the box and fields by hand."],
            )
            cal.crop = crop_to_norm(compute_crop(tuple(cal.bug), fw, fh), fw, fh)
            cal.frames = [
                {"index": i, "time": times[i], "file": files[i], "visible": False, "similarity": 0.0, "reads": {}}
                for i in range(len(frames))
            ]
            return cal, None, None
        source = "local"
        bug, roles, visible, clusters = local.bug, local.roles, local.visible, local.clusters
        teams = teams or local.teams
        warnings.extend(local.notes)
        _, static = detector.static_region(visible, clock_cluster.box if clock_cluster else bug)

    step(0.8, "Validating")
    inside, outside = detector.static_scores(static, bug)
    typical = typical_bug_image(frames, visible, bug)
    fields = size_field_boxes({k: v for k, v in roles.items() if k in field_names}, bug, fw, fh, typical)
    cal = Calibration(
        bug=[round(v, 5) for v in norm_box(bug)],
        fields={k: [round(x, 5) for x in v] for k, v in fields.items()},
        crop=crop_to_norm(compute_crop(bug, fw, fh), fw, fh),
        frame_width=fw, frame_height=fh, source=source,
        teams=teams, broadcaster=broadcaster, warnings=warnings,
        checks={"static_inside": round(inside, 3), "static_outside": round(outside, 3)},
    )
    built = reference_from_frames(frames, visible, tuple(cal.bug), {k: tuple(v) for k, v in cal.fields.items()})
    ref, mask = built if built else (None, None)
    moved = tune_fields(cal, adapter, frames, engine, ref, mask)
    if moved:
        cal.warnings.extend(moved)
        built = reference_from_frames(frames, visible, tuple(cal.bug), {k: tuple(v) for k, v in cal.fields.items()})
        ref, mask = built if built else (None, None)
    validate(cal, adapter, frames, times, files, engine, ref, mask)
    check_teams(cal, expected_teams)
    step(1.0, "Calibration ready")
    return cal, ref, mask


def check_teams(cal: Calibration, expected_teams: dict[str, str] | None) -> None:
    """A wrong game or year is cheapest to catch here: the bug's team labels against the
    picked game's codes."""
    if not expected_teams or not (expected_teams.get("away") and expected_teams.get("home")):
        return
    labels = [v for v in cal.teams.values() if v]
    codes = [expected_teams["away"], expected_teams["home"]]
    if labels and not any(same_team_code(label, code) for label in labels for code in codes):
        cal.warnings.append(
            f"The bug reads {' / '.join(labels)}, but the game picked is {codes[0]} at {codes[1]}. "
            "Check the date and the game in setup."
        )


# (top, bottom, left, right) as fractions of the box's own height and width
TUNE_SHRINKS: tuple[tuple[float, float, float, float], ...] = (
    (0.08, 0.08, 0.0, 0.0), (0.16, 0.16, 0.0, 0.0), (0.25, 0.25, 0.0, 0.0),
    (0.0, 0.0, 0.08, 0.08), (0.0, 0.0, 0.16, 0.16),
    (0.0, 0.0, 0.0, 0.16), (0.0, 0.0, 0.0, 0.3), (0.0, 0.0, 0.16, 0.0), (0.0, 0.0, 0.3, 0.0),
    (0.08, 0.08, 0.08, 0.08), (0.16, 0.16, 0.16, 0.16),
)


def tune_fields(
    cal: Calibration,
    adapter: SportAdapter,
    frames: list[np.ndarray | None],
    engine: OcrEngine,
    reference: np.ndarray | None,
    mask: np.ndarray | None,
) -> list[str]:
    """Pull in boxes that read poorly until they read, and say which ones moved.

    A detected box that reaches a hair into a border line, a logo or the next field reads
    as nothing, or as an extra digit. Each field that fails on the sampled frames is read
    again with its box pulled in a little at the top and bottom, at the sides, or both;
    the smallest change that reads best wins. Scores must also never go down across the
    frames, which are in time order, so a box that clips "129" to "29" does not win.
    """
    shaped = [f for f in frames if f is not None]
    if not shaped or not cal.fields:
        return []
    fh, fw = shaped[0].shape[:2]
    full = make_reader(cal, adapter, fw, fh, engine, reference, mask)
    x, y, w, h = full.roi
    crops = [f[y : y + h, x : x + w] for f in shaped]
    if reference is not None:
        crops = [c for c in crops if full.similarity(c) >= VISIBLE_THRESHOLD]
    if len(crops) < 2:
        return []
    specs = {s.name: s for s in adapter.bug_fields}

    def score(name: str, box: Box) -> tuple[int, int]:
        """(clean reads, -order violations) for one field over the frames that show the bug."""
        reader = BugReader(tuple(cal.bug), {name: box}, adapter.bug_fields, fw, fh, None)
        if name not in reader.field_px:
            return (-1, 0)
        images = [prepare_field(reader.field_crop(c, name)) for c in crops]
        values = []
        for text, _conf in engine.recognize(images, [CHARSETS[specs[name].charset]] * len(images)):
            value = parse_field(name, text.strip())
            if value is not None:
                values.append(value)
        drops = sum(1 for a, b in zip(values, values[1:], strict=False) if b < a) if name.endswith("_score") else 0
        return len(values), -drops

    notes: list[str] = []
    for name in [s.name for s in adapter.bug_fields if s.name in cal.fields]:
        box = tuple(cal.fields[name])
        best_box, best = box, score(name, box)
        if best[0] >= 0.8 * len(crops) and best[1] == 0:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        for top, bottom, left, right in TUNE_SHRINKS:
            cand = (box[0] + left * bw, box[1] + top * bh, box[2] - right * bw, box[3] - bottom * bh)
            got = score(name, cand)
            if got > best:  # more clean reads first, then fewer scores going backwards
                best_box, best = cand, got
        if best_box != box:
            cal.fields[name] = [round(v, 5) for v in best_box]
            notes.append(f"The {name.replace('_', ' ')} box was pulled in so it reads cleanly.")
    return notes


def recalibrate_manual(
    cal: Calibration,
    probe: Probe,
    job_dir: Path,
    adapter: SportAdapter,
    engine: OcrEngine,
) -> tuple[Calibration, np.ndarray | None, np.ndarray | None]:
    """Re-validate after the user edited boxes: rebuild the reference from the saved frames."""
    calib_dir = job_dir / "calib"
    frames: list[np.ndarray | None] = []
    times, files = [], []
    for entry in cal.frames:
        path = calib_dir / entry["file"]
        frames.append(cv2.imread(str(path)) if path.exists() else None)
        times.append(entry["time"])
        files.append(entry["file"])
    fw, fh = probe.display_width, probe.height
    bug = tuple(cal.bug)
    fields = {k: tuple(v) for k, v in cal.fields.items()}
    # which frames show the bug: any frame where the clock or both scores read
    probe_reader = make_reader(cal, adapter, fw, fh, engine)
    visible = []
    for i, frame in enumerate(frames):
        if frame is None:
            continue
        reads = read_frame(probe_reader, frame, check_visible=False)["reads"]
        clock_ok = reads.get("clock", {}).get("ok")
        scores_ok = reads.get("away_score", {}).get("ok") and reads.get("home_score", {}).get("ok")
        if clock_ok or scores_ok:
            visible.append(i)
    built = reference_from_frames(frames, visible, bug, fields) if visible else None
    ref, mask = built if built else (None, None)
    cal.warnings = []
    cal.checks = {}
    validate(cal, adapter, frames, times, files, engine, ref, mask)
    return cal, ref, mask
