"""Boxes and the output crop rule.

All boxes that get stored are normalized to the display frame: (x0, y0, x1, y1) in 0..1.
Normalized boxes survive a change of resolution, so one broadcaster template serves both
the 720p and the 1080p feed of the same network.
"""

from __future__ import annotations

from statistics import median

Box = tuple[float, float, float, float]

# PRD "Output format": the bug fills about 78% of the crop width.
BUG_FILL = 0.78
# The reference crop is roughly 1.18:1. Used when the bug is too small to size the crop from.
REFERENCE_ASPECT = 1.18
MIN_CROP_ASPECT = 1.0
OUTPUT_W, OUTPUT_H = 1080, 1920


def clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def norm_box(box) -> Box:
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((clamp01(x0), clamp01(x1)))
    y0, y1 = sorted((clamp01(y0), clamp01(y1)))
    return (x0, y0, x1, y1)


def to_px(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Normalized box to integer pixel corners (x0, y0, x1, y1), clamped to the frame."""
    x0 = int(round(box[0] * width))
    y0 = int(round(box[1] * height))
    x1 = int(round(box[2] * width))
    y1 = int(round(box[3] * height))
    x0, y0 = max(0, min(width - 1, x0)), max(0, min(height - 1, y0))
    x1, y1 = max(x0 + 1, min(width, x1)), max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def from_px(box, width: int, height: int) -> Box:
    x0, y0, x1, y1 = box
    return norm_box((x0 / width, y0 / height, x1 / width, y1 / height))


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: Box, b: Box) -> Box | None:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    ia = box_area(inter)
    return ia / (box_area(a) + box_area(b) - ia)


def union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def median_box(boxes: list[Box]) -> Box:
    return tuple(median(b[i] for b in boxes) for i in range(4))  # type: ignore[return-value]


def expand(box: Box, dx: float, dy: float) -> Box:
    return norm_box((box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy))


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def compute_crop(bug: Box, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Source crop (x, y, w, h) in pixels for the 9:16 export.

    PRD rule: full source height; width = bug width / 0.78; centred on the bug (not the
    frame); clamped to the frame edges.

    One guard on top of the rule: a small corner bug (common in older NFL/NHL/MLB
    graphics) would produce a sliver narrower than it is tall. In that case the crop
    falls back to the reference 1.18:1 shape, centred on the frame and shifted only as
    far as needed to keep the whole bug inside it.
    """
    bug_w = (bug[2] - bug[0]) * frame_w
    bug_cx = (bug[0] + bug[2]) / 2 * frame_w
    crop_h = frame_h - frame_h % 2
    crop_w = min(frame_w, bug_w / BUG_FILL)

    if crop_w >= MIN_CROP_ASPECT * crop_h:
        crop_w = min(_even(crop_w), frame_w - frame_w % 2)
        x = int(round(bug_cx - crop_w / 2))
    else:
        crop_w = min(_even(REFERENCE_ASPECT * crop_h), frame_w - frame_w % 2)
        x = int(round(frame_w / 2 - crop_w / 2))
        bug_x0, bug_x1 = bug[0] * frame_w, bug[2] * frame_w
        if bug_x0 < x:
            x = int(bug_x0)
        if bug_x1 > x + crop_w:
            x = int(round(bug_x1 - crop_w))
    x = max(0, min(frame_w - crop_w, x))
    return x, 0, crop_w, crop_h


def crop_to_norm(crop: tuple[int, int, int, int], frame_w: int, frame_h: int) -> list[float]:
    x, y, w, h = crop
    return [round(x / frame_w, 6), round(y / frame_h, 6), round(w / frame_w, 6), round(h / frame_h, 6)]


def crop_from_norm(crop, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Stored normalized crop back to even pixel values inside the frame."""
    x, y, w, h = crop
    pw = min(_even(w * frame_w), frame_w - frame_w % 2)
    ph = min(_even(h * frame_h), frame_h - frame_h % 2)
    px = max(0, min(frame_w - pw, int(round(x * frame_w))))
    py = max(0, min(frame_h - ph, int(round(y * frame_h))))
    return px, py, pw, ph


def output_placement(crop_w: int, crop_h: int) -> tuple[int, int, int]:
    """How the crop sits on the 1080x1920 canvas: (scaled_w, scaled_h, y_offset)."""
    scaled_h = _even(OUTPUT_W * crop_h / crop_w)
    scaled_h = min(scaled_h, OUTPUT_H)
    y = (OUTPUT_H - scaled_h) // 2
    y -= y % 2
    return OUTPUT_W, scaled_h, y
