"""Find the score bug with Claude vision.

Two passes, both returning JSON through structured outputs:

1. Full frames: is a score bug on screen, where is it, who is playing, which network.
2. A zoomed crop of the bug: where each field sits inside it.

Vision models place boxes approximately, so the caller snaps these to locally detected
text and to the static region before using them.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..ai.claude import ClaudeClient, ClaudeError, ClaudeUnavailable, encode_image
from ..sports.base import FieldSpec
from .geometry import Box, expand, median_box, norm_box, to_px

log = logging.getLogger(__name__)

_BOX_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "integer"} for k in ("x0", "y0", "x1", "y1")},
    "required": ["x0", "y0", "x1", "y1"],
    "additionalProperties": False,
}
_NULLABLE_BOX = {"anyOf": [_BOX_SCHEMA, {"type": "null"}]}

FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "score_bug_visible": {"type": "boolean"},
        "bug_box": _NULLABLE_BOX,
        "away_team": {"type": "string"},
        "home_team": {"type": "string"},
        "broadcaster": {"type": "string"},
    },
    "required": ["score_bug_visible", "bug_box", "away_team", "home_team", "broadcaster"],
    "additionalProperties": False,
}

FRAME_SYSTEM = (
    "You analyze single frames from a sports TV broadcast to locate the score bug: the "
    "persistent on-screen scoreboard graphic showing both teams, their scores, the period "
    "and the game clock. Coordinates you return are integers normalized to 0-1000 on both "
    "axes, measured from the top-left of the image: x across the width, y down the height. "
    "A frame from a commercial, a studio segment, or a full-screen graphic has no score bug. "
    "Lower-third stat banners, tickers of other games, and sponsor logos are not the score bug."
)

FRAME_PROMPT = (
    "Sport: {sport}. Find the score bug in this frame.\n"
    "- score_bug_visible: true only if the live game's score bug is on screen.\n"
    "- bug_box: a tight box around the whole bug graphic (its background panel edge to edge, "
    "including team blocks, scores, period, clock{extra}), or null if it is not visible.\n"
    "- away_team / home_team: the abbreviations exactly as printed in the bug (the away team "
    "is usually listed first: left, or top). Empty strings if not visible.\n"
    "- broadcaster: the network if you can tell from on-screen branding (e.g. ESPN, ABC, TNT, "
    "NBC, Prime Video), else an empty string."
)

FIELD_SYSTEM = (
    "You are shown a zoomed crop of a sports broadcast score bug. Locate each requested field "
    "inside the crop. Coordinates are integers normalized to 0-1000 on both axes of this crop, "
    "from its top-left corner. Each box should cover the full area where that field's text "
    "appears, with a little margin, and must not include neighbouring fields. Return null for "
    "a field that this bug does not show."
)


@dataclass
class ClaudeDetection:
    visible: list[int]
    bug: Box
    roles: dict[str, Box | None]
    teams: dict[str, str]
    broadcaster: str
    notes: list[str] = field(default_factory=list)


def _box_from(d: dict | None) -> Box | None:
    if not d:
        return None
    box = norm_box((d["x0"] / 1000, d["y0"] / 1000, d["x1"] / 1000, d["y1"] / 1000))
    if box[2] - box[0] < 0.01 or box[3] - box[1] < 0.005:
        return None
    return box


def field_schema(specs: tuple[FieldSpec, ...]) -> dict:
    return {
        "type": "object",
        "properties": {s.name: _NULLABLE_BOX for s in specs},
        "required": [s.name for s in specs],
        "additionalProperties": False,
    }


def field_prompt(specs: tuple[FieldSpec, ...], sport: str) -> str:
    lines = [f"Sport: {sport}. Give the box of each field in this score bug crop:"]
    for s in specs:
        lines.append(f"- {s.name}: {s.description or s.name.replace('_', ' ')}")
    return "\n".join(lines)


def detect_with_claude(
    client: ClaudeClient,
    frames: list[np.ndarray | None],
    specs: tuple[FieldSpec, ...],
    sport_name: str,
    prefer: list[int] | None = None,
    max_frames: int = 12,
) -> ClaudeDetection | None:
    """Run both passes. Returns None when no frame shows a bug. Raises ClaudeError on API trouble."""
    if not client.available:
        raise ClaudeUnavailable(client.unavailable_reason or "Claude is not available")

    order = [i for i in (prefer or []) if frames[i] is not None]
    order += [i for i in range(len(frames)) if frames[i] is not None and i not in order]
    order = order[:max_frames]
    extra = ", shot clock" if any(s.name == "shot_clock" for s in specs) else ""
    prompt = FRAME_PROMPT.format(sport=sport_name, extra=extra)

    def ask_frame(i: int) -> tuple[int, dict | None]:
        try:
            return i, client.json(
                purpose="calibration:locate",
                system=FRAME_SYSTEM,
                content=[encode_image(frames[i], max_side=1568), {"type": "text", "text": prompt}],
                schema=FRAME_SCHEMA,
                effort="medium",
                estimate_usd=0.01,
            )
        except ClaudeUnavailable:
            raise
        except ClaudeError as exc:
            log.warning("Claude calibration call failed for frame %d: %s", i, exc)
            return i, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(ask_frame, order))

    boxes: list[Box] = []
    visible: list[int] = []
    aways: list[str] = []
    homes: list[str] = []
    nets: list[str] = []
    for i, ans in answers:
        if not ans or not ans.get("score_bug_visible"):
            continue
        box = _box_from(ans.get("bug_box"))
        if box is None:
            continue
        boxes.append(box)
        visible.append(i)
        if ans.get("away_team"):
            aways.append(ans["away_team"].strip().upper())
        if ans.get("home_team"):
            homes.append(ans["home_team"].strip().upper())
        if ans.get("broadcaster"):
            nets.append(ans["broadcaster"].strip())
    if not boxes:
        return None

    bug = median_box(boxes)
    # drop frames whose box disagrees with the consensus (a different graphic)
    keep = [k for k, b in enumerate(boxes) if abs(b[1] - bug[1]) < 0.06 and abs(b[3] - bug[3]) < 0.06]
    if keep and len(keep) < len(boxes):
        bug = median_box([boxes[k] for k in keep])
        visible = [visible[k] for k in keep]

    def most_common(values: list[str]) -> str:
        return max(set(values), key=values.count) if values else ""

    # pass 2: field boxes inside a zoomed crop of the bug
    roles_seen: dict[str, list[Box]] = {s.name: [] for s in specs}
    schema = field_schema(specs)
    fprompt = field_prompt(specs, sport_name)

    def ask_fields(i: int) -> dict | None:
        frame = frames[i]
        h, w = frame.shape[:2]
        region = expand(bug, 0.015, (bug[3] - bug[1]) * 0.35)
        rx0, ry0, rx1, ry1 = to_px(region, w, h)
        crop = frame[ry0:ry1, rx0:rx1]
        scale = min(4.0, 1400 / max(1, crop.shape[1]))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            ans = client.json(
                purpose="calibration:fields",
                system=FIELD_SYSTEM,
                content=[encode_image(crop, max_side=1568, quality=92), {"type": "text", "text": fprompt}],
                schema=schema,
                effort="medium",
                estimate_usd=0.01,
            )
        except ClaudeUnavailable:
            raise
        except ClaudeError as exc:
            log.warning("Claude field call failed for frame %d: %s", i, exc)
            return None
        out = {}
        for name, raw in ans.items():
            b = _box_from(raw)
            if b is None:
                continue
            # crop-normalized -> frame-normalized
            out[name] = norm_box((
                region[0] + b[0] * (region[2] - region[0]),
                region[1] + b[1] * (region[3] - region[1]),
                region[0] + b[2] * (region[2] - region[0]),
                region[1] + b[3] * (region[3] - region[1]),
            ))
        return out

    with ThreadPoolExecutor(max_workers=3) as pool:
        field_answers = list(pool.map(ask_fields, visible[:3]))
    for ans in field_answers:
        for name, box in (ans or {}).items():
            if name in roles_seen:
                roles_seen[name].append(box)
    roles: dict[str, Box | None] = {
        name: (median_box(found) if found else None) for name, found in roles_seen.items()
    }
    notes = []
    if not any(roles.values()):
        notes.append("Claude located the bug but not its fields")
    return ClaudeDetection(
        visible=sorted(visible),
        bug=bug,
        roles=roles,
        teams={"away": most_common(aways), "home": most_common(homes)},
        broadcaster=most_common(nets),
        notes=notes,
    )
