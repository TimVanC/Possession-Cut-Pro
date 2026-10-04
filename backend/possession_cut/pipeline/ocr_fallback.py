"""Batched Claude vision re-read for bug samples the local OCR could not settle.

Only samples that matter are sent: ones sitting between the last read of an old score and
the first read of a new one (they decide when a basket is stamped), and stretches where
the score read came back stable but inconsistent with the game. Crops are stacked into
contact sheets, several to an image, and the spend is capped by the per-job budget.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np

from ..ai.claude import BudgetExceeded, ClaudeClient, ClaudeError, ClaudeUnavailable, encode_image
from ..sports.base import ScoreChange, SportAdapter
from .calibration import parse_field
from .sampler import RawSamples
from .timeline import Timeline

log = logging.getLogger(__name__)

ROWS_PER_SHEET = 8
MAX_SAMPLES = 480
EST_USD_PER_SHEET = 0.012
SCORE_FIELDS = ("away_score", "home_score")

SYSTEM = (
    "You read sports broadcast score bugs. The image is a stack of numbered rows; each row is "
    "the same score bug cropped from a different video frame. For every row, read the away "
    "(first-listed) team's score, the home (second-listed) team's score, the period and the game "
    "clock exactly as shown. If a value is covered, mid-animation or unreadable in that row, "
    "return null for it. Do not guess from neighbouring rows."
)

SHEET_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer"},
                    "away_score": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "home_score": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "period": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "clock": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["row", "away_score", "home_score", "period", "clock"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}


def pick_samples(raw: RawSamples, tl: Timeline, events: list[ScoreChange]) -> list[int]:
    """Indexes of samples worth a second opinion, most useful first."""
    have = raw.crops
    if not have:
        return []
    picked: list[int] = []
    seen: set[int] = set()

    def add(i: int) -> None:
        if i in have and i not in seen:
            seen.add(i)
            picked.append(i)

    # 1. between the last read of the old score and the first read of the new one
    for ev in events:
        i0, i1 = tl.index_at(ev.t_prev), tl.index_at(ev.t)
        if i1 - i0 >= 3:  # two or more samples with no settled score: the stamp is uncertain
            for i in range(i0 + 1, i1):
                add(i)
    # 2. everything else that was kept, in time order
    for i in sorted(have):
        add(i)
    return picked[:MAX_SAMPLES]


def contact_sheet(crops: list[np.ndarray], width: int = 1100) -> np.ndarray:
    """Stack bug crops into one image with a row number at the left of each."""
    rows = []
    for n, crop in enumerate(crops, start=1):
        h, w = crop.shape[:2]
        scale = (width - 90) / w
        img = cv2.resize(crop, (width - 90, max(24, int(round(h * scale)))), interpolation=cv2.INTER_CUBIC)
        label = np.full((img.shape[0], 90, 3), 255, dtype=np.uint8)
        cv2.putText(label, str(n), (18, int(img.shape[0] * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    min(1.6, img.shape[0] / 40), (0, 0, 0), 3, cv2.LINE_AA)
        rows.append(np.hstack([label, img]))
        rows.append(np.full((10, width, 3), 255, dtype=np.uint8))
    return np.vstack(rows[:-1])


def reread_with_claude(
    raw: RawSamples,
    tl: Timeline,
    events: list[ScoreChange],
    adapter: SportAdapter,
    claude: ClaudeClient | None,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Send the uncertain samples to Claude and write its reads back into ``raw``.

    Returns a small report. Does nothing (and says why) when Claude is unavailable, the
    budget is spent, or the local OCR was already clean.
    """
    report = {"candidates": 0, "sent": 0, "updated": 0, "sheets": 0, "skipped": ""}
    if claude is None or not claude.available:
        report["skipped"] = (claude.unavailable_reason if claude else None) or "Claude is not configured"
        return report
    visible = int(raw.visible.sum())
    if visible and len(raw.crops) / visible > 0.25:
        report["skipped"] = (
            "more than a quarter of samples read poorly; that is a calibration problem, "
            "not something to spend the Claude budget on"
        )
        return report
    todo = pick_samples(raw, tl, events)
    report["candidates"] = len(todo)
    if not todo:
        return report

    sheets = [todo[i : i + ROWS_PER_SHEET] for i in range(0, len(todo), ROWS_PER_SHEET)]
    for n, sheet in enumerate(sheets):
        if claude.remaining_usd < EST_USD_PER_SHEET:
            report["skipped"] = f"Claude budget reached after {report['sheets']} of {len(sheets)} sheets"
            break
        image = contact_sheet([raw.crops[i] for i in sheet])
        try:
            answer = claude.json(
                purpose="ocr_fallback",
                system=SYSTEM,
                content=[
                    encode_image(image, max_side=1568, quality=92),
                    {"type": "text", "text": f"Sport: {adapter.name}. Read rows 1 to {len(sheet)}."},
                ],
                schema=SHEET_SCHEMA,
                effort="low",
                estimate_usd=EST_USD_PER_SHEET,
            )
        except (BudgetExceeded, ClaudeUnavailable) as exc:
            report["skipped"] = str(exc)
            break
        except ClaudeError as exc:
            log.warning("OCR fallback sheet %d failed: %s", n, exc)
            continue
        report["sheets"] += 1
        report["sent"] += len(sheet)
        for row in answer.get("rows", []):
            k = int(row.get("row", 0)) - 1
            if not 0 <= k < len(sheet):
                continue
            i = sheet[k]
            for name in ("away_score", "home_score", "period", "clock"):
                value = row.get(name)
                if value is None or name not in raw.texts:
                    continue
                text = str(value).strip()
                if parse_field(name, text) is None:
                    continue
                if raw.texts[name][i] != text or raw.confs[name][i] < 0.9:
                    raw.texts[name][i] = text
                    raw.confs[name][i] = 0.95
                    report["updated"] += 1
        if progress:
            progress((n + 1) / len(sheets), f"Claude re-read {report['sent']} uncertain samples")
    return report
