"""Find the score bug with on-device text detection and pixel statistics. No network.

Three passes over the sampled calibration frames:

1. Coarse: read all text in each full frame and look for a game clock ("2:27", "40.2")
   that keeps turning up in the same place. That row is the bug; frames showing it are
   the ones where the bug is on screen.
2. Static region: across those frames the bug is the part of the picture that does not
   change. The connected static region around the clock gives the bug's bounding box.
3. Fine: zoom into the bug, read it again so each field comes back as its own text box,
   and work out which box is which (labels, scores, period, clock, shot clock).

This is the fallback when Claude is unavailable, and the source of pixel-accurate boxes
when Claude supplies the semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .geometry import Box, box_area, expand, from_px, intersection, iou, to_px, union
from .ocr import OcrEngine, TextBox, parse_period

CLOCK_RE = re.compile(r"(?<![\d:.])(\d{1,2}:\d{2}|\d{1,2}\.\d)(?![\d:.])")
# Anywhere inside a longer string: "2ND3:0020" holds the clock 3:00. Minutes are the one
# or two digits before the colon, seconds the two after it.
CLOCK_INSIDE = re.compile(r"(\d{1,2}):([0-5]\d)|(?<!\d)(\d{1,2})\.(\d)(?!\d)")
# A whole text line that is really several fields: [period][clock][shot clock]
MERGED_LINE = re.compile(
    r"^(?P<before>.*?)\s*(?P<clock>\d{1,2}:[0-5]\d|(?<!\d)\d{1,2}\.\d)\s*(?P<after>\d{1,2})?$"
)
_OCR_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})


def _fix_digits(text: str) -> str:
    """'1O:47' -> '10:47'. Only applied next to a colon or period, where digits are expected."""
    return re.sub(r"[OolI](?=[:.\d])|(?<=[:.\d])[OolI]", lambda m: m.group(0).translate(_OCR_DIGIT_FIXES), text)
ANALYSIS_HEIGHT = 720
STATIC_STD = 11.0


@dataclass
class Obs:
    frame: int
    box: Box  # normalized to the frame
    text: str
    conf: float
    split: bool = False  # carved out of a longer line, so its box is an estimate


@dataclass
class Cluster:
    obs: list[Obs] = field(default_factory=list)

    @property
    def frames(self) -> set[int]:
        return {o.frame for o in self.obs}

    @property
    def box(self) -> Box:
        xs0 = sorted(o.box[0] for o in self.obs)
        ys0 = sorted(o.box[1] for o in self.obs)
        xs1 = sorted(o.box[2] for o in self.obs)
        ys1 = sorted(o.box[3] for o in self.obs)
        m = len(self.obs) // 2
        return (xs0[m], ys0[m], xs1[m], ys1[m])

    @property
    def union(self) -> Box:
        # boxes carved out of a merged line are estimates; use them only if nothing better
        exact = [o.box for o in self.obs if not o.split]
        return union(exact or [o.box for o in self.obs])

    @property
    def texts(self) -> list[str]:
        return [o.text for o in self.obs]

    def frac(self, pred) -> float:
        return sum(1 for t in self.texts if pred(t)) / max(1, len(self.obs))


@dataclass
class LocalDetection:
    visible: list[int]  # indexes of frames where the bug was seen
    bug: Box
    roles: dict[str, Box]  # field name -> tight box around the text (not yet padded)
    teams: dict[str, str]
    clusters: list[Cluster]
    static_inside: float = 0.0
    static_outside: float = 0.0
    notes: list[str] = field(default_factory=list)


def _is_clock(text: str) -> bool:
    return bool(CLOCK_RE.fullmatch(text.strip()))


def _is_number(text: str) -> bool:
    t = text.strip()
    return t.isdigit() and len(t) <= 3


def _is_label(text: str) -> bool:
    t = text.strip()
    return 2 <= len(t) <= 5 and t.isalpha() and t.upper() == t and parse_period(t) is None


def _is_period(text: str) -> bool:
    t = text.strip()
    return parse_period(t) is not None and not t.isdigit()


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _char_width(ch: str) -> float:
    return 0.4 if ch in ":.,'" else (0.6 if ch == " " else 1.0)


def _split_merged(obs: Obs) -> list[Obs]:
    """'3rd 2:27', '2:27 22' or '2ND3:0020' read as one line -> one observation per field.

    The pieces' boxes are estimated from character widths, so they are marked as splits.
    """
    text = _fix_digits(obs.text.strip())
    m = MERGED_LINE.match(text)
    if not m:
        return [replace(obs, text=text)] if text != obs.text else [obs]
    before, clock, after = (m.group("before") or "").strip(), m.group("clock"), m.group("after") or ""
    if not before and not after:
        return [replace(obs, text=clock)]
    widths = [_char_width(c) for c in text]
    total = sum(widths) or 1.0
    x0, y0, x1, y1 = obs.box

    def span(start: int, end: int) -> Box:
        a = sum(widths[:start]) / total
        b = sum(widths[:end]) / total
        return (x0 + (x1 - x0) * a, y0, x0 + (x1 - x0) * b, y1)

    out: list[Obs] = []
    c0 = text.index(clock, len(m.group("before") or ""))
    if before:
        out.append(Obs(obs.frame, span(0, len(m.group("before").rstrip())), before, obs.conf, split=True))
    out.append(Obs(obs.frame, span(c0, c0 + len(clock)), clock, obs.conf, split=True))
    if after:
        a0 = text.rindex(after)
        out.append(Obs(obs.frame, span(a0, len(text)), after, obs.conf, split=True))
    return out


def _cluster(observations: list[Obs], min_iou: float = 0.25) -> list[Cluster]:
    """Group observations of the same field across frames.

    Boxes straight from the detector go first and define the clusters. Pieces carved out
    of a fused line (whose boxes are estimates) then join whichever cluster they overlap,
    and only start clusters of their own when nothing exact covers that spot.
    """
    clusters: list[Cluster] = []

    def place(o: Obs) -> None:
        best, best_score = None, 0.0
        for c in clusters:
            cb = c.box
            score = iou(cb, o.box)
            cx, cy = _center(o.box)
            if cb[0] <= cx <= cb[2] and cb[1] <= cy <= cb[3]:
                score = max(score, 0.5)
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score >= min_iou:
            best.obs.append(o)
        else:
            clusters.append(Cluster([o]))

    for o in observations:
        if not o.split:
            place(o)
    for o in observations:
        if o.split:
            place(o)
    return clusters


def _resize_for_analysis(frame: np.ndarray) -> np.ndarray:
    h = frame.shape[0]
    if h <= ANALYSIS_HEIGHT:
        return frame
    scale = ANALYSIS_HEIGHT / h
    return cv2.resize(frame, (int(round(frame.shape[1] * scale)), ANALYSIS_HEIGHT), interpolation=cv2.INTER_AREA)


class LocalDetector:
    def __init__(self, frames: list[np.ndarray | None], engine: OcrEngine) -> None:
        self.engine = engine
        self.frames = [None if f is None else _resize_for_analysis(f) for f in frames]
        self._coarse: list[list[TextBox]] | None = None

    # -- pass 1 ------------------------------------------------------------
    def coarse_text(self) -> list[list[TextBox]]:
        if self._coarse is None:
            self._coarse = [[] if f is None else self.engine.read_text(f) for f in self.frames]
        return self._coarse

    def find_clock_row(self) -> tuple[Cluster | None, list[int]]:
        """The clock-looking text that recurs in one place, and the frames that show it."""
        observations: list[Obs] = []
        for i, boxes in enumerate(self.coarse_text()):
            frame = self.frames[i]
            if frame is None:
                continue
            h, w = frame.shape[:2]
            for tb in boxes:
                text = _fix_digits(tb.text)
                m = CLOCK_INSIDE.search(text)
                if not m:
                    continue
                # keep just the clock's share of a merged line
                x0, y0, x1, y1 = tb.box
                n = max(1, len(text))
                cx0 = x0 + (x1 - x0) * (m.start() / n)
                cx1 = x0 + (x1 - x0) * (m.end() / n)
                observations.append(Obs(i, from_px((cx0, y0, cx1, y1), w, h), m.group(0), tb.conf, split=n > m.end() - m.start()))
        clusters = _cluster(observations, min_iou=0.2)
        clusters = [c for c in clusters if len(c.frames) >= 2]
        if not clusters:
            return None, []
        best = max(clusters, key=lambda c: (len(c.frames), -c.box[1]))
        return best, sorted(best.frames)

    # -- pass 2 ------------------------------------------------------------
    def static_region(self, visible: list[int], seed: Box) -> tuple[Box | None, np.ndarray | None]:
        """Bounding box of the unchanging region around ``seed`` across the visible frames."""
        frames = [self.frames[i] for i in visible if self.frames[i] is not None]
        if len(frames) < 3:
            return None, None
        h, w = frames[0].shape[:2]
        stack = np.stack([f.astype(np.float32) for f in frames])
        std = stack.std(axis=0).max(axis=2)
        static = (std < STATIC_STD).astype(np.uint8)

        sx0, sy0, sx1, sy1 = to_px(seed, w, h)
        text_h = max(8, sy1 - sy0)
        # only look in a band around the clock row; bugs are at most a few text lines tall
        band0, band1 = max(0, sy0 - 3 * text_h), min(h, sy1 + 3 * text_h)
        band = np.zeros_like(static)
        band[band0:band1] = static[band0:band1]
        # close gaps left by the digits that change between frames
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(text_h * 3) | 1, int(text_h * 1.2) | 1))
        closed = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if n <= 1:
            return None, static
        seed_labels = labels[sy0:sy1, sx0:sx1]
        counts = np.bincount(seed_labels.ravel(), minlength=n)
        counts[0] = 0
        if counts.max() == 0:
            return None, static
        comp = int(counts.argmax())
        x, y, cw, ch = (int(stats[comp, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        x0, y0, x1, y1 = x, y, x + cw, y + ch
        # trim ragged edges: every kept row/column must be mostly static
        comp_mask = (labels == comp).astype(np.uint8)
        while y1 - y0 > text_h and comp_mask[y0, x0:x1].mean() < 0.6:
            y0 += 1
        while y1 - y0 > text_h and comp_mask[y1 - 1, x0:x1].mean() < 0.6:
            y1 -= 1
        while x1 - x0 > text_h and comp_mask[y0:y1, x0].mean() < 0.6:
            x0 += 1
        while x1 - x0 > text_h and comp_mask[y0:y1, x1 - 1].mean() < 0.6:
            x1 -= 1
        # The outer rim of a real bug (bevel, glow, a slightly see-through edge) varies a
        # little more than its body. Grow each side while it is still clearly steadier than
        # live picture, so the box takes in the whole graphic and not just its core.
        # "Steadier than live picture" is judged against this video: halfway between how much
        # the bug's body varies and how much the picture a little way off to the sides does.
        inner = float(np.median(std[y0:y1, x0:x1]))
        band = max(8, int(w * 0.05))
        beyond = np.concatenate([
            std[y0:y1, max(0, x0 - 2 * band) : max(0, x0 - band)].ravel(),
            std[y0:y1, min(w, x1 + band) : min(w, x1 + 2 * band)].ravel(),
        ])
        outer = float(np.median(beyond)) if beyond.size else inner + 40.0
        soft = inner + 0.45 * max(0.0, outer - inner)
        grow_x, grow_y = int(w * 0.03), int(text_h * 0.5)
        for _ in range(grow_x):
            if x0 > 0 and std[y0:y1, x0 - 1].mean() < soft:
                x0 -= 1
            else:
                break
        for _ in range(grow_x):
            if x1 < w and std[y0:y1, x1].mean() < soft:
                x1 += 1
            else:
                break
        for _ in range(grow_y):
            if y0 > 0 and std[y0 - 1, x0:x1].mean() < soft:
                y0 -= 1
            else:
                break
        for _ in range(grow_y):
            if y1 < h and std[y1, x0:x1].mean() < soft:
                y1 += 1
            else:
                break
        return from_px((x0, y0, x1, y1), w, h), static

    def static_scores(self, static: np.ndarray | None, bug: Box) -> tuple[float, float]:
        """Share of static pixels inside the bug and in a ring just outside it."""
        if static is None:
            return 0.0, 0.0
        h, w = static.shape
        x0, y0, x1, y1 = to_px(bug, w, h)
        inside = float(static[y0:y1, x0:x1].mean())
        pad = max(6, (y1 - y0) // 2)
        ox0, oy0, ox1, oy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        outer = static[oy0:oy1, ox0:ox1].astype(np.float32)
        total = outer.sum() - static[y0:y1, x0:x1].sum()
        area = outer.size - (y1 - y0) * (x1 - x0)
        return inside, float(total / area) if area > 0 else 0.0

    # -- pass 3 ------------------------------------------------------------
    def fine_text(
        self,
        visible: list[int],
        bug: Box,
        source_frames: list[np.ndarray | None] | None = None,
        text_height: float | None = None,
    ) -> list[Obs]:
        """Zoom into the bug on each visible frame and read it field by field.

        ``text_height`` is the height of the bug's text as a fraction of the frame (from
        the clock found in the coarse pass). The zoom is chosen from it, because how big
        the text is relative to the bug varies (one-row bugs, two-row bugs). Fields that
        fuse at one zoom often separate at another, so two zooms are read.
        """
        out: list[Obs] = []
        frames = source_frames or self.frames
        for i in visible:
            frame = frames[i]
            if frame is None:
                continue
            h, w = frame.shape[:2]
            bh = bug[3] - bug[1]
            region = expand(bug, 0.01, bh * 0.35)
            rx0, ry0, rx1, ry1 = to_px(region, w, h)
            crop = frame[ry0:ry1, rx0:rx1]
            text_px = max(8.0, (text_height or bh * 0.6) * h)
            for target in (46.0, 68.0):
                scale = float(np.clip(target / text_px, 1.0, 4.0))
                if crop.shape[1] * scale > 2300:
                    scale = 2300 / crop.shape[1]
                zoom = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                for tb in self.engine.read_text(zoom, strip=True):
                    x0, y0, x1, y1 = (v / scale for v in tb.box)
                    box = from_px((rx0 + x0, ry0 + y0, rx0 + x1, ry0 + y1), w, h)
                    cx, cy = _center(box)
                    if not (bug[0] <= cx <= bug[2] and bug[1] <= cy <= bug[3]):
                        continue
                    out.extend(_split_merged(Obs(i, box, tb.text, tb.conf)))
        return out

    def assign_roles(
        self,
        observations: list[Obs],
        field_names: list[str],
        expected_teams: dict[str, str] | None = None,
    ) -> tuple[dict[str, Box], dict[str, str], list[Cluster], list[str]]:
        """Decide which text cluster is which bug field."""
        notes: list[str] = []
        clusters = [c for c in _cluster(observations) if len(c.frames) >= 2 or len(observations) < 12]
        roles: dict[str, Cluster] = {}

        def pick(pred, exclude, min_frac=0.5):
            cands = [(c.frac(pred) * len(c.frames), c) for c in clusters if id(c) not in exclude and c.frac(pred) >= min_frac]
            return max(cands, key=lambda sc: sc[0])[1] if cands else None

        used: set[int] = set()
        clock = pick(_is_clock, used)
        if clock is not None:
            roles["clock"] = clock
            used.add(id(clock))
        else:
            notes.append("no clock found inside the bug")

        if "period" in field_names:
            period = pick(_is_period, used)
            if period is not None:
                roles["period"] = period
                used.add(id(period))

        def label_rank(c: Cluster) -> tuple:
            name = max(set(c.texts), key=c.texts.count).strip().upper()
            wanted = [v.upper() for v in (expected_teams or {}).values() if v]
            known = any(name == w or (len(name) >= 2 and (w.startswith(name) or name.startswith(w))) for w in wanted)
            same_row = False
            if clock is not None:
                cy = _center(c.box)[1]
                same_row = clock.box[1] - 0.25 * (clock.box[3] - clock.box[1]) <= cy <= clock.box[3] + 0.25 * (clock.box[3] - clock.box[1])
            return (known, same_row, len(c.frames))

        labels = sorted(
            (c for c in clusters if id(c) not in used and c.frac(_is_label) >= 0.6),
            key=label_rank, reverse=True,
        )[:2]
        numeric = [c for c in clusters if id(c) not in used and c.frac(_is_number) >= 0.7]

        # shot clock: the small number that hugs the game clock
        if "shot_clock" in field_names and clock is not None:
            cw = clock.box[2] - clock.box[0]
            ccx, ccy = _center(clock.box)

            def small(c: Cluster) -> bool:
                vals = [int(t) for t in c.texts if t.strip().isdigit()]
                return bool(vals) and max(vals) <= 35

            near = [
                c for c in numeric
                if small(c) and abs(_center(c.box)[0] - ccx) < 2.2 * cw and abs(_center(c.box)[1] - ccy) < 2.5 * (clock.box[3] - clock.box[1])
            ]
            # a score sits next to a team label, the shot clock next to the game clock
            label_boxes = [lab.box for lab in labels]

            def label_gap(c: Cluster) -> float:
                if not label_boxes:
                    return 1.0
                return min(abs(_center(c.box)[0] - _center(lb)[0]) for lb in label_boxes)

            near = [c for c in near if label_gap(c) > abs(_center(c.box)[0] - ccx)]
            if near:
                shot = min(near, key=lambda c: abs(_center(c.box)[0] - ccx))
                roles["shot_clock"] = shot
                used.add(id(shot))
                numeric = [c for c in numeric if id(c) != id(shot)]

        # reading order: left to right, or top to bottom when the teams are stacked
        def order_key(c: Cluster) -> tuple[float, float]:
            cx, cy = _center(c.box)
            return (cy, cx)

        stacked = len(labels) == 2 and abs(_center(labels[0].box)[1] - _center(labels[1].box)[1]) > 0.6 * (
            labels[0].box[3] - labels[0].box[1]
        )
        labels.sort(key=order_key if stacked else (lambda c: _center(c.box)[0]))

        teams: dict[str, str] = {}
        if len(labels) == 2:
            names = [max(set(c.texts), key=c.texts.count).strip().upper() for c in labels]
            if expected_teams:
                want_away = expected_teams.get("away", "").upper()
                want_home = expected_teams.get("home", "").upper()

                def matches(seen: str, want: str) -> bool:
                    return bool(want) and (seen == want or seen.startswith(want[:2]) or want.startswith(seen[:2]))

                if matches(names[0], want_home) and matches(names[1], want_away) and not matches(names[0], want_away):
                    labels.reverse()
                    names.reverse()
                    notes.append("home team is listed first in this bug")
            roles["away_label"], roles["home_label"] = labels[0], labels[1]
            teams = {"away": names[0], "home": names[1]}
            # each label's score is the closest number to it
            best_pair, best_cost = None, 1e9
            for a in numeric:
                for b in numeric:
                    if a is b:
                        continue
                    cost = _dist(a.box, labels[0].box) + _dist(b.box, labels[1].box)
                    if cost < best_cost:
                        best_pair, best_cost = (a, b), cost
            if best_pair:
                roles["away_score"], roles["home_score"] = best_pair
        else:
            notes.append("team labels not found; scores assigned by position")
            scores = sorted(numeric, key=lambda c: -len(c.frames))[:2]
            scores.sort(key=lambda c: _center(c.box)[0])
            if len(scores) == 2:
                roles["away_score"], roles["home_score"] = scores

        boxes = {name: c.union for name, c in roles.items() if name in field_names}
        return boxes, teams, clusters, notes

    # -- ink ---------------------------------------------------------------
    def ink_runs(self, visible: list[int], bug: Box, row: Box) -> list[tuple[float, float]]:
        """Where there is text along the bug's main row, as (x0, x1) frame fractions.

        Text detectors fuse neighbouring fields unpredictably ("7:5213"). The pixels do not:
        across the frames that show the bug, a column that never has ink is a gap between
        fields. A column has ink when something in the text rows differs sharply from that
        same column just above and below the text. Comparing within a column is what makes
        this blind to the bug's own artwork: a boundary between two coloured blocks runs
        through the text rows and their margins alike, so it does not count.
        """
        frames = [self.frames[i] for i in visible if self.frames[i] is not None]
        if not frames:
            return []
        h, w = frames[0].shape[:2]
        bx0, by0, bx1, by1 = to_px(bug, w, h)
        _, ry0, _, ry1 = to_px(row, w, h)
        text_h = max(6, ry1 - ry0)
        top = [y for y in range(ry0 - 2, ry0 + 1) if by0 <= y < by1]
        bottom = [y for y in range(ry1, ry1 + 3) if by0 <= y < by1]
        margin = top + bottom
        if not margin or ry1 - ry0 < 4:
            return []
        deviation = np.zeros(bx1 - bx0, dtype=np.float32)
        for frame in frames:
            gray = cv2.cvtColor(frame[:, bx0:bx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            background = np.median(gray[margin], axis=0)
            deviation = np.maximum(deviation, np.abs(gray[ry0 + 1 : ry1] - background[None, :]).max(axis=0))
        if deviation.max() < 40.0:
            return []
        ink = deviation > max(35.0, 0.4 * float(np.percentile(deviation, 98)))
        runs: list[list[int]] = []
        for x in np.flatnonzero(ink):
            if runs and x - runs[-1][1] <= 1:
                runs[-1][1] = int(x)
            else:
                runs.append([int(x), int(x)])
        # letters of one word sit closer together than fields do
        word_gap = max(3, int(round(text_h * 0.3)))
        words: list[list[int]] = []
        for r in runs:
            if words and r[0] - words[-1][1] <= word_gap:
                words[-1][1] = r[1]
            else:
                words.append(list(r))
        # a sliver on its own is artwork (the end of a rounded box), not a field
        words = [r for r in words if r[1] - r[0] + 1 >= max(4, int(text_h * 0.3))]
        return [((bx0 + a) / w, (bx0 + b + 1) / w) for a, b in words]

    @staticmethod
    def snap_to_ink(roles: dict[str, Box], runs: list[tuple[float, float]]) -> dict[str, Box]:
        """Set each field's left and right edge from the ink it actually covers."""
        out: dict[str, Box] = {}
        for name, (x0, y0, x1, y1) in roles.items():
            mine = [
                (a, b) for a, b in runs
                if min(b, x1) - max(a, x0) >= 0.5 * (b - a)  # most of that word lies in this box
            ]
            if mine:
                out[name] = (min(a for a, _ in mine), y0, max(b for _, b in mine), y1)
            else:
                out[name] = (x0, y0, x1, y1)
        return out

    # -- all together --------------------------------------------------------
    def detect(self, field_names: list[str], expected_teams: dict[str, str] | None = None) -> LocalDetection | None:
        clock_cluster, visible = self.find_clock_row()
        if clock_cluster is None:
            return None
        notes: list[str] = []
        bug, static = self.static_region(visible, clock_cluster.box)
        if bug is None:
            notes.append("static-region analysis was inconclusive; bug box estimated from its text")
            row = [
                from_px(tb.box, self.frames[i].shape[1], self.frames[i].shape[0])
                for i in visible for tb in self.coarse_text()[i]
            ]
            cy = _center(clock_cluster.box)[1]
            th = clock_cluster.box[3] - clock_cluster.box[1]
            row = [b for b in row if abs(_center(b)[1] - cy) < th]
            bug = expand(union(row), 0.01, th * 0.3) if row else expand(clock_cluster.box, 0.2, th)
        text_height = clock_cluster.box[3] - clock_cluster.box[1]
        observations = self.fine_text(visible, bug, text_height=text_height)
        roles, teams, clusters, role_notes = self.assign_roles(observations, field_names, expected_teams)
        text_boxes = dict(roles)
        roles = self.snap_to_ink(roles, self.ink_runs(visible, bug, clock_cluster.box))
        roles = separate_fields(roles, text_boxes)
        inside, outside = self.static_scores(static, bug)
        return LocalDetection(visible, bug, roles, teams, clusters, inside, outside, notes + role_notes)


def _dist(a: Box, b: Box) -> float:
    (ax, ay), (bx, by) = _center(a), _center(b)
    # vertical offsets count double: a score belongs to the label on its own row
    return abs(ax - bx) + 2.0 * abs(ay - by)


# -- turning tight text boxes into OCR regions -------------------------------------


def _free_run(grad: np.ndarray, x: int, direction: int, limit: int, thresh: float = 10.0) -> int:
    """How many columns can be added from ``x`` going ``direction`` before hitting artwork."""
    n = 0
    w = len(grad) + 1
    while n < limit:
        nx = x + direction
        if nx < 0 or nx >= w:
            break
        if grad[min(x, nx)] > thresh:
            break
        x = nx
        n += 1
    return n


def size_field_boxes(
    roles: dict[str, Box],
    bug: Box,
    frame_w: int,
    frame_h: int,
    bug_image: np.ndarray | None = None,
) -> dict[str, Box]:
    """Pad each field's text box into the region OCR will read.

    Scores get room for one more digit on each side (99 -> 100 must still fit). Padding
    stops at neighbouring fields, at the bug's edge, and at any artwork in the bug (a team
    logo next to a score would otherwise read as an extra digit). ``bug_image`` is a
    typical crop of the bug used to see that artwork.
    """
    out: dict[str, Box] = {}
    names = list(roles)
    bx0, by0, bx1, by1 = to_px(bug, frame_w, frame_h)
    gray = None
    if bug_image is not None:
        gray = cv2.cvtColor(bug_image, cv2.COLOR_BGR2GRAY) if bug_image.ndim == 3 else bug_image
        if gray.shape != (by1 - by0, bx1 - bx0):
            gray = cv2.resize(gray, (bx1 - bx0, by1 - by0))
        gray = gray.astype(np.float32)
    for name in names:
        x0, y0, x1, y1 = to_px(roles[name], frame_w, frame_h)
        h = y1 - y0
        if name.endswith("_score"):
            pad_x = int(round(0.75 * h))
        elif name in ("clock", "shot_clock"):
            pad_x = int(round(0.45 * h))
        else:
            pad_x = int(round(0.3 * h))
        pad_y = max(1, int(round(0.18 * h)))
        left, right = pad_x, pad_x
        for other in names:
            if other == name:
                continue
            ox0, oy0, ox1, oy1 = to_px(roles[other], frame_w, frame_h)
            if oy1 <= y0 or oy0 >= y1:
                continue  # different row
            if ox1 <= x0:
                left = min(left, (x0 - ox1) // 2)
            if ox0 >= x1:
                right = min(right, (ox0 - x1) // 2)
        left, right = min(left, x0 - bx0), min(right, bx1 - x1)
        top, bottom = min(pad_y, y0 - by0), min(pad_y, by1 - y1)
        if gray is not None:
            rows = gray[max(0, y0 - by0) : max(1, y1 - by0)]
            grad = np.abs(np.diff(rows, axis=1)).mean(axis=0)
            # start two pixels out so the glyphs' own edges do not stop the walk
            lx, rx = max(0, x0 - bx0 - 2), min(gray.shape[1] - 1, x1 - bx0 + 1)
            left = min(left, 2 + _free_run(grad, lx, -1, max(0, left - 2)))
            right = min(right, 2 + _free_run(grad, rx, +1, max(0, right - 2)))
            # the same above and below: the bug's border line or a row of timeout dashes
            # must stay out of the box, or the recognizer sees a bar instead of digits
            cols = gray[:, max(0, x0 - bx0) : max(1, x1 - bx0)]
            vgrad = np.abs(np.diff(cols, axis=0)).mean(axis=1)
            ty, bya = max(0, y0 - by0 - 2), min(gray.shape[0] - 1, y1 - by0 + 1)
            top = min(top, 2 + _free_run(vgrad, ty, -1, max(0, top - 2)))
            bottom = min(bottom, 2 + _free_run(vgrad, bya, +1, max(0, bottom - 2)))
        box = (x0 - max(0, left), y0 - max(0, top), x1 + max(0, right), y1 + max(0, bottom))
        out[name] = from_px(box, frame_w, frame_h)
    return out


def separate_fields(snapped: dict[str, Box], text: dict[str, Box]) -> dict[str, Box]:
    """Keep a snapped box out of its neighbours' text.

    Ink runs fuse a game clock and the shot clock beside it when the two panels touch,
    and the clock box then reads "5:5716". Along its row a box may only grow up to where
    the next field's own text begins.
    """
    out: dict[str, Box] = {}
    for name, (x0, y0, x1, y1) in snapped.items():
        tx0, ty0, tx1, ty1 = text.get(name, (x0, y0, x1, y1))
        for other, (ox0, oy0, ox1, oy1) in text.items():
            if other == name or oy1 <= ty0 or oy0 >= ty1:
                continue  # a different row
            if ox0 >= tx1 and x1 > ox0:
                x1 = ox0
            if ox1 <= tx0 and x0 < ox1:
                x0 = ox1
        out[name] = (x0, y0, x1, y1)
    return out


def snap_to_text(rough: dict[str, Box | None], clusters: list[Cluster]) -> dict[str, Box]:
    """Replace approximate field boxes (e.g. from a vision model) with detected text boxes."""
    out: dict[str, Box] = {}
    taken: set[int] = set()
    for name, box in rough.items():
        if box is None:
            continue
        best, best_score = None, 0.0
        for c in clusters:
            if id(c) in taken:
                continue
            inter = intersection(box, c.union)
            if inter is None:
                continue
            score = box_area(inter) / max(1e-9, min(box_area(box), box_area(c.union)))
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score >= 0.3:
            out[name] = best.union
            taken.add(id(best))
        else:
            out[name] = box
    return out
