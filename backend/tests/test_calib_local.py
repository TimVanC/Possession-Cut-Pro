"""The parts of the local detector that real broadcasts stress: fused text and field edges.

On a real ESPN/ABC bug the period, clock and shot clock sit so close together that text
detection returns them as one string ("2ND3:0020"). These tests pin down how that is
taken apart, without needing broadcast footage in the repo.
"""

from __future__ import annotations

import cv2
import numpy as np

from possession_cut.pipeline.calib_local import (
    CLOCK_INSIDE,
    Cluster,
    LocalDetector,
    Obs,
    _cluster,
    _fix_digits,
    _split_merged,
)

BOX = (0.60, 0.88, 0.76, 0.92)


def parts(text: str) -> list[str]:
    return [o.text for o in _split_merged(Obs(0, BOX, text, 0.9))]


def test_fused_lines_are_split_into_fields():
    assert parts("2ND3:0020") == ["2ND", "3:00", "20"]
    assert parts("1ST7:5213") == ["1ST", "7:52", "13"]
    assert parts("10:4722") == ["10:47", "22"]
    assert parts("7:2417") == ["7:24", "17"]
    assert parts("3rd  2:27") == ["3rd", "2:27"]
    assert parts("2:27 22") == ["2:27", "22"]
    assert parts("4TH 38.4") == ["4TH", "38.4"]
    assert parts("4:26") == ["4:26"]
    assert parts("104") == ["104"] and parts("NY") == ["NY"]


def test_split_pieces_keep_their_order_and_are_marked_as_estimates():
    pieces = _split_merged(Obs(3, BOX, "2ND3:0020", 0.9))
    assert all(p.split and p.frame == 3 for p in pieces)
    xs = [p.box[0] for p in pieces]
    assert xs == sorted(xs) and pieces[0].box[0] == BOX[0] and pieces[-1].box[2] == BOX[2]
    assert pieces[1].box[0] >= pieces[0].box[2] - 1e-9, "pieces do not overlap"
    whole = _split_merged(Obs(0, BOX, "88", 0.9))
    assert len(whole) == 1 and not whole[0].split


def test_letter_o_next_to_digits_reads_as_zero():
    assert _fix_digits("1O:47") == "10:47"
    assert _fix_digits("4TH1O:4722") == "4TH10:4722"
    assert _fix_digits("BONUS") == "BONUS" and _fix_digits("NY") == "NY", "words are left alone"
    assert parts("4TH1O:4722") == ["4TH", "10:47", "22"]


def test_clock_is_found_inside_a_longer_string():
    for text, clock in [("1ST7:5213", "7:52"), ("2:5324", "2:53"), ("4TH10:4722", "10:47"), ("38.4", "38.4")]:
        assert CLOCK_INSIDE.search(text).group(0) == clock
    assert CLOCK_INSIDE.search("NYLEADS2-1") is None and CLOCK_INSIDE.search("7MINS") is None
    assert CLOCK_INSIDE.search("7:95") is None, "seconds above 59 are not a clock"


def test_exact_boxes_define_a_field_and_estimates_only_join():
    exact = [Obs(f, (0.640, 0.89, 0.702, 0.92), "4:10", 0.9) for f in range(3)]
    # from a fused "10:4722": the clock piece is too wide, the shot clock piece starts too early
    wide_clock = Obs(3, (0.636, 0.89, 0.716, 0.92), "10:47", 0.9, split=True)
    early_shot = Obs(3, (0.716, 0.89, 0.748, 0.92), "22", 0.9, split=True)
    shot = [Obs(f, (0.722, 0.89, 0.747, 0.92), "24", 0.9) for f in range(3)]
    clusters = _cluster([wide_clock, early_shot, *exact, *shot])
    assert len(clusters) == 2, "estimates join the exact clusters instead of starting their own"
    clock = next(c for c in clusters if "4:10" in c.texts)
    assert clock.union == (0.640, 0.89, 0.702, 0.92), "the field's extent comes from the exact boxes"
    assert clock.frames == {0, 1, 2, 3}
    only_estimates = Cluster([wide_clock])
    assert only_estimates.union == wide_clock.box, "estimates are used when there is nothing better"


def bug_strip(values: list[tuple[str, int]], width: int = 700) -> np.ndarray:
    """A 720p frame with bold broadcast-style text on one row, each string starting at x."""
    from PIL import Image, ImageDraw

    from possession_cut.fonts import load_font

    img = Image.new("RGB", (1280, 720), (90, 90, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([300, 636, 300 + width, 668], fill=(18, 18, 18))
    font = load_font(26, bold=True)
    for text, x in values:
        draw.text((x, 652), text, font=font, fill=(240, 240, 240), anchor="lm")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def test_ink_runs_separate_tightly_packed_fields():
    frames = [
        bug_strip([("SA", 400), ("64", 505), ("NY", 585), ("40", 690), ("2ND", 758), ("4:26", 822), ("17", 928)]),
        bug_strip([("SA", 400), ("92", 505), ("NY", 585), ("75", 690), ("4TH", 758), ("10:47", 814), ("22", 928)]),
        bug_strip([("SA", 400), ("104", 492), ("NY", 585), ("99", 690), ("4TH", 758), ("1:22", 822), ("24", 928)]),
    ]
    det = LocalDetector(frames, engine=None)  # type: ignore[arg-type]
    bug = (300 / 1280, 636 / 720, 1000 / 1280, 668 / 720)
    row = (0.62, 642 / 720, 0.72, 663 / 720)
    runs = [(round(a * 1280), round(b * 1280)) for a, b in det.ink_runs([0, 1, 2], bug, row)]
    assert len(runs) == 7, runs
    period, clock, shot = runs[4], runs[5], runs[6]
    assert period[1] < clock[0] and clock[1] < shot[0], "period, clock and shot clock come out as three fields"
    assert clock[0] <= 816, "the clock's ink spans its widest value (10:47)"
    assert shot[0] >= 926

    # boxes that were estimated too wide (from a fused "10:4722") are pulled back to their own ink
    roles = {
        "clock": ((clock[0] - 6) / 1280, 0.89, (shot[0] + 9) / 1280, 0.92),
        "shot_clock": ((shot[0] - 4) / 1280, 0.89, (shot[1] + 6) / 1280, 0.92),
        "period": ((period[0] - 5) / 1280, 0.89, (period[1] + 4) / 1280, 0.92),
    }
    snapped = det.snap_to_ink(roles, det.ink_runs([0, 1, 2], bug, row))
    as_px = lambda b: (round(b[0] * 1280), round(b[2] * 1280))  # noqa: E731
    assert as_px(snapped["clock"]) == clock and as_px(snapped["shot_clock"]) == shot and as_px(snapped["period"]) == period
    assert snapped["clock"][2] < snapped["shot_clock"][0], "the two fields no longer overlap"


def test_ink_runs_ignore_block_edges_and_empty_rows():
    frame = np.full((720, 1280, 3), 90, np.uint8)
    frame[636:668, 300:650] = 10  # two blocks of different colours side by side, no text
    frame[636:668, 650:1000] = 235
    det = LocalDetector([frame, frame.copy()], engine=None)  # type: ignore[arg-type]
    bug = (300 / 1280, 636 / 720, 1000 / 1280, 668 / 720)
    assert det.ink_runs([0, 1], bug, (0.5, 643 / 720, 0.6, 663 / 720)) == []
