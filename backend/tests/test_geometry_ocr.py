from __future__ import annotations

import cv2
import numpy as np
import pytest

from possession_cut.pipeline.geometry import (
    compute_crop,
    crop_from_norm,
    crop_to_norm,
    iou,
    median_box,
    output_placement,
    to_px,
)
from possession_cut.pipeline.ocr import (
    CHARSETS,
    get_engine,
    parse_clock,
    parse_period,
    parse_score,
    parse_shot_clock,
    prepare_field,
)

# -- crop rule (PRD "Output format") ------------------------------------------------


def test_crop_rule_reference_shape():
    # a centred bug 52% of the frame wide gives the reference ~1.18:1 crop
    bug = (0.24, 0.89, 0.76, 0.95)
    x, y, w, h = compute_crop(bug, 1920, 1080)
    assert (y, h) == (0, 1080), "full source height"
    assert w == pytest.approx(0.52 * 1920 / 0.78, abs=2), "width = bug width / 0.78"
    assert w / h == pytest.approx(1.18, abs=0.03)
    assert x + w / 2 == pytest.approx(960, abs=1)
    assert w % 2 == 0 and h % 2 == 0


def test_crop_centres_on_the_bug_not_the_frame():
    bug = (0.30, 0.89, 0.82, 0.95)  # same width, shifted right
    x, _, w, _ = compute_crop(bug, 1920, 1080)
    assert x + w / 2 == pytest.approx(0.56 * 1920, abs=1)


def test_crop_clamps_to_frame_edges():
    bug = (0.50, 0.89, 0.99, 0.95)
    x, _, w, _ = compute_crop(bug, 1920, 1080)
    assert x + w == 1920, "pushed back inside the right edge"
    wide = (0.02, 0.89, 0.98, 0.95)  # bug wider than 78% of the frame
    x, _, w, _ = compute_crop(wide, 1920, 1080)
    assert (x, w) == (0, 1920), "cannot be wider than the frame"


def test_small_corner_bug_does_not_produce_a_sliver():
    corner = (0.03, 0.04, 0.20, 0.10)  # NFL-style corner bug: 17% of the width
    x, y, w, h = compute_crop(corner, 1920, 1080)
    assert w / h >= 1.0, "falls back to the reference shape"
    assert x <= 0.03 * 1920 and x + w >= 0.20 * 1920, "the whole bug stays inside the crop"


def test_crop_roundtrip_and_canvas_placement():
    crop = compute_crop((0.26, 0.89, 0.78, 0.95), 1280, 720)
    assert crop_from_norm(crop_to_norm(crop, 1280, 720), 1280, 720) == crop
    # the same template applied to a 1080p feed of the same broadcast
    x, y, w, h = crop_from_norm(crop_to_norm(crop, 1280, 720), 1920, 1080)
    assert h == 1080 and abs(w - crop[2] * 1.5) <= 1 and w % 2 == 0
    sw, sh, sy = output_placement(crop[2], crop[3])
    assert sw == 1080 and sh == pytest.approx(1080 * crop[3] / crop[2], abs=2)
    assert sy == (1920 - sh) // 2 - ((1920 - sh) // 2) % 2 and sh % 2 == 0


def test_box_helpers():
    a, b = (0.1, 0.1, 0.3, 0.3), (0.2, 0.2, 0.4, 0.4)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, b) == pytest.approx(0.01 / 0.07)
    assert iou(a, (0.5, 0.5, 0.6, 0.6)) == 0
    assert median_box([a, b, (0.2, 0.2, 0.4, 0.4)]) == (0.2, 0.2, 0.4, 0.4)
    assert to_px((0.0, 0.0, 1.0, 1.0), 1280, 720) == (0, 0, 1280, 720)


# -- parsing ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [("12:00", 720.0), ("1:22", 82.0), ("0:07", 7.0), ("45.3", 45.3), ("0.0", 0.0), ("9.9", 9.9),
     ("122", 82.0), ("1022", 622.0), ("1:75", None), ("abc", None), ("", None), ("25:00", None), ("7", None)],
)
def test_parse_clock(text, want):
    assert parse_clock(text) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [("1st", 1), ("2ND", 2), ("3rd", 3), ("4th", 4), ("Q4", 4), ("OT", 5), ("2OT", 6), ("4", 4),
     ("3RO", 3), ("HALF", None), ("FINAL", None), ("", None)],
)
def test_parse_period(text, want):
    assert parse_period(text) == want


def test_parse_scores_and_shot_clock():
    assert parse_score("104") == 104 and parse_score("0") == 0
    assert parse_score("1045") is None and parse_score("9a") is None and parse_score("") is None
    assert parse_shot_clock("24") == 24.0 and parse_shot_clock("4.3") == 4.3
    assert parse_shot_clock("") is None and parse_shot_clock("99") is None


# -- recognition -------------------------------------------------------------------


def _render(text: str, fg=255, bg=20, w=110, h=40) -> np.ndarray:
    img = np.full((h, w, 3), bg, np.uint8)
    cv2.putText(img, text, (8, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (fg, fg, fg), 2, cv2.LINE_AA)
    return img


def test_prepare_field_normalizes_polarity_and_blank():
    light_on_dark = prepare_field(_render("88"))
    dark_on_light = prepare_field(_render("88", fg=10, bg=235))
    for out in (light_on_dark, dark_on_light):
        assert out[0, 0, 0] == 255, "background ends up white"
        assert out.min() < 80, "text ends up dark"
    blank = prepare_field(np.full((40, 110, 3), 22, np.uint8))
    assert blank.min() == 255, "a field with nothing in it stays empty"


def test_restricted_recognition_reads_bug_values():
    engine = get_engine(1)
    cases = [("104", "score"), ("99", "score"), ("7", "score"), ("1:22", "clock"), ("11:59", "clock"),
             ("45.3", "clock"), ("24", "shot_clock"), ("4th", "period")]
    images = [prepare_field(_render(text)) for text, _ in cases]
    results = engine.recognize(images, [CHARSETS[cs] for _, cs in cases])
    for (want, _), (got, conf) in zip(cases, results, strict=True):
        assert got.strip().lower() == want.lower()
        assert conf > 0.3


def test_charset_restriction_keeps_letters_out_of_scores():
    engine = get_engine(1)
    # 'SO' looks like '50' to a digit-only reader; unrestricted it stays letters
    img = prepare_field(_render("SO"))
    (free, _), = engine.recognize([img], [None])
    (digits, _), = engine.recognize([img], [CHARSETS["score"]])
    assert free.strip().upper().isalpha()
    assert digits == "" or digits.isdigit()


def test_blank_field_reads_empty():
    engine = get_engine(1)
    (text, conf), = engine.recognize([prepare_field(np.full((40, 60, 3), 25, np.uint8))], [CHARSETS["shot_clock"]])
    assert text == "" and conf == 0.0
