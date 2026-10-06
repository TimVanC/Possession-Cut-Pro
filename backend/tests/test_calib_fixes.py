"""Boxes that reach a hair too far: into the bug's border line, the next field, a logo.

Found on the 2016 Finals (ESPN's bottom bar): the home score box took in the bar's white
top line and read as nothing, so every Warriors basket was missing, and the clock box ran
into the shot clock beside it and read "5:5716".
"""

from __future__ import annotations

import cv2
import numpy as np

from possession_cut.pipeline.calib_local import separate_fields, size_field_boxes
from possession_cut.pipeline.calibration import Calibration, tune_fields
from possession_cut.pipeline.geometry import from_px
from possession_cut.pipeline.ocr import drop_rules, prepare_field
from possession_cut.sports import get_adapter

from .test_calib_local import bug_strip

NBA = get_adapter("nba")


def test_lines_are_dropped_from_a_field_but_glyphs_and_dots_stay():
    img = np.full((60, 200), 255, np.uint8)
    img[2:6, 5:195] = 0  # the bug's border line, caught by a box one row too tall
    img[54:57, 20:40] = 0  # a timeout dash
    img[10:56, 0:3] = 0  # the edge of the next panel
    img[12:52, 60:80] = 0  # a digit
    img[12:52, 100:118] = 0  # another
    img[30:36, 88:94] = 0  # a colon's dot between them
    out = drop_rules(img)
    assert out[3, 100] == 255 and out[55, 30] == 255 and out[30, 1] == 255, "lines and the edge bar are gone"
    assert out[30, 70] == 0 and out[30, 110] == 0 and out[33, 91] == 0, "glyphs and the dot stay"


def test_a_border_line_in_the_box_no_longer_blanks_the_read():
    # white digits on a dark panel with the panel's bright top line inside the box
    crop = np.full((44, 69, 3), 25, np.uint8)
    cv2.putText(crop, "11", (14, 36), cv2.FONT_HERSHEY_DUPLEX, 1.1, (240, 240, 240), 3)
    crop[0:3, :] = 230
    prepared = prepare_field(crop)
    ink = prepared[:, :, 0] < 128
    rows = np.flatnonzero(ink.any(axis=1))
    assert rows[-1] - rows[0] > 0.5 * prepared.shape[0], "what is left is the digits, not a thin bar"
    assert not ink[:3].any(), "the line was not kept as ink"


def test_a_snapped_box_stops_at_the_next_fields_text():
    text = {"clock": (0.693, 0.80, 0.745, 0.84), "shot_clock": (0.749, 0.80, 0.773, 0.84), "period": (0.65, 0.80, 0.689, 0.84)}
    snapped = {"clock": (0.693, 0.80, 0.773, 0.84), "shot_clock": (0.749, 0.80, 0.773, 0.84), "period": (0.64, 0.80, 0.70, 0.84)}
    out = separate_fields(snapped, text)
    assert out["clock"] == (0.693, 0.80, 0.749, 0.84), "the clock may not reach into the shot clock"
    assert out["shot_clock"] == snapped["shot_clock"]
    assert out["period"] == (0.64, 0.80, 0.693, 0.84), "a box may grow only up to the neighbour's text"


def test_vertical_padding_stops_at_a_line_in_the_bug():
    fw, fh = 1280, 720
    bug = from_px((300, 636, 1000, 668), fw, fh)
    role = {"away_score": from_px((420, 644, 452, 660), fw, fh)}
    plain = np.full((32, 700), 20, np.uint8)
    lined = plain.copy()
    lined[5, :] = 235  # a bright rule three pixels above the digits
    free = size_field_boxes(role, bug, fw, fh, plain)["away_score"]
    boxed = size_field_boxes(role, bug, fw, fh, lined)["away_score"]
    assert free[1] * fh <= 641.5, "without the line the box gets its full padding"
    assert boxed[1] * fh >= 641.5, "with it the box stays below the line"
    assert boxed[3] == free[3]


def test_a_box_that_reads_an_extra_digit_is_pulled_in(ocr_engine):
    frames = [bug_strip([(a, 425), (b, 478)]) for a, b in (("29", "17"), ("31", "19"), ("35", "22"))]
    fw, fh = 1280, 720
    cal = Calibration(
        bug=list(from_px((300, 636, 1000, 668), fw, fh)),
        fields={"away_score": list(from_px((400, 640, 486, 664), fw, fh))},
        crop=[0.0, 0.0, 1.0, 1.0], frame_width=fw, frame_height=fh,
    )
    before = list(cal.fields["away_score"])
    notes = tune_fields(cal, NBA, frames, ocr_engine, None, None)
    after = cal.fields["away_score"]
    assert notes == ["The away score box was pulled in so it reads cleanly."]
    assert after[2] < before[2] and after[2] * fw <= 480, "the right edge no longer covers the next number"
    assert after[0] * fw <= 425, "the digits themselves are still inside"


def test_a_bug_that_names_other_teams_than_the_picked_game_is_flagged():
    from possession_cut.pipeline.calibration import check_teams

    cal = Calibration(bug=[0, 0, 1, 1], fields={}, crop=[0, 0, 1, 1], frame_width=1280, frame_height=720, teams={"away": "CLE", "home": "GS"})
    check_teams(cal, {"away": "CLE", "home": "GSW"})
    assert cal.warnings == [], "GS is the Warriors"
    check_teams(cal, {"away": "SAS", "home": "NYK"})
    assert cal.warnings and cal.warnings[0].startswith("The bug reads CLE / GS, but the game picked is SAS at NYK")
    blank = Calibration(bug=[0, 0, 1, 1], fields={}, crop=[0, 0, 1, 1], frame_width=1280, frame_height=720)
    check_teams(blank, {"away": "SAS", "home": "NYK"})
    check_teams(cal, None)
    assert blank.warnings == [] and len(cal.warnings) == 1, "no labels, or no game: nothing to compare"
