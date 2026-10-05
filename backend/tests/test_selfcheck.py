"""The self-check a server runs before each deploy must itself keep working."""

from __future__ import annotations

import pytest

from possession_cut import selfcheck
from possession_cut.synth.script import HOME


def test_check_game_is_short_and_covers_baskets_and_free_throws():
    game = selfcheck.check_game()
    assert 60 < game.duration < 180, "long enough to calibrate on, short enough to run on every deploy"
    cuts = game.cutlist(HOME)
    assert [c["kind"] for c in cuts].count("free_throws") == 1 and len(cuts) == 4
    assert not game.hidden and not game.replays, "nothing in it that needs the network or luck"


@pytest.mark.video
def test_selfcheck_passes(capsys):
    assert selfcheck.run() == 0
    out = capsys.readouterr().out
    assert "Self-check passed" in out and "FAIL" not in out
