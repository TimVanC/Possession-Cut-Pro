"""Claude-backed pieces, tested against stand-ins for the SDK and the client.

No test here talks to the real API. The live route could not be exercised during the
build (see BUILD_NOTES: the supplied key is not scoped to a workspace).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from possession_cut.ai import claude as claude_mod
from possession_cut.ai.caption import CutFacts, make_caption, suggested_title, template_caption
from possession_cut.ai.claude import (
    BudgetExceeded,
    ClaudeClient,
    ClaudeError,
    ClaudeUnavailable,
    encode_image,
)
from possession_cut.pipeline.events import detect_score_events
from possession_cut.pipeline.ocr_fallback import contact_sheet, pick_samples, reread_with_claude
from possession_cut.pipeline.timeline import build_timeline
from possession_cut.sports import get_adapter

from .test_timeline import make_raw, running

NBA = get_adapter("nba")


def response(text='{"ok": true}', stop="end_turn", inp=1000, out=200, model="claude-sonnet-5-5"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop, model=model,
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


class FakeSDK:
    def __init__(self, beta_error: Exception | None = None, replies=None):
        self.beta_calls: list[dict] = []
        self.plain_calls: list[dict] = []
        self.replies = list(replies or [])
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._beta))
        self.messages = SimpleNamespace(create=self._plain)
        self._beta_error = beta_error

    def _next(self):
        return self.replies.pop(0) if self.replies else response()

    def _beta(self, **kw):
        self.beta_calls.append(kw)
        if self._beta_error:
            raise self._beta_error
        return self._next()

    def _plain(self, **kw):
        self.plain_calls.append(kw)
        return self._next()


@pytest.fixture()
def client(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from possession_cut import config

    config.get_settings.cache_clear()
    c = ClaudeClient(budget_usd=0.05)
    yield c
    config.get_settings.cache_clear()


def test_no_key_means_unavailable(settings):
    c = ClaudeClient()
    assert not c.available and "ANTHROPIC_API_KEY" in c.unavailable_reason
    with pytest.raises(ClaudeUnavailable):
        c.json(purpose="x", system="s", content=[], schema={})


def test_json_request_shape_and_metering(client):
    sdk = FakeSDK(replies=[response('{"a": 1}', inp=10_000, out=1_000)])
    client._client = sdk
    out = client.json(purpose="calibration:locate", system="sys", content=[{"type": "text", "text": "hi"}],
                      schema={"type": "object"}, effort="medium")
    assert out == {"a": 1}
    (call,) = sdk.beta_calls
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert call["fallbacks"] == "default" and call["betas"] == [claude_mod.FALLBACK_BETA]
    assert "thinking" not in call and "temperature" not in call and "tool_choice" not in call
    # Sonnet 5.5: $2 per million in, $10 per million out
    assert client.usage.cost_usd == pytest.approx(10_000 * 2 / 1e6 + 1_000 * 10 / 1e6)
    assert client.usage.calls == 1 and client.usage.log[0]["purpose"] == "calibration:locate"


def test_budget_is_enforced_before_the_call(client):
    sdk = FakeSDK(replies=[response(inp=20_000, out=500)])
    client._client = sdk
    client.json(purpose="a", system="s", content=[], schema={}, estimate_usd=0.02)
    assert client.usage.cost_usd == pytest.approx(0.045)
    with pytest.raises(BudgetExceeded):
        client.json(purpose="b", system="s", content=[], schema={}, estimate_usd=0.02)
    assert len(sdk.beta_calls) == 1, "the over-budget request was never sent"
    assert client.remaining_usd == pytest.approx(0.005)


def test_falls_back_to_a_plain_request_when_the_fallback_beta_is_refused(client):
    sdk = FakeSDK(beta_error=TypeError("unexpected keyword argument 'fallbacks'"))
    client._client = sdk
    assert client.json(purpose="a", system="s", content=[], schema={}) == {"ok": True}
    assert len(sdk.beta_calls) == 1 and len(sdk.plain_calls) == 1
    client.json(purpose="b", system="s", content=[], schema={})
    assert len(sdk.beta_calls) == 1 and len(sdk.plain_calls) == 2, "it does not try the beta again"
    assert "fallbacks" not in sdk.plain_calls[0]


def test_refusal_bad_json_and_truncation(client):
    client._client = FakeSDK(replies=[response(stop="refusal")])
    with pytest.raises(ClaudeError, match="declined"):
        client.json(purpose="a", system="s", content=[], schema={})
    client._client = FakeSDK(replies=[response("not json")])
    with pytest.raises(ClaudeError, match="not JSON"):
        client.json(purpose="a", system="s", content=[], schema={})
    sdk = FakeSDK(replies=[response('{"a":', stop="max_tokens"), response('{"a": 2}')])
    client._client = sdk
    client.budget_usd = 1.0
    assert client.json(purpose="a", system="s", content=[], schema={}, max_tokens=1000) == {"a": 2}
    assert sdk.beta_calls[1]["max_tokens"] == 3000, "retried once with more room"


def test_workspace_error_disables_the_client_for_the_job(client):
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.BadRequestError(
        "This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header",
        response=httpx2.Response(400, request=req), body=None,
    )
    client._client = FakeSDK(beta_error=err)
    with pytest.raises(ClaudeUnavailable, match="ANTHROPIC_WORKSPACE_ID"):
        client.json(purpose="a", system="s", content=[], schema={})
    assert not client.available and "workspace" in client.unavailable_reason


def test_workspace_id_is_sent_as_a_header(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    from possession_cut import config

    config.get_settings.cache_clear()
    sdk = ClaudeClient()._sdk()
    assert sdk.default_headers.get("anthropic-workspace-id") == "wrkspc_test"
    config.get_settings.cache_clear()


def test_encode_image_downscales():
    block = encode_image(np.zeros((1080, 1920, 3), dtype=np.uint8), max_side=1568)
    assert block["type"] == "image" and block["source"]["media_type"] == "image/jpeg"
    assert len(block["source"]["data"]) > 100


# -- captions -----------------------------------------------------------------------


FACTS = CutFacts(
    team="Knicks", team_abbr="NYK", opponent="Spurs", opponent_abbr="SAS", game_label="NBA Finals · Game 4",
    date="2026-06-10", final_score="NYK 107, SAS 106", won=True, start_label="Q3 9:40", deficit=29,
    clips=22, points=55, duration_seconds=300, top_scorers=["J. Brunson 21"],
)


def test_titles():
    assert suggested_title(FACTS) == "Knicks 29-point comeback vs Spurs"
    assert suggested_title(CutFacts(team="Knicks", opponent="Spurs", deficit=18, won=False)) == "Knicks run from 18 down vs Spurs"
    assert suggested_title(CutFacts(team="Knicks", opponent="Spurs")) == "Every Knicks bucket vs Spurs"
    assert suggested_title(CutFacts(team_abbr="NY")) == "Every NY bucket"


def test_template_caption_states_only_the_facts():
    text = template_caption(FACTS)
    body, tags = text.split("\n\n")
    assert "down 29" in body and "Spurs" in body and "NYK 107, SAS 106" in body
    assert tags.split() == ["#NBA", "#Knicks", "#Spurs", "#NBAFinals", "#comeback", "#highlights", "#basketball"]
    plain = template_caption(CutFacts(team="Knicks", opponent="Spurs"))
    assert "comeback" not in plain and "#comeback" not in plain


def test_caption_uses_claude_when_available_and_template_otherwise():
    class Yes:
        available = True

        def text(self, **kw):
            assert "Knicks" in kw["prompt"] and kw["purpose"] == "caption"
            return "Down 29 in the third. Knicks took Game 4 anyway.\n\n#Knicks #NBAFinals"

    class Broken(Yes):
        def text(self, **kw):
            raise ClaudeError("boom")

    class NoTags(Yes):
        def text(self, **kw):
            return "A caption with no hashtags"

    assert make_caption(FACTS, Yes()) == ("Down 29 in the third. Knicks took Game 4 anyway.\n\n#Knicks #NBAFinals", "claude")
    assert make_caption(FACTS, Broken())[1] == "template"
    assert make_caption(FACTS, NoTags())[1] == "template"
    assert make_caption(FACTS, None)[1] == "template"


# -- OCR fallback ---------------------------------------------------------------------


def uncertain_game():
    """A basket whose score is unreadable for four samples before the new value shows."""
    rows = running(300, 20, home=50)
    rows += [{"clock": r["clock"], "home": ""} for r in running(290, 4)]
    rows += running(288, 16, home=52)
    raw = make_raw(rows)
    for i in range(20, 24):
        raw.crops[i] = np.full((46, 664, 3), 30, dtype=np.uint8)
    tl = build_timeline(raw, NBA)
    return raw, tl, detect_score_events(tl, NBA)


class SheetReader:
    """Answers each contact sheet: rows 1-2 still show 50, rows 3-4 already show 52."""

    available = True
    unavailable_reason = None

    def __init__(self, remaining=1.0):
        self.remaining_usd = remaining
        self.calls = 0

    def json(self, *, purpose, system, content, schema, **kw):
        assert purpose == "ocr_fallback" and content[0]["type"] == "image"
        self.calls += 1
        return {"rows": [
            {"row": 1, "away_score": 10, "home_score": 50, "period": "2nd", "clock": None},
            {"row": 2, "away_score": 10, "home_score": 50, "period": "2nd", "clock": None},
            {"row": 3, "away_score": 10, "home_score": 52, "period": "2nd", "clock": None},
            {"row": 4, "away_score": 10, "home_score": 52, "period": None, "clock": "25:99"},
            {"row": 9, "away_score": 1, "home_score": 1, "period": None, "clock": None},
        ]}


def test_fallback_pins_the_moment_a_score_appeared():
    raw, tl, events = uncertain_game()
    (before,) = events
    assert before.t == 12.0, "local OCR only sees the new score once it becomes readable"
    assert pick_samples(raw, tl, events) == [20, 21, 22, 23]

    report = reread_with_claude(raw, tl, events, NBA, SheetReader())
    assert report["sheets"] == 1 and report["sent"] == 4 and report["updated"] >= 4 and not report["skipped"]
    assert raw.texts["home_score"][20:24] == ["50", "50", "52", "52"]
    assert raw.texts["clock"][23] != "25:99", "values that do not parse are ignored"
    (after,) = detect_score_events(build_timeline(raw, NBA), NBA)
    assert after.t == 11.0, "the basket is now stamped a second earlier"


def test_fallback_respects_budget_and_availability():
    raw, tl, events = uncertain_game()
    broke = SheetReader(remaining=0.001)
    report = reread_with_claude(raw, tl, events, NBA, broke)
    assert broke.calls == 0 and "budget" in report["skipped"]

    off = SimpleNamespace(available=False, unavailable_reason="The API key is not scoped to a workspace.")
    assert "workspace" in reread_with_claude(raw, tl, events, NBA, off)["skipped"]
    assert reread_with_claude(raw, tl, events, NBA, None)["skipped"]


def test_fallback_declines_when_ocr_is_broadly_bad():
    raw, tl, events = uncertain_game()
    for i in range(len(raw)):
        raw.crops[i] = np.zeros((46, 664, 3), dtype=np.uint8)
    reader = SheetReader()
    report = reread_with_claude(raw, tl, events, NBA, reader)
    assert reader.calls == 0 and "calibration problem" in report["skipped"]


def test_contact_sheet_layout():
    sheet = contact_sheet([np.full((46, 664, 3), 40, dtype=np.uint8)] * 3)
    assert sheet.shape[1] == 1100 and sheet.shape[0] > 3 * 60
    assert sheet[:, :90].min() < 50, "row numbers are drawn at the left"
