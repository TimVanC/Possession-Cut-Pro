"""Claude access for the worker: calibration vision, OCR fallback, captions.

Every call goes through ``ClaudeClient.json`` / ``.text`` so spend is metered against the
per-job budget (``CLAUDE_BUDGET_PER_JOB_USD``). When there is no API key the client
reports ``available == False`` and callers use their local fallbacks.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import get_settings

log = logging.getLogger(__name__)

# USD per million tokens (input, output). Unknown models are priced like Sonnet.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_PRICE = (2.0, 10.0)

# Lets the API re-run a request on a fallback model if a safety classifier declines it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeError(RuntimeError):
    """The call failed or returned something unusable."""


class ClaudeUnavailable(ClaudeError):
    """No API key configured."""


class BudgetExceeded(ClaudeError):
    """The per-job budget would be exceeded."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    log: list[dict] = field(default_factory=list)


def encode_image(image: np.ndarray, max_side: int = 1568, quality: int = 88) -> dict:
    """BGR array to an image content block. Downscales so the long side is at most ``max_side``."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ClaudeError("could not encode image")
    data = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


class ClaudeClient:
    def __init__(self, budget_usd: float | None = None, spent_usd: float = 0.0, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.claude_model
        self.budget_usd = settings.claude_budget_per_job_usd if budget_usd is None else budget_usd
        self.usage = Usage(cost_usd=spent_usd)
        self._key = settings.anthropic_api_key
        self._workspace = settings.anthropic_workspace_id
        self._client = None
        self._disabled: str | None = None  # set when the account/config cannot work at all
        self._lock = threading.Lock()
        self._use_fallbacks = True

    @property
    def available(self) -> bool:
        return bool(self._key) and self._disabled is None

    @property
    def unavailable_reason(self) -> str | None:
        if not self._key:
            return "ANTHROPIC_API_KEY is not set"
        return self._disabled

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.usage.cost_usd)

    def _sdk(self):
        if not self.available:
            raise ClaudeUnavailable(self.unavailable_reason or "Claude is not available")
        if self._client is None:
            import anthropic

            headers = {"anthropic-workspace-id": self._workspace} if self._workspace else None
            self._client = anthropic.Anthropic(
                api_key=self._key, max_retries=3, timeout=120.0, default_headers=headers
            )
        return self._client

    def _disable(self, reason: str) -> ClaudeUnavailable:
        """Stop calling for the rest of this job; the problem will not fix itself mid-run."""
        self._disabled = reason
        log.warning("Claude disabled for this job: %s", reason)
        return ClaudeUnavailable(reason)

    def _charge(self, response, purpose: str) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        fresh = getattr(usage, "input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        out_tokens = getattr(usage, "output_tokens", 0) or 0
        price_in, price_out = PRICES.get(getattr(response, "model", None) or self.model, PRICES.get(self.model, DEFAULT_PRICE))
        cost = (fresh * price_in + cache_write * price_in * 1.25 + cache_read * price_in * 0.1 + out_tokens * price_out) / 1e6
        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += fresh + cache_write + cache_read
            self.usage.output_tokens += out_tokens
            self.usage.cost_usd += cost
            self.usage.log.append({"purpose": purpose, "input": fresh + cache_write + cache_read,
                                   "output": out_tokens, "cost_usd": round(cost, 5)})

    def _create(self, purpose: str, estimate_usd: float, **kwargs):
        import anthropic

        if self.usage.cost_usd + estimate_usd > self.budget_usd:
            raise BudgetExceeded(
                f"Claude budget of ${self.budget_usd:.2f} for this job is used up "
                f"(${self.usage.cost_usd:.2f} spent)."
            )
        client = self._sdk()
        try:
            if self._use_fallbacks:
                try:
                    response = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
                except (TypeError, anthropic.BadRequestError, anthropic.NotFoundError) as exc:
                    if "workspace" in str(getattr(exc, "message", exc)).lower():
                        raise
                    # Older SDKs / accounts without the fallback beta: plain call from here on.
                    log.info("server-side fallback not accepted (%s); using plain requests", type(exc).__name__)
                    self._use_fallbacks = False
                    response = client.messages.create(**kwargs)
            else:
                response = client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise self._disable("Anthropic rejected the API key (check ANTHROPIC_API_KEY in .env).") from exc
        except anthropic.PermissionDeniedError as exc:
            raise self._disable(f"Anthropic denied the request: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise ClaudeError(f"Model {self.model!r} was not found (check CLAUDE_MODEL in .env).") from exc
        except anthropic.RateLimitError as exc:
            raise ClaudeError("Anthropic rate limit hit; try again in a minute.") from exc
        except anthropic.BadRequestError as exc:
            if "workspace" in str(exc.message).lower():
                raise self._disable(
                    "The API key is not scoped to a workspace. Set ANTHROPIC_WORKSPACE_ID in .env "
                    "(Claude Console > Settings > Workspaces) or use a workspace-scoped key."
                ) from exc
            raise ClaudeError(f"Anthropic rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise ClaudeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ClaudeError("Could not reach the Anthropic API (network).") from exc
        self._charge(response, purpose)
        if response.stop_reason == "refusal":
            raise ClaudeError("Claude declined the request.")
        return response

    @staticmethod
    def _text(response) -> str:
        return "".join(block.text for block in response.content if block.type == "text")

    def json(
        self,
        *,
        purpose: str,
        system: str,
        content: list[dict],
        schema: dict[str, Any],
        max_tokens: int = 4096,
        effort: str = "low",
        estimate_usd: float = 0.02,
    ) -> dict:
        """One request that must return JSON matching ``schema`` (structured outputs)."""
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        response = self._create(purpose, estimate_usd, **kwargs)
        if response.stop_reason == "max_tokens":
            kwargs["max_tokens"] = max_tokens * 3
            response = self._create(purpose, estimate_usd, **kwargs)
        text = self._text(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClaudeError(f"Claude returned text that is not JSON: {text[:200]!r}") from exc

    def text(
        self,
        *,
        purpose: str,
        system: str,
        prompt: str,
        max_tokens: int = 2048,
        effort: str = "low",
        estimate_usd: float = 0.01,
    ) -> str:
        response = self._create(
            purpose,
            estimate_usd,
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
        )
        return self._text(response).strip()
