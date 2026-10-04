"""Suggested title and post caption for an export.

The caption is written by Claude from the play-by-play facts when it is available, and
from a template otherwise. Either way it only states what the data supports.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from .claude import ClaudeClient, ClaudeError

log = logging.getLogger(__name__)

CAPTION_SYSTEM = (
    "You write the post caption for a short vertical sports highlight video (TikTok, Reels, "
    "Shorts). You are given facts about the video as JSON. Write one or two short, punchy "
    "sentences in the voice of a fan account, then a blank line, then 5 to 8 relevant hashtags "
    "on one line. Use only the facts given: do not invent stats, quotes, records or player "
    "feats. No emojis unless the facts include them. Under 300 characters before the hashtags. "
    "Return only the caption text."
)


@dataclass
class CutFacts:
    sport: str = "NBA"
    team: str = ""  # followed team, short name ("Knicks")
    team_abbr: str = ""
    opponent: str = ""
    opponent_abbr: str = ""
    game_label: str = ""  # "NBA Finals · Game 4"
    date: str = ""
    final_score: str = ""  # "NYK 107, SAS 106"
    won: bool | None = None
    start_label: str = ""
    deficit: int | None = None  # largest deficit the cut starts from, if it is a comeback cut
    clips: int = 0
    points: int = 0
    duration_seconds: float = 0.0
    top_scorers: list[str] = field(default_factory=list)  # "J. Brunson 18"
    plays: list[str] = field(default_factory=list)  # a few play descriptions

    def to_dict(self) -> dict:
        return asdict(self)


def suggested_title(facts: CutFacts) -> str:
    team = facts.team or facts.team_abbr or "Team"
    versus = f" vs {facts.opponent or facts.opponent_abbr}" if (facts.opponent or facts.opponent_abbr) else ""
    if facts.deficit and facts.deficit >= 10 and facts.won:
        return f"{team} {facts.deficit}-point comeback{versus}"
    if facts.deficit and facts.deficit >= 10:
        return f"{team} run from {facts.deficit} down{versus}"
    return f"Every {team} bucket{versus}"


def _tag(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text)
    return f"#{cleaned}" if cleaned else ""


def template_caption(facts: CutFacts) -> str:
    team = facts.team or facts.team_abbr or "They"
    opp = facts.opponent or facts.opponent_abbr or "the other team"
    if facts.deficit and facts.deficit >= 10 and facts.won:
        line = f"{team} were down {facts.deficit} to {opp} and won it. Every bucket of the comeback, no dead time."
    elif facts.deficit and facts.deficit >= 10:
        line = f"{team} from {facts.deficit} down against {opp}. Every bucket of the run, no dead time."
    else:
        line = f"Every {team} bucket against {opp}, back to back, no dead time."
    if facts.final_score:
        line += f" Final: {facts.final_score}."
    tags = [_tag(facts.sport), _tag(team), _tag(opp)]
    if "final" in facts.game_label.lower():
        tags.append(_tag(f"{facts.sport}Finals"))
    if facts.deficit and facts.deficit >= 10:
        tags.append("#comeback")
    tags += ["#highlights", "#basketball" if facts.sport.upper() == "NBA" else ""]
    seen: list[str] = []
    for t in tags:
        if t and t.lower() not in (s.lower() for s in seen):
            seen.append(t)
    return f"{line}\n\n{' '.join(seen)}"


def make_caption(facts: CutFacts, claude: ClaudeClient | None) -> tuple[str, str]:
    """(caption text, source) where source is "claude" or "template"."""
    if claude is not None and claude.available:
        try:
            text = claude.text(
                purpose="caption",
                system=CAPTION_SYSTEM,
                prompt="Facts about the video:\n" + json.dumps(facts.to_dict(), indent=1),
                max_tokens=2000,
                effort="low",
            )
            if text and "#" in text:
                return text.strip(), "claude"
            log.info("Claude caption came back without hashtags; using the template")
        except ClaudeError as exc:
            log.info("caption fell back to the template: %s", exc)
    return template_caption(facts), "template"
