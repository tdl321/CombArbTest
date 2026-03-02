"""Competition prompt registry with auto-detection."""

from __future__ import annotations

import logging
from typing import Any

from .base import CompetitionPrompt
from .f1 import F1_PROMPT
from .nfl import NFL_PROMPT
from .nba import NBA_PROMPT
from .world_cup import WORLD_CUP_PROMPT

logger = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────

_REGISTRY: dict[str, CompetitionPrompt] = {
    p.competition: p
    for p in [F1_PROMPT, NFL_PROMPT, NBA_PROMPT, WORLD_CUP_PROMPT]
}


def register(prompt: CompetitionPrompt) -> None:
    """Register a new competition prompt at runtime."""
    _REGISTRY[prompt.competition] = prompt
    logger.debug("Registered competition prompt: %s", prompt.competition)


def get(competition: str) -> CompetitionPrompt | None:
    """Get a competition prompt by slug."""
    return _REGISTRY.get(competition)


def list_competitions() -> list[str]:
    """List all registered competition slugs."""
    return list(_REGISTRY.keys())


# ── Auto-detection ──────────────────────────────────────

def detect_competition(
    markets: list[dict[str, Any]] | None = None,
    theme: str = "",
) -> CompetitionPrompt | None:
    """Auto-detect competition type from market data.

    Scores each registered prompt by keyword hits in the
    combined text of market questions + theme. Returns the
    highest-scoring match, or None if no keywords match.

    Args:
        markets: List of dicts with at least a "question" key.
        theme: Optional theme/title string.

    Returns:
        Best-matching CompetitionPrompt, or None.
    """
    text_parts = [theme.lower()]
    for m in (markets or []):
        q = m.get("question", "")
        text_parts.append(q.lower())
    combined = " ".join(text_parts)

    best: CompetitionPrompt | None = None
    best_score = 0

    for prompt in _REGISTRY.values():
        score = sum(1 for kw in prompt.detection_keywords if kw in combined)
        if score > best_score:
            best_score = score
            best = prompt

    if best:
        logger.info(
            "Auto-detected competition: %s (score=%d)",
            best.competition, best_score,
        )
    else:
        logger.debug("No competition detected from market text")

    return best
