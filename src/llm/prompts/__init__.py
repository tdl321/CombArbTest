"""Competition-specific prompt system.

Usage:
    from src.llm.prompts import get_competition_prompt, compose_system_prompt

    # Explicit
    prompt = get_competition_prompt("f1")

    # Auto-detect from market data
    prompt = detect_competition(markets=[{"question": "..."}], theme="F1 2026")

    # Compose into a full system prompt string
    system = compose_system_prompt(prompt)
"""

from .base import CompetitionPrompt, compose_system_prompt
from .registry import detect_competition, get as get_competition_prompt, list_competitions, register

__all__ = [
    "CompetitionPrompt",
    "compose_system_prompt",
    "detect_competition",
    "get_competition_prompt",
    "list_competitions",
    "register",
]
