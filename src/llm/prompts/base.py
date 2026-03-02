"""Base types and composition for competition-specific prompts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompetitionPrompt:
    """Domain-specific prompt for a competition type.

    Each competition module (f1.py, nfl.py, etc.) exports one of these.
    The registry auto-detects which one to use from market data.
    """

    competition: str                           # e.g. "f1", "nfl"
    display_name: str                          # e.g. "Formula 1"
    domain_rules: str                          # Logical rules for this sport
    constraint_examples: str                   # Concrete constraint examples
    structural_hints: str                      # Known structural relationships
    detection_keywords: tuple[str, ...] = ()   # For auto-detecting from market text


# ── Generic constraint type definitions (shared across all) ──

BASE_CONSTRAINT_TYPES = """## CONSTRAINT TYPES TO DETECT:

### 1. IMPLIES (A → B)
- If A is true, B must also be true
- Constraint: P(A) ≤ P(B)
- Example: "Team wins championship" IMPLIES "Team made playoffs"

### 2. PREREQUISITE (B requires A)
- B cannot happen without A happening first
- Constraint: P(B) ≤ P(A)
- Example: "Wins finals" REQUIRES "Wins semifinals"

### 3. MUTUALLY_EXCLUSIVE (A XOR B)
- A and B cannot both be true
- Constraint: P(A) + P(B) ≤ 1
- Example: "Team A wins" and "Team B wins" cannot both happen

### 4. INCOMPATIBLE (structural impossibility)
- Events are structurally impossible together
- Constraint: P(A) + P(B) ≤ 1
- Example: Two teams in the same bracket cannot BOTH win"""


def compose_system_prompt(competition: CompetitionPrompt | None = None) -> str:
    """Compose the full system prompt from base + competition-specific parts.

    If no competition is provided, returns the generic prompt.
    """
    parts = [
        "Identify LOGICAL CONSTRAINTS between prediction markets for combinatorial arbitrage.",
        "",
        BASE_CONSTRAINT_TYPES,
    ]

    if competition:
        parts.extend([
            "",
            f"## DOMAIN RULES: {competition.display_name}",
            "",
            competition.domain_rules,
            "",
            "## CONCRETE EXAMPLES",
            "",
            competition.constraint_examples,
        ])

        if competition.structural_hints:
            parts.extend([
                "",
                "## STRUCTURAL KNOWLEDGE",
                "",
                competition.structural_hints,
            ])

    parts.extend([
        "",
        "IMPORTANT: Binary Yes/No markets CAN have logical relationships with OTHER markets.",
        "Focus on cross-market implications, not internal Yes/No structure.",
        "Return high-confidence constraints only. Valid JSON only.",
    ])

    return "\n".join(parts)
