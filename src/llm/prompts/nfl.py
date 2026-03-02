"""NFL domain prompt — stub for user to flesh out."""

from .base import CompetitionPrompt

NFL_PROMPT = CompetitionPrompt(
    competition="nfl",
    display_name="NFL",
    detection_keywords=(
        "nfl", "super bowl", "touchdown", "quarterback",
        "afc", "nfc", "playoffs",
        "chiefs", "eagles", "49ers", "bills", "ravens",
        "cowboys", "packers", "lions",
    ),
    domain_rules="""\
NFL has a single-elimination playoff bracket after a regular season.

### Key Logical Rules
- IMPLIES: Winning the Super Bowl IMPLIES winning the conference championship.
- PREREQUISITE: Winning conference championship REQUIRES making playoffs.
- MUTUALLY_EXCLUSIVE: Only one team wins the Super Bowl.
  AFC and NFC champions are from different conferences.
- Division winners auto-qualify for playoffs. Wild card spots are limited.\
""",
    constraint_examples="""\
- "Chiefs win Super Bowl" IMPLIES "Chiefs win AFC Championship" (confidence: 1.0)
- "Chiefs win AFC" MUTUALLY_EXCLUSIVE with "Bills win AFC" (confidence: 1.0)
- "Eagles win Super Bowl" PREREQUISITE "Eagles make playoffs" (confidence: 1.0)\
""",
    structural_hints="",  # User to add conference/division structure
)
