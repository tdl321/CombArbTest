"""NBA domain prompt — stub for user to flesh out."""

from .base import CompetitionPrompt

NBA_PROMPT = CompetitionPrompt(
    competition="nba",
    display_name="NBA",
    detection_keywords=(
        "nba", "basketball", "finals", "mvp",
        "eastern conference", "western conference",
        "celtics", "lakers", "nuggets", "bucks",
        "warriors", "76ers", "knicks", "heat",
    ),
    domain_rules="""\
NBA has a best-of-7 playoff bracket after an 82-game regular season.

### Key Logical Rules
- IMPLIES: Winning the NBA Finals IMPLIES winning the conference finals.
- PREREQUISITE: Conference finals REQUIRES making the playoffs (top 10 per conf via play-in).
- MUTUALLY_EXCLUSIVE: Only one team wins the championship.
  Eastern and Western conference champions are from different conferences.
- MVP: Regular season award — independent of playoff success.\
""",
    constraint_examples="""\
- "Celtics win Finals" IMPLIES "Celtics win Eastern Conference" (confidence: 1.0)
- "Celtics win East" MUTUALLY_EXCLUSIVE with "Bucks win East" (confidence: 1.0)
- "Nuggets win Finals" PREREQUISITE "Nuggets make playoffs" (confidence: 1.0)\
""",
    structural_hints="",  # User to add conference/division structure
)
