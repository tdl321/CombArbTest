"""FIFA World Cup domain prompt — stub for user to flesh out."""

from .base import CompetitionPrompt

WORLD_CUP_PROMPT = CompetitionPrompt(
    competition="world_cup",
    display_name="FIFA World Cup",
    detection_keywords=(
        "world cup", "fifa", "group stage",
        "brazil", "argentina", "france", "germany",
        "england", "spain", "portugal",
    ),
    domain_rules="""\
World Cup has a group stage followed by single-elimination knockout rounds.

### Key Logical Rules
- IMPLIES: Winning the World Cup IMPLIES winning the semifinal, quarterfinal, and advancing from group.
- PREREQUISITE: Knockout stage REQUIRES finishing top 2 in group.
- MUTUALLY_EXCLUSIVE: Only one team wins the World Cup. Two teams in the same group
  cannot BOTH finish first in that group.
- GROUP CONSTRAINTS: Each group has exactly 1 winner and 1 runner-up advancing.\
""",
    constraint_examples="""\
- "France wins World Cup" IMPLIES "France wins semifinal" (confidence: 1.0)
- "Brazil wins Group G" MUTUALLY_EXCLUSIVE with "Argentina wins Group G" — if same group (confidence: 1.0)
- "Germany wins World Cup" PREREQUISITE "Germany advances from group stage" (confidence: 1.0)\
""",
    structural_hints="",  # User to add group draw when available
)
