"""Formula 1 domain prompt.

Replace domain_rules, constraint_examples, and structural_hints
with a researched draft to improve constraint accuracy.
"""

from .base import CompetitionPrompt

F1_PROMPT = CompetitionPrompt(
    competition="f1",
    display_name="Formula 1",
    detection_keywords=(
        "f1", "formula 1", "formula one",
        "drivers' championship", "constructors' championship",
        "grand prix", "pole position",
        "verstappen", "hamilton", "leclerc", "norris", "sainz",
        "red bull", "ferrari", "mclaren", "mercedes",
    ),
    domain_rules="""\
F1 has two parallel championships resolved at season end:

1. DRIVERS' CHAMPIONSHIP — individual drivers compete for points.
2. CONSTRUCTORS' CHAMPIONSHIP — teams score the SUM of their drivers' points.

### Key Logical Rules

- IMPLIES (driver → constructor): If a driver wins the Drivers' Championship,
  their constructor MUST finish high in the Constructors' Championship
  (extremely likely top 2, guaranteed top half).

- PREREQUISITE (constructor competitiveness → driver viability): A driver
  on an uncompetitive constructor cannot realistically win the championship.
  If a constructor is priced very low, its drivers should be priced ≤ that.

- MUTUALLY_EXCLUSIVE: Only one driver can win the Drivers' Championship.
  Only one constructor can win the Constructors' Championship.

- TEAMMATE COUPLING: Two drivers on the same team contribute to the SAME
  constructor score. P(Constructor wins) ≥ max(P(Driver A wins), P(Driver B wins))
  for that team's drivers.

- INCOMPATIBLE: Drivers on different teams are implicitly competing —
  if Driver A on Team X wins, then NO driver on Team Y wins the championship
  (mutually exclusive by definition). But also: Team Y is less likely to win
  constructors if their best driver lost.\
""",
    constraint_examples="""\
- "Max Verstappen wins Drivers' Championship" IMPLIES "Red Bull wins Constructors'"
  with high probability (confidence: 0.85) — not 1.0 because the 2nd Red Bull
  driver may underperform vs rival teams' combined points.

- "Lewis Hamilton wins Drivers' Championship" IMPLIES "Ferrari finishes top 2
  in Constructors'" (confidence: 0.90).

- "McLaren wins Constructors'" PREREQUISITE for "Lando Norris wins Drivers'"
  is NOT strictly true (a driver can win WDC while team loses WCC), but
  P(Norris WDC) ≤ P(McLaren top-2 WCC) is a soft constraint (confidence: 0.75).

- "Verstappen wins WDC" MUTUALLY_EXCLUSIVE with "Norris wins WDC" (confidence: 1.0).

- If both McLaren drivers are priced at 0.20 each for WDC, McLaren WCC
  should be priced ≥ 0.20 (teammate coupling floor).\
""",
    structural_hints="""\
### 2026 Team–Driver Pairings (update each season)
- Red Bull: Max Verstappen, Liam Lawson
- Ferrari: Lewis Hamilton, Charles Leclerc
- McLaren: Lando Norris, Oscar Piastri
- Mercedes: George Russell, Andrea Kimi Antonelli
- Aston Martin: Fernando Alonso, Lance Stroll
- Alpine: Pierre Gasly, Jack Doohan
- Williams: Carlos Sainz, Alexander Albon
- RB (VCARB): Yuki Tsunoda, Isack Hadjar
- Haas: Esteban Ocon, Oliver Bearman
- Sauber/Audi: Nico Hulkenberg, Gabriel Bortoleto

### Constraint Direction
- driver_wins_wdc → constructor_top2_wcc (IMPLIES)
- constructor_wins_wcc → at_least_one_driver_top5 (IMPLIES)
- same_team_drivers: MUTUALLY_EXCLUSIVE for WDC, but COUPLED for WCC\
""",
)
