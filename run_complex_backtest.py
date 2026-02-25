"""Run backtest with COMPLEX CONSTRAINTS to exercise the Frank-Wolfe solver.

Unlike partition constraints (sum = 1) which can be checked algebraically,
this backtest uses complex logical constraints (implies, prerequisite, etc.)
that REQUIRE the solver to evaluate.

The key difference from run_fixed_backtest.py:
- is_partition=False on all clusters -> forces solver path
- Uses implies/prerequisite constraints -> solver must iterate
- DEBUG logging enabled -> shows solver iterations
"""
import sys
sys.path.insert(0, "/root/combarbbot")

import logging

# Enable DEBUG logging for optimizer to see solver iterations
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Set specific loggers
logging.getLogger("src.optimizer.frank_wolfe").setLevel(logging.DEBUG)
logging.getLogger("src.optimizer.lmo").setLevel(logging.DEBUG)
logging.getLogger("src.backtest.simulator").setLevel(logging.DEBUG)

# Reduce noise from other modules
logging.getLogger("src.data").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from src.data import MarketLoader
from src.backtest import run_backtest, print_report
# Use optimizer schema to avoid type mismatch with simulator
from src.optimizer.schema import RelationshipGraph, MarketCluster, MarketRelationship

DATA_DIR = "/root/prediction-market-analysis/data/polymarket"

print("=" * 70)
print("COMPLEX CONSTRAINT BACKTEST (Frank-Wolfe Solver)")
print("=" * 70)
print()
print("This backtest uses COMPLEX LOGICAL CONSTRAINTS that require the solver:")
print("- MUTUALLY_EXCLUSIVE: P(A) + P(B) <= 1 (NOT = 1)")
print("- IMPLIES: P(A) <= P(B)")
print("- is_partition=False -> FORCES SOLVER PATH")
print()

market_loader = MarketLoader(DATA_DIR)

# =============================================================================
# Test Case 1: Mutual Exclusivity (Trump vs Harris)
# Both markets created same day, have overlapping trade data
# =============================================================================
print("-" * 70)
print("TEST CASE 1: Mutual Exclusivity Constraint")
print("-" * 70)
print()

trump_id = "253591"     # Trump wins
harris_id = "253597"    # Harris wins

trump_m = market_loader.get_market(trump_id)
harris_m = market_loader.get_market(harris_id)

if trump_m and harris_m:
    print("Market A: {} - {}".format(trump_id, trump_m.question))
    print("Market B: {} - {}".format(harris_id, harris_m.question))
    print()
    print("Logical relationship: MUTUALLY_EXCLUSIVE")
    print("Both cannot win, but sum doesn't have to equal 1 (other candidates)")
    print("Constraint: P(Trump) + P(Harris) <= 1 (NOT = 1)")
    print()

    # Create cluster with MUTUALLY_EXCLUSIVE constraint
    mutex_cluster = MarketCluster(
        cluster_id="trump-harris-exclusive",
        market_ids=[trump_id, harris_id],
        relationships=[
            MarketRelationship(
                type="mutually_exclusive",
                from_market=trump_id,
                to_market=harris_id,
                confidence=1.0,
            ),
        ],
        is_partition=False,  # CRITICAL: Forces solver path
    )

    mutex_graph = RelationshipGraph(clusters=[mutex_cluster])

    print("Running backtest with MUTUALLY_EXCLUSIVE constraint...")
    print("Expected: Solver should be called (look for [FW] log entries)")
    print()

    report1 = run_backtest(
        market_ids=[trump_id, harris_id],
        relationship_graph=mutex_graph,
        data_dir=DATA_DIR,
        kl_threshold=0.001,
        transaction_cost=0.015,
        max_ticks=5000,
    )

    print()
    print("=== MUTUAL EXCLUSIVITY RESULTS ===")
    print_report(report1)

    if report1.opportunities:
        print()
        print("Top Opportunities (verifying solver was used):")
        for idx, opp in enumerate(report1.opportunities[:3]):
            i = idx + 1
            print("  #{}: Block {}, method={}".format(i, opp.block_number, opp.detection_method))
            if opp.solver_result:
                print("      Solver iterations: {}".format(opp.solver_result.iterations))
                print("      KL divergence: {:.6f}".format(opp.solver_result.kl_divergence))

# =============================================================================
# Test Case 2: Three markets with multiple constraints
# =============================================================================
print()
print("-" * 70)
print("TEST CASE 2: Three Markets with Multiple Constraints")
print("-" * 70)
print()

# Trump wins, Harris wins, and Other candidate
trump_id = "253591"
harris_id = "253597"
other_id = "253641"  # Other Democratic politician

trump_m = market_loader.get_market(trump_id)
harris_m = market_loader.get_market(harris_id)
other_m = market_loader.get_market(other_id)

if trump_m and harris_m and other_m:
    print("Market A: {} - {}".format(trump_id, trump_m.question))
    print("Market B: {} - {}".format(harris_id, harris_m.question))
    print("Market C: {} - {}".format(other_id, other_m.question))
    print()
    print("Constraints:")
    print("  1. Trump vs Harris: P(A) + P(B) <= 1")
    print("  2. Trump vs Other: P(A) + P(C) <= 1")
    print("  3. Harris vs Other: P(B) + P(C) <= 1")
    print()

    # Create cluster with multiple pairwise mutual exclusivity
    multi_cluster = MarketCluster(
        cluster_id="three-way-election",
        market_ids=[trump_id, harris_id, other_id],
        relationships=[
            MarketRelationship(
                type="mutually_exclusive",
                from_market=trump_id,
                to_market=harris_id,
                confidence=1.0,
            ),
            MarketRelationship(
                type="mutually_exclusive",
                from_market=trump_id,
                to_market=other_id,
                confidence=1.0,
            ),
            MarketRelationship(
                type="mutually_exclusive",
                from_market=harris_id,
                to_market=other_id,
                confidence=1.0,
            ),
        ],
        is_partition=False,  # NOT a partition - no exhaustive constraint
    )

    multi_graph = RelationshipGraph(clusters=[multi_cluster])

    print("Running backtest with 3-market mutual exclusivity...")
    print()

    report2 = run_backtest(
        market_ids=[trump_id, harris_id, other_id],
        relationship_graph=multi_graph,
        data_dir=DATA_DIR,
        kl_threshold=0.001,
        transaction_cost=0.015,
        max_ticks=10000,
    )

    print()
    print("=== THREE-MARKET RESULTS ===")
    print_report(report2)

    if report2.opportunities:
        print()
        print("Detailed Opportunities:")
        for idx, opp in enumerate(report2.opportunities[:5]):
            i = idx + 1
            print()
            print("#{}: Block {}".format(i, opp.block_number))
            print("    Method: {}".format(opp.detection_method))
            print("    Constraint: {}".format(opp.trade.constraint_type))
            print("    Locked profit: ${:.4f}".format(opp.trade.locked_profit))
            if opp.solver_result:
                print("    Solver iterations: {}".format(opp.solver_result.iterations))
                print("    Converged: {}".format(opp.solver_result.converged))

# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 70)
print("COMPLEX CONSTRAINT BACKTEST SUMMARY")
print("=" * 70)
print()
print("This backtest used complex logical constraints requiring the solver:")
print("  - MUTUALLY_EXCLUSIVE: P(A) + P(B) <= 1")
print("  - is_partition=False on all clusters")
print()
print("Look for '[FW]' and '[SIM] SOLVER' log entries to verify solver usage.")
