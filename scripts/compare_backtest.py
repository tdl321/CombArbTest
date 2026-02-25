#!/usr/bin/env python3
"""Compare old vs new backtest logic.

Phase 5 Task 5.2: Validate that new arbitrage extraction produces
sensible results compared to the old approach.

The OLD approach: profit = sum(|coherent - market|) per market
The NEW approach: profit = violation_magnitude (constraint-specific)

For partition constraints, these should produce similar results.
For implies/mutex constraints, the new approach is more correct.
"""

import sys
import logging
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_scenarios():
    """Create test scenarios for comparison."""
    return [
        # Scenario 1: Partition underpriced (sum < 1)
        {
            "name": "Partition Underpriced",
            "market_prices": {"A": 0.30, "B": 0.30, "C": 0.30},
            "coherent_prices": {"A": 0.333, "B": 0.333, "C": 0.333},
            "expected_direction": "BUY ALL",
            "expected_profit_approx": 0.10,  # 1.0 - 0.90
        },
        # Scenario 2: Partition overpriced (sum > 1)
        {
            "name": "Partition Overpriced",
            "market_prices": {"A": 0.40, "B": 0.40, "C": 0.30},
            "coherent_prices": {"A": 0.364, "B": 0.364, "C": 0.273},
            "expected_direction": "SELL ALL",
            "expected_profit_approx": 0.10,  # 1.10 - 1.0
        },
        # Scenario 3: Binary underpriced
        {
            "name": "Binary Underpriced",
            "market_prices": {"YES": 0.40, "NO": 0.50},
            "coherent_prices": {"YES": 0.444, "NO": 0.556},
            "expected_direction": "BUY BOTH",
            "expected_profit_approx": 0.10,  # 1.0 - 0.90
        },
        # Scenario 4: Large partition violation
        {
            "name": "Large Partition Violation",
            "market_prices": {"A": 0.20, "B": 0.20, "C": 0.20, "D": 0.20},
            "coherent_prices": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            "expected_direction": "BUY ALL",
            "expected_profit_approx": 0.20,  # 1.0 - 0.80
        },
    ]


def calculate_old_profit(market_prices: dict, coherent_prices: dict) -> tuple:
    """Calculate profit using OLD method (price convergence).
    
    OLD: profit = sum(|coherent[m] - market[m]|) for all m
    """
    profit = 0.0
    directions = {}
    
    for m in market_prices:
        diff = coherent_prices.get(m, market_prices[m]) - market_prices[m]
        profit += abs(diff)
        directions[m] = "BUY" if diff > 0 else "SELL"
    
    return profit, directions


def calculate_new_profit(market_prices: dict) -> tuple:
    """Calculate profit using NEW method (violation magnitude).
    
    NEW: profit = |1 - sum(prices)| for partition
    """
    from src.arbitrage.extractor import ArbitrageExtractor
    from src.optimizer.schema import ArbitrageResult
    
    extractor = ArbitrageExtractor(min_profit_threshold=0.0, fee_per_leg=0.01)
    
    result = ArbitrageResult(
        market_prices=market_prices,
        coherent_prices=market_prices,  # Not used by partition check
        kl_divergence=0.01,
        constraints_violated=[],
        converged=True,
        iterations=10,
    )
    
    trades = extractor.extract_trades(result)
    
    if not trades:
        return 0.0, {}
    
    best = max(trades, key=lambda t: t.locked_profit)
    return best.locked_profit, best.positions


def run_comparison():
    """Run comparison between old and new methods."""
    scenarios = create_test_scenarios()
    
    print("=" * 80)
    print("                 BACKTEST METHOD COMPARISON")
    print("=" * 80)
    print()
    print("Comparing OLD (price-convergence) vs NEW (violation-magnitude) profit calculation")
    print()
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print("-" * 80)
        print("Scenario {}: {}".format(i, scenario["name"]))
        print("-" * 80)
        
        market_prices = scenario["market_prices"]
        coherent_prices = scenario["coherent_prices"]
        
        print("Market prices:   {}".format(market_prices))
        print("Sum of prices:   {:.4f}".format(sum(market_prices.values())))
        print("Coherent prices: {}".format(coherent_prices))
        print()
        
        # OLD method
        old_profit, old_directions = calculate_old_profit(market_prices, coherent_prices)
        
        # NEW method
        new_profit, new_directions = calculate_new_profit(market_prices)
        
        print("OLD method profit:  ${:.4f}  (directions: {})".format(old_profit, old_directions))
        print("NEW method profit:  ${:.4f}  (directions: {})".format(new_profit, new_directions))
        print("Expected profit:    ${:.4f}".format(scenario["expected_profit_approx"]))
        print()
        
        # Analysis
        diff = abs(old_profit - new_profit)
        match_expected = abs(new_profit - scenario["expected_profit_approx"]) < 0.01
        
        if diff < 0.01:
            print("Result: METHODS AGREE (diff={:.4f})".format(diff))
        else:
            print("Result: METHODS DIFFER (diff={:.4f})".format(diff))
            print("        OLD uses price convergence, NEW uses violation magnitude")
        
        if match_expected:
            print("Validation: NEW method matches expected profit")
        else:
            print("Validation: WARNING - NEW method differs from expected")
        
        print()
        
        results.append({
            "scenario": scenario["name"],
            "old_profit": old_profit,
            "new_profit": new_profit,
            "expected": scenario["expected_profit_approx"],
            "diff": diff,
            "new_matches_expected": match_expected,
        })
    
    # Summary
    print("=" * 80)
    print("                           SUMMARY")
    print("=" * 80)
    print()
    header = "{:<25} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "Scenario", "Old", "New", "Expected", "Diff", "Valid")
    print(header)
    print("-" * 80)
    
    total_old = 0
    total_new = 0
    all_valid = True
    
    for r in results:
        valid_str = "YES" if r["new_matches_expected"] else "NO"
        row = "{:<25} ${:>8.4f} ${:>8.4f} ${:>8.4f} ${:>8.4f} {:>8}".format(
            r["scenario"], r["old_profit"], r["new_profit"], 
            r["expected"], r["diff"], valid_str)
        print(row)
        total_old += r["old_profit"]
        total_new += r["new_profit"]
        if not r["new_matches_expected"]:
            all_valid = False
    
    print("-" * 80)
    print("{:<25} ${:>8.4f} ${:>8.4f}".format("TOTAL", total_old, total_new))
    print()
    
    if all_valid:
        print("VALIDATION: All NEW method results match expected profits")
        print("The new arbitrage extraction logic produces correct results.")
    else:
        print("WARNING: Some NEW method results do not match expected values")
    
    print()
    print("KEY INSIGHT:")
    print("- OLD method: profit = sum of per-market adjustments toward coherent")
    print("- NEW method: profit = constraint violation magnitude")
    print("- For partition constraints, both should give similar results")
    print("- The NEW method is mathematically correct for arbitrage profit")


def main():
    print()
    print("=" * 80)
    print("           BACKTEST COMPARISON SCRIPT (Phase 5 Task 5.2)")
    print("=" * 80)
    print()
    
    try:
        run_comparison()
        return 0
    except Exception as e:
        logger.error("Comparison failed: {}".format(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
