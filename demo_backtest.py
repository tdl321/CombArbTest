#!/usr/bin/env python3
"""Demo script for running a full backtest with report.

This script demonstrates the complete backtest pipeline:
1. Load markets from Polymarket data
2. Create synthetic relationships (or use LLM)
3. Run walk-forward simulation
4. Generate and print full report
"""

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.data import MarketLoader
from src.backtest import (
    run_backtest_with_synthetic_relationships,
    print_report,
    format_report,
)

DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


def main():
    """Run a demo backtest."""
    print("=" * 70)
    print("COMBINATORIAL ARBITRAGE BACKTESTER - DEMO")
    print("=" * 70)
    
    # Load high-volume markets
    market_loader = MarketLoader(DATA_DIR)
    markets_df = market_loader.query_markets(min_volume=100_000_000, limit=10)
    
    print(f"\nFound {len(markets_df)} high-volume markets (>$100M volume)")
    
    # Show markets
    print("\nMarkets to analyze:")
    market_ids = []
    for row in markets_df.iter_rows(named=True):
        market = market_loader.get_market(row["id"])
        if market and market.clob_token_ids:
            market_ids.append(market.id)
            print(f"  {market.id}: {market.question[:60]}...")
            print(f"       Volume: ${market.volume:,.0f}")
    
    if len(market_ids) < 2:
        print("ERROR: Need at least 2 markets")
        return
    
    # Use Trump win and Harris win markets - should be mutually exclusive
    # Trump: 253591, Harris: 253597
    test_markets = [market_ids[0], market_ids[1]]  # Trump, Harris
    print(f"\nUsing {len(test_markets)} markets for backtest")
    print(f"  {test_markets[0]}: Trump wins")
    print(f"  {test_markets[1]}: Harris wins")
    
    # Test with IMPLIES relationship
    # If Trump wins the election, then Trump will be inaugurated
    # This is a valid logical constraint - the markets should respect this
    
    # But let's use trump/harris with an implies relationship
    # to demonstrate detection when prices violate constraints
    relationships = [
        # Testing: "If Trump wins, Harris cannot win" -> implies with negation
        # For simplicity, we use implies: If Trump > 0.5, Harris should be < 0.5
        # Actually let's use A implies B where A=Trump, B=Trump (identity)
        # That won't find anything interesting
        
        # Better test: pretend trump winning IMPLIES harris winning (nonsensical but will find divergence)
        ("implies", test_markets[0], test_markets[1], 0.9),
    ]
    
    print(f"\nRelationships defined (intentionally wrong to find arbitrage):")
    print(f"  implies: Trump wins -> Harris wins (confidence: 0.9)")
    print(f"  This is logically WRONG, so we expect to find 'arbitrage'")
    print(f"  when Trump is priced higher than Harris")
    
    # Run backtest
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST...")
    print("=" * 70)
    
    report = run_backtest_with_synthetic_relationships(
        market_ids=test_markets,
        relationships=relationships,
        data_dir=DATA_DIR,
        max_ticks=1000,  # Limited for demo
        progress_interval=200,
        kl_threshold=0.001,  # Low threshold to catch more opportunities
        transaction_cost=0.015,  # 1.5% round-trip
    )
    
    # Print full report
    print("\n")
    print_report(report)
    
    # Additional analysis
    if report.opportunities:
        print("\n" + "=" * 70)
        print("SAMPLE OPPORTUNITIES (First 5)")
        print("=" * 70)
        
        for i, opp in enumerate(report.opportunities[:5]):
            print(f"\n--- Opportunity {i+1} ---")
            print(f"  Time: {opp.timestamp}")
            print(f"  Block: {opp.position[0]}")
            print(f"  KL Divergence: {opp.kl_divergence:.6f}")
            print(f"  Gross Profit: ${opp.theoretical_profit:.4f}")
            print(f"  Net Profit: ${opp.net_profit:.4f}")
            print(f"  Market Prices:")
            for mid, price in opp.market_prices.items():
                coherent = opp.coherent_prices.get(mid, price)
                direction = opp.trade_direction.get(mid, "HOLD")
                print(f"    {mid}: {price:.4f} -> {coherent:.4f} ({direction})")
            if opp.constraints_violated:
                print(f"  Constraints Violated: {opp.constraints_violated}")
    
    return report


if __name__ == "__main__":
    main()
