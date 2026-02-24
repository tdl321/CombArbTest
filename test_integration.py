#!/usr/bin/env python3
"""Integration test showing optimizer usage with LLM interface.

This demonstrates how the optimizer module connects with the LLM module
to find arbitrage opportunities in prediction markets.
"""

import sys
sys.path.insert(0, "/root/combarbbot")

from src.optimizer import (
    find_arbitrage,
    MarketRelationship,
    MarketCluster,
    RelationshipGraph,
    OptimizationConfig,
)


def test_election_scenario():
    """Test a realistic election prediction market scenario."""
    print("=" * 70)
    print("Election Market Arbitrage Detection")
    print("=" * 70)
    
    # Simulated output from LLM module:
    # The LLM has identified that "Trump wins" implies "Republican wins"
    # and "DeSantis wins" implies "Republican wins"
    # Also, Trump and DeSantis are mutually exclusive (can't both win)
    
    relationships = RelationshipGraph(
        clusters=[
            MarketCluster(
                cluster_id="republican_primary",
                market_ids=[
                    "trump_wins",
                    "desantis_wins",
                    "republican_wins",
                ],
                relationships=[
                    # If Trump wins, Republicans win
                    MarketRelationship(
                        type="implies",
                        from_market="trump_wins",
                        to_market="republican_wins",
                        confidence=1.0,
                    ),
                    # If DeSantis wins, Republicans win
                    MarketRelationship(
                        type="implies",
                        from_market="desantis_wins",
                        to_market="republican_wins",
                        confidence=1.0,
                    ),
                    # Trump and DeSantis can't both win
                    MarketRelationship(
                        type="mutually_exclusive",
                        from_market="trump_wins",
                        to_market="desantis_wins",
                        confidence=1.0,
                    ),
                ],
            )
        ]
    )
    
    # Current market prices (simulated from Polymarket)
    # This contains arbitrage: Trump (0.45) + DeSantis (0.15) = 0.60
    # but Republican wins is only 0.55, which violates the implication
    market_prices = {
        "trump_wins": 0.45,
        "desantis_wins": 0.15,
        "republican_wins": 0.55,
    }
    
    print("\nMarket Prices:")
    for market, price in market_prices.items():
        print("  %s: %.2f" % (market, price))
    
    print("\nRelationships Detected by LLM:")
    print("  - trump_wins -> republican_wins (implies)")
    print("  - desantis_wins -> republican_wins (implies)")
    print("  - trump_wins, desantis_wins (mutually exclusive)")
    
    # Run optimizer
    config = OptimizationConfig(verbose=False)
    result = find_arbitrage(market_prices, relationships, config)
    
    print("\n" + "-" * 70)
    print("Optimization Result:")
    print("-" * 70)
    
    print("\nCoherent (Arbitrage-Free) Prices:")
    for market, price in result.coherent_prices.items():
        original = market_prices[market]
        diff = price - original
        print("  %s: %.4f (was %.2f, %+.4f)" % (market, price, original, diff))
    
    print("\nMetrics:")
    print("  KL Divergence: %.6f" % result.kl_divergence)
    print("  Converged: %s" % result.converged)
    print("  Iterations: %d" % result.iterations)
    print("  Has Arbitrage: %s" % result.has_arbitrage)
    
    if result.constraints_violated:
        print("\nConstraints Violated by Original Prices:")
        for cv in result.constraints_violated:
            print("  - %s (by %.4f)" % (cv.description, cv.violation_amount))
    
    # Verify constraints are now satisfied
    print("\nVerification:")
    coherent = result.coherent_prices
    
    # Check implies constraints
    trump_implies_rep = coherent["trump_wins"] <= coherent["republican_wins"] + 1e-4
    print("  trump_wins <= republican_wins: %s (%.4f <= %.4f)" % (
        trump_implies_rep,
        coherent["trump_wins"],
        coherent["republican_wins"]
    ))
    
    desantis_implies_rep = coherent["desantis_wins"] <= coherent["republican_wins"] + 1e-4
    print("  desantis_wins <= republican_wins: %s (%.4f <= %.4f)" % (
        desantis_implies_rep,
        coherent["desantis_wins"],
        coherent["republican_wins"]
    ))
    
    # Check mutex
    mutex_satisfied = coherent["trump_wins"] + coherent["desantis_wins"] <= 1.0 + 1e-4
    print("  trump_wins + desantis_wins <= 1: %s (%.4f)" % (
        mutex_satisfied,
        coherent["trump_wins"] + coherent["desantis_wins"]
    ))
    
    all_satisfied = trump_implies_rep and desantis_implies_rep and mutex_satisfied
    print("\nAll constraints satisfied: %s" % all_satisfied)
    
    return result


def test_sports_scenario():
    """Test a sports betting scenario with bracket implications."""
    print("\n" + "=" * 70)
    print("Sports Tournament Arbitrage Detection")
    print("=" * 70)
    
    # Teams in a bracket: if A beats B in round 1, A advances
    # If A advances, A could win the whole thing
    
    relationships = RelationshipGraph(
        clusters=[
            MarketCluster(
                cluster_id="tournament",
                market_ids=[
                    "team_a_wins_r1",
                    "team_b_wins_r1",
                    "team_a_wins_tournament",
                ],
                relationships=[
                    # A and B can't both win round 1
                    MarketRelationship(
                        type="mutually_exclusive",
                        from_market="team_a_wins_r1",
                        to_market="team_b_wins_r1",
                        confidence=1.0,
                    ),
                    # If A wins tournament, A must have won round 1
                    MarketRelationship(
                        type="implies",
                        from_market="team_a_wins_tournament",
                        to_market="team_a_wins_r1",
                        confidence=1.0,
                    ),
                ],
            )
        ]
    )
    
    # Arbitrage: tournament win priced higher than round 1 win
    market_prices = {
        "team_a_wins_r1": 0.60,
        "team_b_wins_r1": 0.55,  # Sum > 1 (mutex violation)
        "team_a_wins_tournament": 0.65,  # > r1 win (implies violation)
    }
    
    print("\nMarket Prices:")
    for market, price in market_prices.items():
        print("  %s: %.2f" % (market, price))
    
    result = find_arbitrage(market_prices, relationships)
    
    print("\nCoherent Prices:")
    for market, price in result.coherent_prices.items():
        original = market_prices[market]
        print("  %s: %.4f (was %.2f)" % (market, price, original))
    
    print("\nArbitrage Detected: %s" % result.has_arbitrage)
    print("KL Divergence: %.6f" % result.kl_divergence)
    
    return result


if __name__ == "__main__":
    result1 = test_election_scenario()
    result2 = test_sports_scenario()
    
    print("\n" + "=" * 70)
    print("Integration Tests Complete!")
    print("=" * 70)
