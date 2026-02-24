#!/usr/bin/env python3
"""Integration tests for the Backtest module.

Tests BT-01 through BT-04 with real data and synthetic relationships.
"""

import logging
import sys
from datetime import datetime
from decimal import Decimal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


def test_pnl_calculations():
    """Test PnL calculation functions."""
    from src.backtest.pnl import (
        calculate_theoretical_profit,
        calculate_opportunity_pnl,
        apply_transaction_costs,
    )
    
    print("\n" + "=" * 60)
    print("TEST: PnL Calculations")
    print("=" * 60)
    
    # Test case: market underpriced vs coherent
    market_prices = {"A": 0.4, "B": 0.5}
    coherent_prices = {"A": 0.5, "B": 0.5}
    
    profit, directions = calculate_theoretical_profit(
        market_prices=market_prices,
        coherent_prices=coherent_prices,
        kl_divergence=0.05,
    )
    
    print(f"Market prices:   {market_prices}")
    print(f"Coherent prices: {coherent_prices}")
    print(f"Theoretical profit: ${profit:.4f}")
    print(f"Trade directions: {directions}")
    
    assert profit > 0, "Should have positive profit when prices differ"
    assert directions.get("A") == "BUY", "Should buy underpriced market"
    assert "B" not in directions, "Should not trade when prices equal"
    
    # Test transaction costs
    gross = 0.10
    net = apply_transaction_costs(gross, num_trades=2, transaction_cost_rate=0.015)
    print(f"\nGross profit: ${gross:.4f}")
    print(f"Net profit (after 1.5% x 2 trades): ${net:.4f}")
    
    assert net < gross, "Net should be less than gross"
    assert net == gross - (2 * 0.015), "Net should be gross minus costs"
    
    # Full opportunity PnL
    gross_pnl, net_pnl, dirs = calculate_opportunity_pnl(
        market_prices=market_prices,
        coherent_prices=coherent_prices,
        kl_divergence=0.05,
        transaction_cost_rate=0.015,
    )
    
    print(f"\nFull opportunity PnL:")
    print(f"  Gross: ${gross_pnl:.4f}")
    print(f"  Net:   ${net_pnl:.4f}")
    
    print("\nPnL calculations: PASSED")
    return True


def test_report_generation():
    """Test report generation from opportunities."""
    from src.backtest.schema import ArbitrageOpportunity
    from src.backtest.report import generate_report, format_report
    
    print("\n" + "=" * 60)
    print("TEST: Report Generation")
    print("=" * 60)
    
    # Create some sample opportunities
    opportunities = [
        ArbitrageOpportunity(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            position=(18000000, 1),
            cluster_id="test_cluster",
            market_prices={"A": 0.4, "B": 0.5},
            coherent_prices={"A": 0.5, "B": 0.5},
            kl_divergence=0.05,
            constraints_violated=["implies(A->B)"],
            theoretical_profit=0.1,
            net_profit=0.085,
            trade_direction={"A": "BUY"},
        ),
        ArbitrageOpportunity(
            timestamp=datetime(2024, 1, 1, 11, 0, 0),
            position=(18001000, 5),
            cluster_id="test_cluster",
            market_prices={"A": 0.6, "B": 0.4},
            coherent_prices={"A": 0.5, "B": 0.5},
            kl_divergence=0.08,
            constraints_violated=["implies(A->B)"],
            theoretical_profit=0.2,
            net_profit=0.17,
            trade_direction={"A": "SELL", "B": "BUY"},
        ),
        ArbitrageOpportunity(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            position=(18002000, 10),
            cluster_id="test_cluster",
            market_prices={"A": 0.45, "B": 0.48},
            coherent_prices={"A": 0.5, "B": 0.5},
            kl_divergence=0.02,
            constraints_violated=[],
            theoretical_profit=0.05,
            net_profit=-0.01,  # Loss after costs
            trade_direction={"A": "BUY", "B": "BUY"},
        ),
    ]
    
    report = generate_report(
        opportunities=opportunities,
        start_date=datetime(2024, 1, 1, 10, 0, 0),
        end_date=datetime(2024, 1, 1, 12, 0, 0),
        markets_analyzed=2,
        clusters_found=1,
        cluster_themes={"test_cluster": "Test Theme"},
        cluster_market_ids={"test_cluster": ["A", "B"]},
    )
    
    print(f"Total opportunities: {report.total_opportunities}")
    print(f"Gross PnL: ${report.gross_pnl:.4f}")
    print(f"Net PnL: ${report.net_pnl:.4f}")
    print(f"Win rate: {report.win_rate * 100:.1f}%")
    print(f"Max drawdown: ${report.max_drawdown:.4f}")
    
    assert report.total_opportunities == 3
    assert report.win_count == 2
    assert report.loss_count == 1
    assert report.win_rate == 2/3
    
    # Test formatted output
    formatted = format_report(report)
    print("\nFormatted Report Preview:")
    print(formatted[:500] + "...")
    
    print("\nReport generation: PASSED")
    return True


def test_data_loading():
    """Test that we can load data from the Polymarket dataset."""
    from src.data import MarketLoader, TradeLoader, BlockLoader
    
    print("\n" + "=" * 60)
    print("TEST: Data Loading")
    print("=" * 60)
    
    market_loader = MarketLoader(DATA_DIR)
    block_loader = BlockLoader(DATA_DIR)
    trade_loader = TradeLoader(DATA_DIR, block_loader=block_loader)
    
    # Load some high-volume markets
    markets_df = market_loader.query_markets(min_volume=1_000_000, limit=10)
    print(f"Found {len(markets_df)} markets with volume > $1M")
    
    if markets_df.is_empty():
        print("WARNING: No markets found, skipping data test")
        return True
    
    # Get first market details
    market_id = markets_df.row(0, named=True)["id"]
    market = market_loader.get_market(market_id)
    
    print(f"\nSample market:")
    print(f"  ID: {market.id}")
    print(f"  Question: {market.question[:60]}...")
    print(f"  Volume: ${market.volume:,.0f}")
    print(f"  Outcomes: {market.outcomes}")
    print(f"  Token IDs: {market.clob_token_ids}")
    
    # Load some trades
    if market.clob_token_ids:
        trades_df = trade_loader.query_trades(
            asset_ids=market.clob_token_ids,
            limit=100,
        )
        print(f"\nLoaded {len(trades_df)} trades for this market")
        
        if not trades_df.is_empty():
            # Check block range
            min_block = trades_df["block_number"].min()
            max_block = trades_df["block_number"].max()
            print(f"Block range: {min_block:,} - {max_block:,}")
    
    print("\nData loading: PASSED")
    return True


def test_optimizer_integration():
    """Test that we can call the optimizer with test data."""
    from src.optimizer import (
        find_arbitrage,
        RelationshipGraph,
        MarketCluster,
        MarketRelationship,
    )
    
    print("\n" + "=" * 60)
    print("TEST: Optimizer Integration")
    print("=" * 60)
    
    # Create a simple relationship graph
    relationships = [
        MarketRelationship(
            type="implies",
            from_market="A",
            to_market="B",
            confidence=0.9,
        ),
    ]
    
    cluster = MarketCluster(
        cluster_id="test",
        theme="Test Cluster",
        market_ids=["A", "B"],
        relationships=relationships,
    )
    
    graph = RelationshipGraph(clusters=[cluster])
    
    # Test case 1: Prices violate implication (A > B, but A implies B)
    prices_violation = {"A": 0.7, "B": 0.5}
    result = find_arbitrage(prices_violation, graph)
    
    print(f"Test 1: A implies B violation")
    print(f"  Market prices:   {prices_violation}")
    print(f"  Coherent prices: {result.coherent_prices}")
    print(f"  KL divergence:   {result.kl_divergence:.6f}")
    print(f"  Has arbitrage:   {result.has_arbitrage}")
    print(f"  Converged:       {result.converged}")
    
    # Test case 2: Prices satisfy implication (A < B)
    prices_ok = {"A": 0.4, "B": 0.6}
    result_ok = find_arbitrage(prices_ok, graph)
    
    print(f"\nTest 2: A implies B satisfied")
    print(f"  Market prices:   {prices_ok}")
    print(f"  Coherent prices: {result_ok.coherent_prices}")
    print(f"  KL divergence:   {result_ok.kl_divergence:.6f}")
    print(f"  Has arbitrage:   {result_ok.has_arbitrage}")
    
    print("\nOptimizer integration: PASSED")
    return True


def test_small_backtest():
    """Run a small backtest with real data and synthetic relationships."""
    from src.data import MarketLoader, TradeLoader, BlockLoader
    from src.backtest import (
        run_backtest_with_synthetic_relationships,
        print_report,
    )
    
    print("\n" + "=" * 60)
    print("TEST: Small Backtest with Real Data")
    print("=" * 60)
    
    # Load markets
    market_loader = MarketLoader(DATA_DIR)
    markets_df = market_loader.query_markets(min_volume=1_000_000, limit=5)
    
    if len(markets_df) < 2:
        print("WARNING: Not enough markets for backtest, skipping")
        return True
    
    # Get 2 markets
    market_ids = markets_df["id"].to_list()[:2]
    print(f"Testing with markets: {market_ids}")
    
    # Get market questions for context
    for mid in market_ids:
        market = market_loader.get_market(mid)
        if market:
            print(f"  {mid[:20]}...: {market.question[:50]}...")
    
    # Create a synthetic relationship
    relationships = [
        ("implies", market_ids[0], market_ids[1], 0.8),
    ]
    
    print(f"\nSynthetic relationship: {market_ids[0][:20]} implies {market_ids[1][:20]}")
    
    # Run backtest with limited ticks
    print("\nRunning backtest (max 500 ticks)...")
    try:
        report = run_backtest_with_synthetic_relationships(
            market_ids=market_ids,
            relationships=relationships,
            data_dir=DATA_DIR,
            max_ticks=500,
            progress_interval=100,
            kl_threshold=0.001,  # Lower threshold to find more opportunities
        )
        
        print("\n" + "-" * 40)
        print(f"Backtest Results:")
        print(f"  Opportunities found: {report.total_opportunities}")
        print(f"  Gross PnL: ${report.gross_pnl:.4f}")
        print(f"  Net PnL: ${report.net_pnl:.4f}")
        print(f"  Win rate: {report.win_rate * 100:.1f}%")
        
        if report.total_opportunities > 0:
            print(f"\nSample opportunity:")
            opp = report.opportunities[0]
            print(f"  Block: {opp.position[0]}")
            print(f"  KL divergence: {opp.kl_divergence:.6f}")
            print(f"  Market prices: {opp.market_prices}")
            print(f"  Coherent prices: {opp.coherent_prices}")
        
        print("\nSmall backtest: PASSED")
        return True
        
    except Exception as e:
        print(f"Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_market_iterator():
    """Test the CrossMarketIterator directly."""
    from src.data import (
        MarketLoader,
        TradeLoader, 
        BlockLoader,
        CrossMarketIterator,
    )
    
    print("\n" + "=" * 60)
    print("TEST: Cross-Market Iterator")
    print("=" * 60)
    
    market_loader = MarketLoader(DATA_DIR)
    block_loader = BlockLoader(DATA_DIR)
    trade_loader = TradeLoader(DATA_DIR, block_loader=block_loader)
    
    # Get 2 markets
    markets_df = market_loader.query_markets(min_volume=1_000_000, limit=2)
    if len(markets_df) < 2:
        print("WARNING: Not enough markets, skipping")
        return True
    
    market_ids = markets_df["id"].to_list()[:2]
    print(f"Testing iterator with {len(market_ids)} markets")
    
    iterator = CrossMarketIterator(
        trade_loader=trade_loader,
        block_loader=block_loader,
        market_loader=market_loader,
        market_ids=market_ids,
    )
    
    # Iterate through first 100 snapshots
    count = 0
    snapshots_with_prices = 0
    
    for snapshot in iterator.iter_snapshots(batch_size=200):
        count += 1
        if snapshot.has_all_prices():
            snapshots_with_prices += 1
            if snapshots_with_prices <= 3:
                prices = snapshot.get_prices()
                print(f"  Snapshot {count}: Block {snapshot.position.block_number}, prices: {prices}")
        
        if count >= 100:
            break
    
    print(f"\nProcessed {count} snapshots")
    print(f"Snapshots with all prices: {snapshots_with_prices}")
    
    print("\nCross-market iterator: PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("BACKTEST MODULE INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("PnL Calculations", test_pnl_calculations),
        ("Report Generation", test_report_generation),
        ("Data Loading", test_data_loading),
        ("Optimizer Integration", test_optimizer_integration),
        ("Cross-Market Iterator", test_cross_market_iterator),
        ("Small Backtest", test_small_backtest),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nTEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "PASSED" if p else "FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
