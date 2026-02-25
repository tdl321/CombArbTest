#!/usr/bin/env python3
"""Full deep-dive backtest."""

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.data import MarketLoader
from src.backtest import run_backtest, print_report
from src.llm import build_relationship_graph, MarketInfo

DATA_DIR = "/root/prediction-market-analysis/data/polymarket"

print("=" * 70)
print("DEEP-DIVE COMBINATORIAL ARBITRAGE BACKTEST")
print("=" * 70)

# Focus on politically related markets that SHOULD have strong relationships
market_loader = MarketLoader(DATA_DIR)

# Get high-volume markets
print("\nSearching for high-volume markets...")
markets_df = market_loader.query_markets(min_volume=100_000_000, limit=30)

market_ids = []
market_questions = {}
for row in markets_df.iter_rows(named=True):
    market = market_loader.get_market(row["id"])
    if market and market.clob_token_ids:
        market_ids.append(market.id)
        market_questions[market.id] = market.question
        print(f"  {market.id[:12]}... | Vol: ${market.volume/1e6:.1f}M | {market.question[:55]}...")

print(f"\n{len(market_ids)} markets loaded")

# First, let's see what relationships the LLM discovers
print("\n" + "=" * 70)
print("LLM RELATIONSHIP DISCOVERY")
print("=" * 70)

market_infos = []
for mid in market_ids:
    market = market_loader.get_market(mid)
    if market:
        market_infos.append(MarketInfo(
            id=mid,
            question=market.question,
            outcomes=market.outcomes or ["Yes", "No"]
        ))

graph = build_relationship_graph(market_infos)

print(f"\nClusters found: {len(graph.clusters)}")
total_rels = 0
for cluster in graph.clusters:
    rel_count = len(cluster.relationships) if cluster.relationships else 0
    total_rels += rel_count
    print(f"\n  Cluster: {cluster.cluster_id}")
    print(f"  Theme: {cluster.theme}")
    print(f"  Markets: {len(cluster.market_ids)}")
    for mid in cluster.market_ids[:3]:
        q = market_questions.get(mid, "Unknown")[:50]
        print(f"    - {q}...")
    if len(cluster.market_ids) > 3:
        print(f"    ... and {len(cluster.market_ids) - 3} more")
    print(f"  Relationships: {rel_count}")
    if cluster.relationships:
        for rel in cluster.relationships[:3]:
            print(f"    - {rel.type}: {rel.from_market[:12]} -> {rel.to_market[:12]} (conf: {rel.confidence:.2f})")
        if len(cluster.relationships) > 3:
            print(f"    ... and {len(cluster.relationships) - 3} more")

print(f"\nTotal relationships discovered: {total_rels}")

# Run the backtest with very low threshold
print("\n" + "=" * 70)
print("RUNNING DEEP BACKTEST")
print("KL Threshold: 0.0001 (very sensitive)")
print("Max ticks: 100,000")
print("=" * 70)

report = run_backtest(
    market_ids=market_ids,
    data_dir=DATA_DIR,
    relationship_graph=graph,  # Use the graph we already built
    kl_threshold=0.0001,  # Very low threshold
    transaction_cost=0.015,
    max_ticks=100000,
    progress_interval=10000,
    store_all_opportunities=True,
)

print("\n")
print_report(report)

# Detailed opportunity analysis
if report.opportunities:
    print("\n" + "=" * 70)
    print(f"TOP {min(15, len(report.opportunities))} ARBITRAGE OPPORTUNITIES")
    print("=" * 70)

    sorted_opps = sorted(report.opportunities, key=lambda x: x.kl_divergence, reverse=True)

    for i, opp in enumerate(sorted_opps[:15]):
        print(f"\n--- Opportunity {i+1} (KL: {opp.kl_divergence:.6f}) ---")
        print(f"  Cluster: {opp.cluster_id}")
        print(f"  Block: {opp.position[0]}")
        print(f"  Gross: ${opp.theoretical_profit:.4f} | Net: ${opp.net_profit:.4f}")
        for mid, price in opp.market_prices.items():
            coherent = opp.coherent_prices.get(mid, price)
            direction = opp.trade_direction.get(mid, "HOLD")
            q = market_questions.get(mid, "")[:30]
            print(f"    {q}: {price:.3f} -> {coherent:.3f} [{direction}]")
else:
    print("\nNo opportunities found. Markets appear efficiently priced.")
    print("This suggests the relationship constraints are satisfied by market prices.")

print("\n" + "=" * 70)
print("BACKTEST COMPLETE")
print("=" * 70)
