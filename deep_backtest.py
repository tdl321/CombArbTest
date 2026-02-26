#!/usr/bin/env python3
"""Deep Combinatorial Arbitrage Backtest.

Supports both LLM-discovered relationships and curated tournament partitions.
"""

import sys
sys.path.insert(0, "/root/combarbbot")

import argparse
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
from src.optimizer.schema import RelationshipGraph, MarketCluster, MarketRelationship

DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


# =============================================================================
# CURATED TOURNAMENTS (known partition constraints)
# =============================================================================

NBA_MVP_2023_24_IDS = ["500334", "500332", "500333", "500336", "500335", "500337"]

SUPER_BOWL_2025_TOP10_IDS = [
    "503313",  # Chiefs
    "503324",  # 49ers
    "503322",  # Eagles
    "503307",  # Lions
    "503299",  # Ravens
    "503300",  # Bills
    "503310",  # Texans
    "503309",  # Packers
    "503328",  # Commanders
    "503305",  # Cowboys
]


def build_partition_graph(market_ids: list[str], name: str) -> RelationshipGraph:
    """Build partition constraint graph (mutually exclusive markets)."""
    relationships = []
    for i, m1 in enumerate(market_ids):
        for m2 in market_ids[i+1:]:
            relationships.append(MarketRelationship(
                type="mutually_exclusive",
                from_market=m1,
                to_market=m2,
                confidence=1.0,
            ))
    
    cluster = MarketCluster(
        cluster_id=name.lower().replace(" ", "-"),
        theme=name,
        market_ids=market_ids,
        is_partition=True,
        relationships=relationships,
    )
    
    return RelationshipGraph(
        clusters=[cluster],
        total_markets=len(market_ids),
        total_relationships=len(relationships),
    )


def run_llm_backtest(min_volume: float, max_markets: int, kl_threshold: float, max_ticks: int):
    """Run backtest with LLM-discovered relationships."""
    print("=" * 70)
    print("DEEP-DIVE COMBINATORIAL ARBITRAGE BACKTEST (LLM Mode)")
    print("=" * 70)

    market_loader = MarketLoader(DATA_DIR)

    print("\nSearching for high-volume markets...")
    markets_df = market_loader.query_markets(min_volume=min_volume, limit=max_markets)

    market_ids = []
    market_questions = {}
    for row in markets_df.iter_rows(named=True):
        market = market_loader.get_market(row["id"])
        if market and market.clob_token_ids:
            market_ids.append(market.id)
            market_questions[market.id] = market.question
            print(f"  {market.id[:12]}... | Vol: ${market.volume/1e6:.1f}M | {market.question[:55]}...")

    print(f"\n{len(market_ids)} markets loaded")

    # Build LLM relationship graph
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
    total_rels = sum(len(c.relationships) if c.relationships else 0 for c in graph.clusters)
    print(f"Total relationships: {total_rels}")

    # Run backtest
    print("\n" + "=" * 70)
    print(f"RUNNING BACKTEST (KL threshold: {kl_threshold})")
    print("=" * 70)

    report = run_backtest(
        market_ids=market_ids,
        data_dir=DATA_DIR,
        relationship_graph=graph,
        kl_threshold=kl_threshold,
        transaction_cost=0.015,
        max_ticks=max_ticks,
        progress_interval=1000,
        store_all_opportunities=True,
    )

    print("\n")
    print_report(report)
    return report


def run_tournament_backtest(tournament: str, kl_threshold: float, max_ticks: int):
    """Run backtest on curated tournament partition."""
    if tournament == "nba-mvp":
        market_ids = NBA_MVP_2023_24_IDS
        name = "NBA MVP 2023-24"
    elif tournament == "super-bowl":
        market_ids = SUPER_BOWL_2025_TOP10_IDS
        name = "Super Bowl 2025 (Top 10)"
    else:
        raise ValueError(f"Unknown tournament: {tournament}")

    print("=" * 70)
    print(f"COMBINATORIAL ARBITRAGE BACKTEST: {name}")
    print("=" * 70)

    market_loader = MarketLoader(DATA_DIR)
    
    # Load markets
    market_questions = {}
    for mid in market_ids:
        market = market_loader.get_market(mid)
        if market:
            market_questions[mid] = market.question
            print(f"  {mid}: {market.question[:60]}...")

    # Build partition graph
    graph = build_partition_graph(market_ids, name)
    print(f"\nPartition constraint: {len(graph.clusters[0].relationships)} mutual exclusions")

    # Run backtest
    print("\n" + "=" * 70)
    print(f"RUNNING BACKTEST (KL threshold: {kl_threshold})")
    print("=" * 70)

    report = run_backtest(
        market_ids=market_ids,
        data_dir=DATA_DIR,
        relationship_graph=graph,
        kl_threshold=kl_threshold,
        transaction_cost=0.015,
        max_ticks=max_ticks,
        progress_interval=1000,
        store_all_opportunities=True,
    )

    print("\n")
    print_report(report)
    
    # Detailed opportunities
    if report.opportunities:
        print("\n" + "=" * 70)
        print(f"TOP {min(10, len(report.opportunities))} ARBITRAGE OPPORTUNITIES")
        print("=" * 70)

        sorted_opps = sorted(report.opportunities, key=lambda x: x.kl_divergence, reverse=True)

        for i, opp in enumerate(sorted_opps[:10]):
            print(f"\n--- Opportunity {i+1} (KL: {opp.kl_divergence:.6f}) ---")
            print(f"  Gross: ${opp.theoretical_profit:.4f} | Net: ${opp.net_profit:.4f}")
            for mid, price in opp.market_prices.items():
                coherent = opp.coherent_prices.get(mid, price)
                direction = opp.trade_direction.get(mid, "HOLD")
                q = market_questions.get(mid, "")[:30]
                print(f"    {q}: {price:.3f} -> {coherent:.3f} [{direction}]")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combinatorial Arbitrage Backtest")
    parser.add_argument("--mode", type=str, default="tournament",
                       choices=["llm", "tournament"],
                       help="Backtest mode: llm (discover relationships) or tournament (use curated)")
    parser.add_argument("--tournament", type=str, default="nba-mvp",
                       choices=["nba-mvp", "super-bowl"],
                       help="Tournament to backtest (if mode=tournament)")
    parser.add_argument("--kl-threshold", type=float, default=0.001,
                       help="KL divergence threshold for arbitrage detection")
    parser.add_argument("--max-ticks", type=int, default=10000,
                       help="Maximum ticks to process")
    parser.add_argument("--min-volume", type=float, default=100_000_000,
                       help="Minimum volume filter (if mode=llm)")
    parser.add_argument("--max-markets", type=int, default=30,
                       help="Max markets to analyze (if mode=llm)")
    
    args = parser.parse_args()

    if args.mode == "llm":
        run_llm_backtest(
            min_volume=args.min_volume,
            max_markets=args.max_markets,
            kl_threshold=args.kl_threshold,
            max_ticks=args.max_ticks,
        )
    else:
        run_tournament_backtest(
            tournament=args.tournament,
            kl_threshold=args.kl_threshold,
            max_ticks=args.max_ticks,
        )

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
