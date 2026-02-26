"""Backtest Runner - Entry Point."""

import logging
import time
from datetime import datetime
from typing import Optional

from src.data import MarketLoader, TradeLoader, BlockLoader, CategoryIndex
from src.llm import MarketInfo, build_relationship_graph, RelationshipGraph, MarketCluster, MarketRelationship

from .schema import BacktestConfig, BacktestReport
from .simulator import WalkForwardSimulator
from .report import generate_report, format_report

logger = logging.getLogger(__name__)

from src.config import DEFAULT_DATA_DIR


def run_backtest(
    market_ids: list[str] | None = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    kl_threshold: float = 0.01,
    transaction_cost: float = 0.015,
    data_dir: str = DEFAULT_DATA_DIR,
    relationship_graph: Optional[RelationshipGraph] = None,
    use_llm: bool = False,
    max_ticks: Optional[int] = None,
    progress_interval: int = 1000,
    store_all_opportunities: bool = True,
    category: str | None = None,
    subcategory: str | None = None,
) -> BacktestReport:
    """Run backtest pipeline.

    Markets can be specified by:
    - market_ids: Explicit list of market IDs
    - category: Fetch all markets in a category from DuckDB

    Args:
        market_ids: Explicit market IDs (mutually exclusive with category)
        category: Category filter (e.g., "politics", "sports")
        subcategory: Subcategory filter (requires category)
        start_date: Optional start date filter
        end_date: Optional end date filter
        start_block: Optional start block filter
        end_block: Optional end block filter
        kl_threshold: KL divergence threshold for arbitrage detection
        transaction_cost: Fee per leg of trade
        data_dir: Path to data directory
        relationship_graph: Pre-built relationship graph
        use_llm: Whether to use LLM for relationship extraction
        max_ticks: Maximum ticks to process
        progress_interval: Log progress every N ticks
        store_all_opportunities: Store all opportunities in report
    """
    # Validation
    if market_ids is not None and category is not None:
        raise ValueError("Specify market_ids OR category, not both")

    if subcategory is not None and category is None:
        raise ValueError("subcategory requires category")

    # Resolve from category
    if category is not None:
        idx = CategoryIndex()
        market_ids = idx.query_by_category(category, subcategory, limit=None)
        idx.close()
        if not market_ids:
            raise ValueError(f"No markets found for category={category!r}")
        logger.info("[BACKTEST] Resolved %d markets from category=%s",
                    len(market_ids), category)

    if not market_ids:
        raise ValueError("Must specify market_ids or category")

    logger.info("[BACKTEST] Starting backtest with %d markets", len(market_ids))
    logger.info("[BACKTEST] Parameters: violation_threshold=%.4f, fee_per_leg=%.4f, llm_relationships=%s",
                kl_threshold, transaction_cost, use_llm)
    start_time = time.time()

    # Load data
    logger.info("[BACKTEST] Loading market data from %s", data_dir)
    market_loader = MarketLoader(data_dir)
    block_loader = BlockLoader(data_dir)
    trade_loader = TradeLoader(data_dir, block_loader=block_loader)

    markets = []
    market_info = {}
    for mid in market_ids:
        market = market_loader.get_market(mid)
        if market:
            markets.append(market)
            market_info[mid] = MarketInfo(
                id=market.id,
                question=market.question,
                outcomes=market.outcomes,
            )
        else:
            logger.warning("[BACKTEST] Market not found: %s", mid)

    if len(markets) < 2:
        logger.error("[BACKTEST] Need >= 2 markets, got %d", len(markets))
        raise ValueError("Need >= 2 markets, got %d" % len(markets))

    logger.info("[BACKTEST] Loaded %d markets successfully", len(markets))

    # Build relationship graph
    if relationship_graph is None:
        if use_llm:
            logger.info("[BACKTEST] Building relationships via LLM...")
            relationship_graph = build_relationship_graph(list(market_info.values()))
        else:
            logger.info("[BACKTEST] Using default graph (no constraints)")
            relationship_graph = _default_graph(market_ids)

    logger.info("[BACKTEST] Relationship graph: %d clusters, %d relationships",
                len(relationship_graph.clusters), len(relationship_graph.get_all_relationships()))

    # Config
    config = BacktestConfig(
        market_ids=market_ids,
        start_block=start_block,
        end_block=end_block,
        start_date=start_date,
        end_date=end_date,
        kl_threshold=kl_threshold,
        transaction_cost=transaction_cost,
        max_ticks=max_ticks,
        progress_interval=progress_interval,
        store_all_opportunities=store_all_opportunities,
    )

    # Simulate
    logger.info("[BACKTEST] Running simulation...")
    simulator = WalkForwardSimulator(
        market_loader=market_loader,
        trade_loader=trade_loader,
        block_loader=block_loader,
    )

    opportunities = list(simulator.run(
        market_ids=market_ids,
        relationship_graph=relationship_graph,
        config=config,
    ))

    elapsed = time.time() - start_time
    logger.info("[BACKTEST] Simulation complete: %d opportunities found in %.3fs",
                len(opportunities), elapsed)

    # Report
    if opportunities:
        actual_start = min(o.timestamp for o in opportunities)
        actual_end = max(o.timestamp for o in opportunities)
    else:
        actual_start = start_date or datetime.now()
        actual_end = end_date or datetime.now()

    cluster_themes = {}
    cluster_market_ids = {}
    for c in relationship_graph.clusters:
        theme = getattr(c, "theme", None) or c.cluster_id
        cluster_themes[c.cluster_id] = theme
        cluster_market_ids[c.cluster_id] = c.market_ids

    report = generate_report(
        opportunities=opportunities,
        start_date=actual_start,
        end_date=actual_end,
        markets_analyzed=len(markets),
        clusters_found=len(relationship_graph.clusters),
        cluster_themes=cluster_themes,
        cluster_market_ids=cluster_market_ids,
        kl_threshold=kl_threshold,
        transaction_cost_rate=transaction_cost,
        store_all_opportunities=store_all_opportunities,
    )

    logger.info("[BACKTEST] Report generated: locked_profit=%.4f, net_profit=%.4f, win_rate=%.2f%%",
                report.gross_pnl, report.net_pnl, report.win_rate * 100)

    return report


def _default_graph(market_ids: list[str]) -> RelationshipGraph:
    """Default graph with no constraints."""
    return RelationshipGraph(clusters=[
        MarketCluster(cluster_id="default", market_ids=market_ids, relationships=[])
    ])


def run_backtest_with_synthetic_relationships(
    market_ids: list[str],
    relationships: list[tuple[str, str, str, float]],
    data_dir: str = DEFAULT_DATA_DIR,
    **kwargs,
) -> BacktestReport:
    """Backtest with manual relationships."""
    logger.info("[BACKTEST] Running with %d synthetic relationships", len(relationships))

    rel_objects = [
        MarketRelationship(
            type=rel_type,
            from_market=from_market,
            to_market=to_market,
            confidence=confidence,
        )
        for rel_type, from_market, to_market, confidence in relationships
    ]

    graph = RelationshipGraph(clusters=[
        MarketCluster(cluster_id="synthetic", theme="synthetic", market_ids=market_ids, relationships=rel_objects)
    ])

    return run_backtest(market_ids=market_ids, relationship_graph=graph, data_dir=data_dir, **kwargs)


def print_report(report: BacktestReport) -> None:
    """Print formatted report."""
    print(format_report(report))
