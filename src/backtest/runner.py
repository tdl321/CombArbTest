"""High-Level Backtest Runner (Entry Point).

Provides run_backtest() as the main entry point for running backtests.
"""

from datetime import datetime
from typing import Optional
import logging

from src.data import (
    MarketLoader,
    TradeLoader,
    BlockLoader,
    Market,
)
from src.optimizer import RelationshipGraph, MarketCluster, MarketRelationship
from src.llm import MarketInfo, build_relationship_graph

from .schema import (
    ArbitrageOpportunity,
    BacktestConfig,
    BacktestReport,
)
from .simulator import WalkForwardSimulator
from .report import generate_report, format_report
from .pnl import PnLTracker

logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


def run_backtest(
    market_ids: list[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    kl_threshold: float = 0.01,
    transaction_cost: float = 0.015,
    data_dir: str = DEFAULT_DATA_DIR,
    relationship_graph: Optional[RelationshipGraph] = None,
    use_llm_for_relationships: bool = False,
    max_ticks: Optional[int] = None,
    progress_interval: int = 1000,
    store_all_opportunities: bool = True,
) -> BacktestReport:
    """Run full backtest pipeline.
    
    This is the main entry point for running a backtest. It:
    1. Loads market data
    2. Optionally calls LLM to build relationship graph (or uses provided one)
    3. Runs walk-forward simulation
    4. Generates summary report
    
    Args:
        market_ids: List of market IDs to backtest
        start_date: Start date for backtest (optional, uses blocks if not provided)
        end_date: End date for backtest (optional, uses blocks if not provided)
        start_block: Start block number (alternative to start_date)
        end_block: End block number (alternative to end_date)
        kl_threshold: Minimum KL divergence to flag opportunity (default 0.01)
        transaction_cost: Transaction cost rate (default 1.5%)
        data_dir: Path to Polymarket data directory
        relationship_graph: Pre-computed relationship graph (optional)
        use_llm_for_relationships: Whether to call LLM to build relationships
        max_ticks: Maximum ticks to process (for testing)
        progress_interval: Print progress every N ticks
        store_all_opportunities: Whether to store all opportunities in report
        
    Returns:
        BacktestReport with all statistics and opportunities
    """
    logger.info(f"Starting backtest with {len(market_ids)} markets")
    
    # Initialize data loaders
    market_loader = MarketLoader(data_dir)
    block_loader = BlockLoader(data_dir)
    trade_loader = TradeLoader(data_dir, block_loader=block_loader)
    
    # Load market metadata
    markets = []
    market_info = {}
    for market_id in market_ids:
        market = market_loader.get_market(market_id)
        if market:
            markets.append(market)
            market_info[market_id] = MarketInfo(
                id=market.id,
                question=market.question,
                outcomes=market.outcomes,
            )
        else:
            logger.warning(f"Market not found: {market_id}")
    
    if len(markets) < 2:
        raise ValueError(f"Need at least 2 markets, found {len(markets)}")
    
    logger.info(f"Loaded {len(markets)} markets")
    
    # Get or build relationship graph
    if relationship_graph is None:
        if use_llm_for_relationships:
            logger.info("Building relationship graph using LLM...")
            relationship_graph = build_relationship_graph(
                markets=list(market_info.values()),
            )
            logger.info(
                f"Found {len(relationship_graph.clusters)} clusters, "
                f"{relationship_graph.total_relationships} relationships"
            )
        else:
            # Create a simple default graph with all markets in one cluster
            logger.info("Using default relationship graph (no LLM)")
            relationship_graph = _create_default_graph(market_ids)
    
    # Build config
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
    
    # Run simulation
    logger.info("Running walk-forward simulation...")
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
    
    logger.info(f"Found {len(opportunities)} opportunities")
    
    # Determine actual time range from opportunities
    if opportunities:
        actual_start = min(o.timestamp for o in opportunities)
        actual_end = max(o.timestamp for o in opportunities)
    else:
        actual_start = start_date or datetime.now()
        actual_end = end_date or datetime.now()
    
    # Build cluster info for report - handle both optimizer and LLM schema
    cluster_themes = {}
    cluster_market_ids = {}
    for c in relationship_graph.clusters:
        # optimizer schema doesn't have 'theme', LLM schema does
        theme = getattr(c, 'theme', None) or c.cluster_id
        cluster_themes[c.cluster_id] = theme
        cluster_market_ids[c.cluster_id] = c.market_ids
    
    # Generate report
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
    
    return report


def _create_default_graph(market_ids: list[str]) -> RelationshipGraph:
    """Create a simple default relationship graph.
    
    Without LLM analysis, we create a basic graph with all markets
    in one cluster and no relationships. This will only detect
    arbitrage if markets have explicit violations.
    
    Args:
        market_ids: List of market IDs
        
    Returns:
        RelationshipGraph with default structure
    """
    cluster = MarketCluster(
        cluster_id="default",
        market_ids=market_ids,
        relationships=[],
    )
    
    return RelationshipGraph(clusters=[cluster])


def run_backtest_with_synthetic_relationships(
    market_ids: list[str],
    relationships: list[tuple[str, str, str, float]],  # (type, from, to, confidence)
    data_dir: str = DEFAULT_DATA_DIR,
    **kwargs,
) -> BacktestReport:
    """Run backtest with manually specified relationships.
    
    Useful for testing specific relationship types without LLM.
    
    Args:
        market_ids: Markets to backtest
        relationships: List of (type, from_market, to_market, confidence) tuples
        data_dir: Data directory path
        **kwargs: Additional arguments passed to run_backtest
        
    Returns:
        BacktestReport
    """
    # Build relationship objects
    rel_objects = [
        MarketRelationship(
            type=rel_type,
            from_market=from_market,
            to_market=to_market,
            confidence=confidence,
        )
        for rel_type, from_market, to_market, confidence in relationships
    ]
    
    # Create cluster with all markets
    cluster = MarketCluster(
        cluster_id="synthetic",
        market_ids=market_ids,
        relationships=rel_objects,
    )
    
    graph = RelationshipGraph(clusters=[cluster])
    
    return run_backtest(
        market_ids=market_ids,
        relationship_graph=graph,
        data_dir=data_dir,
        **kwargs,
    )


def print_report(report: BacktestReport) -> None:
    """Print a formatted backtest report.
    
    Args:
        report: The BacktestReport to print
    """
    print(format_report(report))
