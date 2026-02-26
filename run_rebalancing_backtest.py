#!/usr/bin/env python3
"""Demonstration runner: rebalancing arbitrage via the modular pipeline.

Shows how the new modular architecture supports both instant-arb
(partition_arb) and held-position (rebalancing_arb) strategies
in a single pipeline run.

Usage:
    python run_rebalancing_backtest.py
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure strategies are registered
import src.strategies.partition_arb
import src.strategies.combinatorial_arb
import src.strategies.rebalancing_arb

from src.core.types import StrategyConfig
from src.data.adapter import ParquetMarketSource
from src.grouping.manual_grouper import ManualGrouper
from src.grouping.correlation_grouper import CorrelationGrouper
from src.pipeline import Pipeline
from src.strategies.registry import list_strategies
from src.config import get_data_dir, get_db_path


def main():
    logger.info("=" * 60)
    logger.info("MODULAR PIPELINE DEMO: Rebalancing + Partition Arbitrage")
    logger.info("=" * 60)

    # Show available strategies
    logger.info("Registered strategies: %s", list_strategies())

    # Initialize data source
    data_dir = get_data_dir()
    db_path = get_db_path()
    logger.info("Data dir: %s", data_dir)
    logger.info("DB path: %s", db_path)

    data_source = ParquetMarketSource(data_dir=data_dir, db_path=db_path)

    # Create pipeline
    pipeline = Pipeline(data_source=data_source)

    # Add groupers
    pipeline.add_grouper(ManualGrouper([
        {
            "group_id": "nba_mvp_2025",
            "name": "NBA MVP 2024-25",
            "market_ids": [],  # Will be populated from data
            "is_partition": True,
        },
    ]))
    pipeline.add_grouper(CorrelationGrouper(
        min_correlation=0.5,
        lookback_days=30,
    ))

    # Add strategies
    pipeline.add_strategy("partition_arb", StrategyConfig(
        strategy_name="partition_arb",
        min_profit_threshold=0.01,
        fee_per_leg=0.01,
    ))

    pipeline.add_strategy("rebalancing_arb", StrategyConfig(
        strategy_name="rebalancing_arb",
        min_profit_threshold=0.02,
        fee_per_leg=0.01,
        extra={
            "lookback_hours": 168,
            "deviation_threshold": 0.05,
            "max_holding_hours": 48,
        },
    ))

    logger.info("Pipeline configured with strategies: %s", pipeline.registered_strategies)
    logger.info("Pipeline configured with groupers: %s", pipeline.registered_groupers)

    # Run the pipeline
    logger.info("Running pipeline...")
    try:
        results = pipeline.run(limit=100)  # Limit markets for demo
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        import traceback
        traceback.print_exc()
        results = {}

    # Report results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for strategy_name, result in results.items():
        logger.info("")
        logger.info("Strategy: %s", strategy_name)
        logger.info("  Opportunities: %d", result.total_opportunities)
        logger.info("  Trades: %d", result.total_trades)
        logger.info("  Gross P&L: $%.4f", result.gross_pnl)
        logger.info("  Fees: $%.4f", result.total_fees)
        logger.info("  Net P&L: $%.4f", result.net_pnl)
        logger.info("  Max Drawdown: $%.4f", result.max_drawdown)
        logger.info("  Win Rate: %.1f%%", result.win_rate * 100)

        if result.equity_curve:
            logger.info("  Equity curve: %d points", len(result.equity_curve))
            logger.info("  Period: %s to %s", result.start_date, result.end_date)

    if not results:
        logger.info("No results (no groups matched the available markets)")
        logger.info("This is expected if market IDs are not pre-configured.")
        logger.info("The pipeline architecture is verified by the unit tests.")

    # Cleanup
    data_source.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
