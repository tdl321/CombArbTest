"""Backtest Module for Combinatorial Arbitrage.

This module implements the backtesting pipeline:
- BT-01: Walk-Forward Simulation
- BT-02: Arbitrage Detection
- BT-03: PnL Calculation
- BT-04: Summary Report

Focused on partition constraint checking (3+ markets):
- Partition = exhaustive + mutually exclusive markets
- Sum P = 1 exactly for partitions
- Violations in either direction indicate arbitrage

NOTE: Simple 2-market constraints have been removed.
Focus is exclusively on 3+ market combinatorial arbitrage.

Usage (Legacy):
    from src.backtest import run_backtest, print_report
    
    # Run a backtest with default settings
    report = run_backtest(
        market_ids=["market1", "market2", "market3"],
        start_block=18000000,
        end_block=18100000,
    )
    print_report(report)

Usage (New - Recommended):
    from src.backtest import run_backtest_with_report, BacktestOutput
    
    # Run backtest with full report generation
    output: BacktestOutput = run_backtest_with_report(
        market_ids=["market1", "market2", "market3"],
        relationship_graph=graph,
        market_loader=market_loader,
        trade_loader=trade_loader,
        block_loader=block_loader,
        config=config,
        output_dir="./reports",
    )
    
    # Access results
    print(f"Total opportunities: {output.report.total_opportunities}")
    print(f"Visualizations: {output.visualization_paths}")
"""

# Import ArbitrageTrade from arbitrage module (re-export for convenience)
from src.arbitrage.extractor import ArbitrageTrade

from .schema import (
    # Core schemas (new arbitrage model)
    ArbitrageOpportunity,
    BacktestConfig,
    BacktestOutput,  # NEW: Complete backtest output
    # Report schemas
    BacktestReport,
    ClusterPerformance,
    SimulationState,
)

# -----------------------------------------------------------------------------
# PnL Functions
# DEPRECATION NOTE: calculate_theoretical_profit, calculate_kl_profit, and
# calculate_opportunity_pnl are deprecated. Use ArbitrageExtractor instead.
# See src/backtest/pnl.py module docstring for migration guide.
# -----------------------------------------------------------------------------
from .pnl import (
    # Deprecated (use ArbitrageExtractor.extract_trades() instead)
    calculate_theoretical_profit,  # DEPRECATED
    calculate_kl_profit,           # DEPRECATED
    apply_transaction_costs,
    calculate_opportunity_pnl,     # DEPRECATED
    PnLTracker,                    # DEPRECATED (uses old schema)
    # New correct functions
    calculate_arbitrage_pnl,
    calculate_partition_pnl,
    calculate_implies_pnl,
)

from .report import (
    generate_report,
    calculate_max_drawdown,
    calculate_cluster_performance,
    format_report,
    report_to_dict,
)

# -----------------------------------------------------------------------------
# NEW: Report Generator with correct arbitrage model
# -----------------------------------------------------------------------------
from .report_generator import (
    ArbitrageBacktestReport,
    ConstraintTypeStats,
    EdgeDistribution,
    generate_full_report,
    generate_report_text,
    generate_simplex_visualizations,
)

from .constraint_checker import (
    # Partition constraint checking (3+ markets)
    PartitionViolation,
    check_partition,
    is_partition_constraint,
    get_partition_market_ids,
    compute_partition_coherent_prices,
    compute_partition_trades,
    format_partition_violation,
)

from .simulator import (
    WalkForwardSimulator,
    run_simulation,
    run_signal_simulation,
    run_backtest_with_report,  # NEW: Full backtest with report
)

from .runner import (
    run_backtest,
    run_backtest_with_synthetic_relationships,
    print_report,
)

from .signal_report import generate_signal_report

__all__ = [
    # Schema - Core (new arbitrage model)
    "ArbitrageTrade",  # Re-exported from src.arbitrage.extractor
    "ArbitrageOpportunity",
    "BacktestConfig",
    "BacktestOutput",  # NEW
    "BacktestReport",
    "ClusterPerformance",
    "SimulationState",
    
    # PnL - Deprecated (kept for backward compatibility)
    "calculate_theoretical_profit",  # DEPRECATED
    "calculate_kl_profit",           # DEPRECATED
    "apply_transaction_costs",
    "calculate_opportunity_pnl",     # DEPRECATED
    "PnLTracker",                    # DEPRECATED
    # PnL - New correct functions
    "calculate_arbitrage_pnl",
    "calculate_partition_pnl",
    "calculate_implies_pnl",
    
    # Report (old)
    "generate_report",
    "calculate_max_drawdown",
    "calculate_cluster_performance",
    "format_report",
    "report_to_dict",
    
    # Report Generator (new - correct arbitrage model)
    "ArbitrageBacktestReport",
    "ConstraintTypeStats",
    "EdgeDistribution",
    "generate_full_report",
    "generate_report_text",
    "generate_simplex_visualizations",
    
    # Signal Report
    "generate_signal_report",
    
    # Partition Constraint Checker (3+ markets)
    "PartitionViolation",
    "check_partition",
    "is_partition_constraint",
    "get_partition_market_ids",
    "compute_partition_coherent_prices",
    "compute_partition_trades",
    "format_partition_violation",
    
    # Simulator
    "WalkForwardSimulator",
    "run_simulation",
    "run_signal_simulation",
    "run_backtest_with_report",  # NEW: Full backtest with report generation
    
    # Runner (main entry points)
    "run_backtest",
    "run_backtest_with_synthetic_relationships",
    "print_report",
    # Schema (NEW)
    "ArbitrageTrade",
    "BacktestOutput",
    # Report Generator (NEW)
    "ArbitrageBacktestReport",
    "generate_full_report",
    "generate_report_text",
    # Simulator (NEW)
    "run_backtest_with_report",
]
