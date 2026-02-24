"""Backtest Module for Combinatorial Arbitrage.

This module implements the backtesting pipeline:
- BT-01: Walk-Forward Simulation
- BT-02: Arbitrage Detection
- BT-03: PnL Calculation
- BT-04: Summary Report

Usage:
    from src.backtest import run_backtest, print_report
    
    # Run a backtest with default settings
    report = run_backtest(
        market_ids=["market1", "market2", "market3"],
        start_block=18000000,
        end_block=18100000,
        kl_threshold=0.01,
        transaction_cost=0.015,
    )
    
    # Print formatted report
    print_report(report)
    
    # Or with synthetic relationships for testing
    from src.backtest import run_backtest_with_synthetic_relationships
    
    report = run_backtest_with_synthetic_relationships(
        market_ids=["market1", "market2"],
        relationships=[
            ("implies", "market1", "market2", 0.9),
        ],
        max_ticks=1000,
    )
"""

from .schema import (
    ArbitrageOpportunity,
    BacktestConfig,
    BacktestReport,
    ClusterPerformance,
    SimulationState,
)

from .pnl import (
    calculate_theoretical_profit,
    calculate_kl_profit,
    apply_transaction_costs,
    calculate_opportunity_pnl,
    PnLTracker,
)

from .report import (
    generate_report,
    calculate_max_drawdown,
    calculate_cluster_performance,
    format_report,
    report_to_dict,
)

from .simulator import (
    WalkForwardSimulator,
    run_simulation,
)

from .runner import (
    run_backtest,
    run_backtest_with_synthetic_relationships,
    print_report,
)

__all__ = [
    # Schema
    "ArbitrageOpportunity",
    "BacktestConfig",
    "BacktestReport",
    "ClusterPerformance",
    "SimulationState",
    # PnL
    "calculate_theoretical_profit",
    "calculate_kl_profit",
    "apply_transaction_costs",
    "calculate_opportunity_pnl",
    "PnLTracker",
    # Report
    "generate_report",
    "calculate_max_drawdown",
    "calculate_cluster_performance",
    "format_report",
    "report_to_dict",
    # Simulator
    "WalkForwardSimulator",
    "run_simulation",
    # Runner (main entry points)
    "run_backtest",
    "run_backtest_with_synthetic_relationships",
    "print_report",
]
