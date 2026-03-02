"""Backtest Module for Combinatorial Arbitrage.

This module implements the backtesting pipeline:
- BT-01: Walk-Forward Simulation
- BT-02: Arbitrage Detection
- BT-03: PnL Calculation
- BT-04: Summary Report

Focused on combinatorial arbitrage using the Frank-Wolfe solver
to find constraint violations across market clusters.
"""

# Import ArbitrageTrade from arbitrage module (re-export for convenience)
from src.arbitrage.extractor import ArbitrageTrade

from .schema import (
    # Core schemas (new arbitrage model)
    ArbitrageOpportunity,
    BacktestConfig,
    BacktestOutput,
    BacktestReport,
    ClusterPerformance,
    SimulationState,
)

# -----------------------------------------------------------------------------
# PnL Functions
# -----------------------------------------------------------------------------
from .pnl import (
    apply_transaction_costs,
    calculate_arbitrage_pnl,
    calculate_implies_pnl,
)

from .report import (
    generate_report,
    calculate_max_drawdown,
    calculate_cluster_performance,
    format_report,
    report_to_dict,
)

__all__ = [
    # Schema - Core (new arbitrage model)
    "ArbitrageTrade",
    "ArbitrageOpportunity",
    "BacktestConfig",
    "BacktestOutput",
    "BacktestReport",
    "ClusterPerformance",
    "SimulationState",
    
    # PnL
    "apply_transaction_costs",
    "calculate_arbitrage_pnl",
    "calculate_implies_pnl",
    
    # Report
    "generate_report",
    "calculate_max_drawdown",
    "calculate_cluster_performance",
    "format_report",
    "report_to_dict",
]
