"""Backtest Schema Definitions (BT-01 to BT-04).

Pydantic models for arbitrage opportunities and backtest reports.
"""

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, computed_field, field_validator

from src.arbitrage.extractor import ArbitrageTrade
from src.optimizer.schema import ArbitrageResult

logger = logging.getLogger(__name__)


class ArbitrageOpportunity(BaseModel):
    """A detected arbitrage opportunity at a specific point in time.
    
    Embeds ArbitrageTrade for correct profit calculation based on
    violation magnitude, not price-convergence model.
    """
    model_config = {"arbitrary_types_allowed": True}
    
    # Detection metadata
    timestamp: datetime
    block_number: int
    cluster_id: str
    detection_method: str = "solver"  # "partition" or "solver"
    
    # The actual arbitrage trade (contains positions, locked_profit, etc.)
    trade: ArbitrageTrade
    
    # Solver output (for diagnostics) - Optional since partition doesn't use solver
    solver_result: Optional[ArbitrageResult] = None
    
    @field_validator('trade', mode='before')
    @classmethod
    def validate_trade(cls, v: Any) -> ArbitrageTrade:
        """Accept both ArbitrageTrade instances and dicts."""
        if isinstance(v, ArbitrageTrade):
            return v
        if isinstance(v, dict):
            logger.debug("[BACKTEST] Converting dict to ArbitrageTrade")
            return ArbitrageTrade(**v)
        raise ValueError("Expected ArbitrageTrade or dict, got %s" % type(v))
    
    # Legacy fields for backward compatibility (deprecated)
    # These will be computed from trade
    @computed_field
    @property
    def locked_profit(self) -> float:
        """Guaranteed profit at settlement (before fees)."""
        return self.trade.locked_profit
    
    @computed_field
    @property
    def theoretical_profit(self) -> float:
        """Alias for locked_profit (deprecated, use locked_profit)."""
        return self.trade.locked_profit
    
    def net_profit(self, fee_per_leg: float = 0.01) -> float:
        """Profit after transaction fees."""
        return self.trade.net_profit(fee_per_leg=fee_per_leg)
    
    @computed_field
    @property
    def positions(self) -> dict[str, str]:
        """Market positions (BUY/SELL)."""
        return self.trade.positions
    
    @computed_field
    @property
    def trade_direction(self) -> dict[str, str]:
        """Alias for positions (deprecated, use positions)."""
        return self.trade.positions
    
    @computed_field
    @property
    def market_prices(self) -> dict[str, float]:
        """Market prices at detection time."""
        return self.trade.market_prices
    
    @computed_field
    @property
    def constraint_type(self) -> str:
        """Type of constraint violated."""
        return self.trade.constraint_type
    
    @computed_field
    @property
    def is_profitable_net(self) -> bool:
        """Whether this opportunity is profitable after default costs."""
        return self.trade.net_profit(fee_per_leg=0.01) > 0


class ClusterPerformance(BaseModel):
    """Performance metrics for a single cluster."""
    cluster_id: str
    theme: str
    market_ids: list[str]
    num_opportunities: int
    gross_pnl: float
    net_pnl: float
    win_count: int  # Trades profitable after costs
    loss_count: int
    avg_kl_divergence: float
    max_kl_divergence: float
    
    @computed_field
    @property
    def win_rate(self) -> float:
        """Win rate for this cluster."""
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0


class BacktestConfig(BaseModel):
    """Configuration for running a backtest."""
    market_ids: list[str]
    start_block: Optional[int] = None
    end_block: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    kl_threshold: float = 0.01  # Minimum KL divergence to flag opportunity
    transaction_cost: float = 0.015  # 1.5% round-trip cost
    min_profit: float = 0.001  # Minimum profit to record opportunity
    progress_interval: int = 1000  # Print progress every N ticks
    max_ticks: Optional[int] = None  # Stop after N ticks (for testing)
    store_all_opportunities: bool = True  # Store full opportunity list in report
    signal_only: bool = False  # If True, skip PnL calculation
    
    def __init__(self, **data):
        super().__init__(**data)
        logger.debug(
            "[BACKTEST] Config created: %d markets, kl=%.4f, tx_cost=%.4f",
            len(self.market_ids),
            self.kl_threshold,
            self.transaction_cost
        )


class BacktestReport(BaseModel):
    """Complete backtest results and statistics.
    
    Contains summary metrics and individual opportunities for detailed analysis.
    """
    model_config = {"arbitrary_types_allowed": True}
    
    # Time range
    start_date: datetime
    end_date: datetime
    duration_hours: float = 0.0
    
    # Market coverage
    markets_analyzed: int
    clusters_found: int
    
    # Opportunity stats
    total_opportunities: int
    opportunities_per_hour: float = 0.0
    
    # PnL metrics
    gross_pnl: float
    net_pnl: float
    transaction_costs: float
    
    # Win/loss stats
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    
    # KL divergence stats
    avg_kl_divergence: float = 0.0
    max_kl_divergence: float = 0.0
    min_kl_divergence: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0  # Placeholder for more sophisticated analysis
    
    # Best/worst trades
    best_trade: Optional[ArbitrageOpportunity] = None
    worst_trade: Optional[ArbitrageOpportunity] = None
    
    # Cluster-level breakdown
    cluster_performance: list[ClusterPerformance] = Field(default_factory=list)
    
    # All opportunities (optional, can be large)
    opportunities: list[ArbitrageOpportunity] = Field(default_factory=list)
    
    # Config used
    kl_threshold: float = 0.01
    transaction_cost_rate: float = 0.015
    
    # Additional fields for report compatibility
    avg_violation: float = 0.0
    by_constraint_type: dict = Field(default_factory=dict)
    min_profit_threshold: float = 0.001


class BacktestOutput(BaseModel):
    """Complete output from a backtest run."""
    model_config = {"arbitrary_types_allowed": True}
    
    # Metadata
    run_id: str
    run_timestamp: datetime
    config: BacktestConfig
    
    # Results
    report: BacktestReport
    opportunities: list[ArbitrageOpportunity]
    
    # Output paths
    report_text_path: Optional[str] = None
    visualization_paths: list[str] = Field(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        logger.info(
            "[BACKTEST] Output created: run_id=%s, %d opportunities, %d visualizations",
            self.run_id,
            len(self.opportunities),
            len(self.visualization_paths)
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return self.model_dump()


class SimulationState(BaseModel):
    """Internal state tracking during simulation."""
    current_block: int = 0
    current_log_index: int = 0
    current_timestamp: Optional[datetime] = None
    ticks_processed: int = 0
    opportunities_found: int = 0
    cumulative_pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    def update_pnl(self, pnl: float) -> None:
        """Update cumulative PnL and track drawdown."""
        self.cumulative_pnl += pnl
        if self.cumulative_pnl > self.peak_pnl:
            self.peak_pnl = self.cumulative_pnl
        
        drawdown = self.peak_pnl - self.cumulative_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            logger.debug(
                "[SIM] New max drawdown: %.4f (peak=%.4f, current=%.4f)",
                self.max_drawdown,
                self.peak_pnl,
                self.cumulative_pnl
            )
