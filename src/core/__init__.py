"""Core type system and protocols for the modular pipeline.

This package defines the canonical types shared across all layers.
No module outside core/ should define its own Market, Opportunity, etc.
"""

from .types import (
    MarketStatus,
    MarketMeta,
    PricePoint,
    MarketSnapshot,
    MarketTimeSeries,
    GroupSnapshot,
    GroupingType,
    ConstraintType,
    Constraint,
    MarketGroup,
    TradeDirection,
    TradeLeg,
    TradeType,
    Opportunity,
    ExecutedTrade,
    DataRequirements,
    StrategyConfig,
)

from .protocols import (
    MarketDataSource,
    MarketGrouper,
    ArbitrageStrategy,
    BacktestEvaluator,
    ReportPlugin,
)

__all__ = [
    "MarketStatus",
    "MarketMeta",
    "PricePoint",
    "MarketSnapshot",
    "MarketTimeSeries",
    "GroupSnapshot",
    "GroupingType",
    "ConstraintType",
    "Constraint",
    "MarketGroup",
    "TradeDirection",
    "TradeLeg",
    "TradeType",
    "Opportunity",
    "ExecutedTrade",
    "DataRequirements",
    "StrategyConfig",
    "MarketDataSource",
    "MarketGrouper",
    "ArbitrageStrategy",
    "BacktestEvaluator",
    "ReportPlugin",
]
