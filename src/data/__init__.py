"""Data layer for Polymarket parquet files.

Implements:
- DATA-01: Load market metadata
- DATA-02: Load trade history  
- DATA-03: Build price time series
- DATA-04: Point-in-time data access
- DATA-05: Market query filters
- DATA-06: Tick-level data access (arbitrage detection)
"""

from .loader import BlockLoader, DataLoader, MarketLoader, TradeLoader
from .models import BlockTimestamp, Market, MarketStatus, Trade
from .price_series import (
    PointInTimeDataAccess,
    PriceSeriesBuilder,
    Resolution,
)
from .tick_stream import (
    CrossMarketIterator,
    CrossMarketSnapshot,
    MarketStateSnapshot,
    Tick,
    TickPosition,
    TickStream,
    detect_price_divergence,
)

__all__ = [
    # Loaders
    "DataLoader",
    "MarketLoader",
    "TradeLoader", 
    "BlockLoader",
    # Price series (OHLCV)
    "PriceSeriesBuilder",
    "PointInTimeDataAccess",
    "Resolution",
    # Tick-level (arbitrage)
    "TickPosition",
    "Tick",
    "MarketStateSnapshot",
    "CrossMarketSnapshot",
    "TickStream",
    "CrossMarketIterator",
    "detect_price_divergence",
    # Models
    "Market",
    "Trade",
    "BlockTimestamp",
    "MarketStatus",
]
