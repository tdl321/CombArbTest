"""Data layer for Polymarket parquet files.

Implements:
- DATA-01: Load market metadata
- DATA-02: Load trade history  
- DATA-05: Market query filters
- DATA-06: Tick-level data access (arbitrage detection)
- DATA-07: Market category index
"""

from .loader import BlockLoader, DataLoader, MarketLoader, TradeLoader
from .models import BlockTimestamp, Market, MarketStatus, Trade
from .tick_stream import (
    CrossMarketIterator,
    CrossMarketSnapshot,
    MarketStateSnapshot,
    Tick,
    TickPosition,
    TickStream,
    detect_price_divergence,
)
from .category_index import CategoryIndex, MarketCategory

__all__ = [
    # Loaders
    "DataLoader",
    "MarketLoader",
    "TradeLoader", 
    "BlockLoader",
    # Tick-level (arbitrage)
    "TickPosition",
    "Tick",
    "MarketStateSnapshot",
    "CrossMarketSnapshot",
    "TickStream",
    "CrossMarketIterator",
    "detect_price_divergence",
    # Category index
    "CategoryIndex",
    "MarketCategory",
    # Models
    "Market",
    "Trade",
    "BlockTimestamp",
    "MarketStatus",
]
