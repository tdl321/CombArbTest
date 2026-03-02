"""Polymarket live data pipeline.

Provides API clients, type mappings, constraint inference,
and a dataset builder for feeding live Polymarket data to the solver.
"""

from .clob_client import ClobClient
from .config import PolymarketConfig
from .dataset import DatasetBuilder, DatasetSpec, LiveDataset
from .gamma_client import GammaClient
from .mapping import (
    to_group_snapshot,
    to_market_meta,
    to_market_snapshot,
    to_market_time_series,
    to_price_point,
)
from .relationship_inference import RelationshipInferrer
from .types import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketOrderBook,
    PolymarketPriceHistory,
)

__all__ = [
    # Config
    "PolymarketConfig",
    # API Clients
    "GammaClient",
    "ClobClient",
    # Dataset
    "DatasetSpec",
    "DatasetBuilder",
    "LiveDataset",
    # Types
    "PolymarketEvent",
    "PolymarketMarket",
    "PolymarketOrderBook",
    "PolymarketPriceHistory",
    # Mapping
    "to_market_meta",
    "to_price_point",
    "to_market_snapshot",
    "to_market_time_series",
    "to_group_snapshot",
    # Inference
    "RelationshipInferrer",
]
