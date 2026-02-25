"""Signal schema - pure arbitrage detection without PnL."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class ArbitrageSignal:
    """Pure signal proving arbitrage exists."""
    timestamp: datetime
    cluster_id: str
    markets: list[str]
    constraint_type: str  # "partition", "exhaustive", "implies", "mutually_exclusive", "prerequisite"
    
    # The edge
    market_prices: dict[str, float]
    coherent_prices: dict[str, float]
    
    # Signal strength
    edge_magnitude: float      # L2 distance
    kl_divergence: float       # KL divergence
    direction: dict[str, float]  # Per-market: positive=buy, negative=sell
    
    # Proof
    constraint_violation: str  # Human-readable description
    
    # Metadata
    block_number: Optional[int] = None
    detection_method: str = "solver"  # "simple" or "solver"
