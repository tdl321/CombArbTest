"""Partition constraint violation checks for combinatorial arbitrage.

This module implements partition checking for 3+ market clusters
where the sum of probabilities must equal 1 (exhaustive AND mutually exclusive).

Partition checking:
- A partition is a set of markets that are BOTH exhaustive AND mutually exclusive
- Constraint: Σ P(markets) = 1 exactly
- Violations in BOTH directions indicate arbitrage:
  - sum < 1: markets underpriced → buy all outcomes → guaranteed profit
  - sum > 1: markets overpriced → sell all outcomes → guaranteed profit

NOTE: Simple 2-market constraints (implies, prerequisite, pairwise exclusive) 
have been removed. This module focuses exclusively on 3+ market partitions
for true combinatorial arbitrage.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# PARTITION VIOLATIONS (N-market constraints where Σ P = 1, N >= 3)
# =============================================================================

@dataclass
class PartitionViolation:
    """Violation where partition sum != 1.
    
    A partition is a set of mutually exclusive AND exhaustive markets.
    The probabilities MUST sum to exactly 1.
    
    Violations indicate arbitrage:
    - sum < 1 (underpriced): buy all outcomes for <$1, one MUST pay $1 → profit
    - sum > 1 (overpriced): sell all outcomes for >$1, one pays $1 → profit
    """
    market_ids: list[str]
    prices: dict[str, float]  # market_id -> observed price
    total: float  # Sum of all prices
    violation_amount: float  # total - 1.0 (negative = underpriced, positive = overpriced)
    direction: str  # "underpriced" or "overpriced"
    coherent_prices: dict[str, float] = field(default_factory=dict)  # Normalized prices
    
    @property
    def profit_potential(self) -> float:
        """Theoretical profit per $1 wagered on all outcomes."""
        return abs(self.violation_amount) / self.total if self.total > 0 else 0.0
    
    def __post_init__(self):
        """Calculate coherent (normalized) prices if not provided."""
        if not self.coherent_prices and self.total > 0:
            # Normalize prices to sum to 1
            self.coherent_prices = {
                mid: p / self.total
                for mid, p in self.prices.items()
            }


def check_partition(
    market_ids: list[str],
    prices: dict[str, float],
    tolerance: float = 0.01,
) -> Optional[PartitionViolation]:
    """Check if partition sum equals 1. Violations in either direction.
    
    Args:
        market_ids: List of market IDs in the partition (must have 3+)
        prices: Dict mapping market_id to current price
        tolerance: How much deviation from 1.0 to allow (default 1%)
        
    Returns:
        PartitionViolation if |sum - 1| > tolerance, None otherwise
    """
    # Require 3+ markets for combinatorial arbitrage
    if len(market_ids) < 3:
        return None
    
    # Get prices for all markets in the partition
    partition_prices = {
        mid: prices[mid]
        for mid in market_ids
        if mid in prices
    }
    
    # Need prices for all markets in the partition
    if len(partition_prices) != len(market_ids):
        missing = set(market_ids) - set(partition_prices.keys())
        logger.debug("[PARTITION] Missing prices for %d markets: %s", 
                     len(missing), list(missing)[:3])
        return None
    
    total = sum(partition_prices.values())
    violation = total - 1.0
    
    if abs(violation) > tolerance:
        direction = "overpriced" if violation > 0 else "underpriced"
        logger.debug("[PARTITION] Violation detected: sum=%.4f (%s by %.4f)",
                     total, direction, abs(violation))
        
        return PartitionViolation(
            market_ids=list(partition_prices.keys()),
            prices=partition_prices,
            total=total,
            violation_amount=violation,
            direction=direction,
        )
    
    return None


def is_partition_constraint(cluster) -> bool:
    """Check if cluster represents a partition (exhaustive + all pairwise exclusive).
    
    A partition requires:
    1. At least 3 markets (for combinatorial arbitrage)
    2. At least one exhaustive relationship
    3. Most pairs being mutually exclusive
    
    Args:
        cluster: MarketCluster from src.llm.schema or src.optimizer.schema
        
    Returns:
        True if the cluster represents a valid partition constraint
    """
    # Require 3+ markets for combinatorial arbitrage
    if len(cluster.market_ids) < 3:
        return False
    
    if not hasattr(cluster, 'relationships') or not cluster.relationships:
        return False
    
    # Check for exhaustive constraint
    has_exhaustive = any(r.type == "exhaustive" for r in cluster.relationships)
    
    if not has_exhaustive:
        return False
    
    # Count exclusive pairs
    exclusive_pairs = sum(
        1 for r in cluster.relationships 
        if r.type in ("mutually_exclusive", "incompatible") and r.to_market is not None
    )
    
    # Expected pairs for full mutual exclusivity: n(n-1)/2
    n = len(cluster.market_ids)
    expected_pairs = n * (n - 1) // 2
    
    # It's a partition if exhaustive AND at least 50% of pairs are exclusive
    # (We use 50% threshold because LLM might not enumerate all pairs explicitly)
    if expected_pairs > 0:
        coverage = exclusive_pairs / expected_pairs
        is_partition = coverage >= 0.5
        
        logger.debug("[PARTITION] Cluster %s: exhaustive=%s, exclusive_pairs=%d/%d (%.1f%%), is_partition=%s",
                     cluster.cluster_id, has_exhaustive, exclusive_pairs, expected_pairs, 
                     coverage * 100, is_partition)
        
        return is_partition
    
    return False


def get_partition_market_ids(cluster) -> list[str]:
    """Get the market IDs that form the partition from a cluster.
    
    For exhaustive constraints, this identifies which markets are included
    in the partition. The exhaustive constraint typically references the 
    first market, and we use the exclusive pairs to find the full set.
    
    Args:
        cluster: MarketCluster
        
    Returns:
        List of market IDs in the partition
    """
    # Start with exhaustive markets
    exhaustive_markets = set()
    for r in cluster.relationships:
        if r.type == "exhaustive":
            exhaustive_markets.add(r.from_market)
            if r.to_market:
                exhaustive_markets.add(r.to_market)
    
    # Add all markets connected by exclusive relationships
    exclusive_graph = set()
    for r in cluster.relationships:
        if r.type in ("mutually_exclusive", "incompatible") and r.to_market:
            exclusive_graph.add(r.from_market)
            exclusive_graph.add(r.to_market)
    
    # The partition is the intersection of exhaustive-connected and exclusive-connected
    # If exhaustive markets exist, use them as the base
    if exhaustive_markets:
        # Take all cluster markets that are either exhaustive or exclusive with exhaustive
        partition = exhaustive_markets.union(exclusive_graph)
        # Filter to only include markets in the cluster
        partition = [mid for mid in cluster.market_ids if mid in partition]
    else:
        # Fallback: use all markets in exclusive graph
        partition = [mid for mid in cluster.market_ids if mid in exclusive_graph]
    
    return partition if partition else cluster.market_ids


def compute_partition_coherent_prices(
    partition_prices: dict[str, float],
) -> dict[str, float]:
    """Compute coherent (arbitrage-free) prices for a partition.
    
    For a partition, the coherent prices are simply the normalized versions
    that sum to 1. Each market's coherent price = observed_price / total.
    
    Args:
        partition_prices: Dict of market_id -> observed price
        
    Returns:
        Dict of market_id -> coherent price (summing to 1)
    """
    total = sum(partition_prices.values())
    if total <= 0:
        return partition_prices.copy()
    
    return {mid: p / total for mid, p in partition_prices.items()}


def compute_partition_trades(
    prices: dict[str, float],
    coherent_prices: dict[str, float],
    direction: str,
) -> dict[str, str]:
    """Compute trade directions for partition arbitrage.
    
    For partitions:
    - If underpriced (sum < 1): BUY all outcomes
    - If overpriced (sum > 1): SELL all outcomes
    
    Individual market adjustments tell us which markets are MORE mispriced:
    - Larger adjustment = more mispriced = larger position
    
    Args:
        prices: Observed prices
        coherent_prices: Arbitrage-free prices
        direction: "underpriced" or "overpriced"
        
    Returns:
        Dict of market_id -> "BUY" or "SELL"
    """
    if direction == "underpriced":
        # All outcomes underpriced -> buy all
        return {mid: "BUY" for mid in prices}
    else:
        # All outcomes overpriced -> sell all
        return {mid: "SELL" for mid in prices}


def format_partition_violation(violation: PartitionViolation) -> str:
    """Format partition violation for logging/display.
    
    Args:
        violation: PartitionViolation object
        
    Returns:
        Human-readable description
    """
    market_strs = [f"{mid[:12]}={p:.3f}" for mid, p in list(violation.prices.items())[:4]]
    if len(violation.prices) > 4:
        market_strs.append(f"... +{len(violation.prices) - 4} more")
    
    return (
        f"Partition({len(violation.market_ids)} markets): "
        f"sum={violation.total:.4f}, {violation.direction} by {abs(violation.violation_amount):.4f} "
        f"[{', '.join(market_strs)}]"
    )
