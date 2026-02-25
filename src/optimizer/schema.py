"""Data structures for marginal polytope arbitrage optimization.

This module defines the condition space model where each market has multiple
conditions (outcomes), and the marginal polytope is the convex hull of all
valid joint outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


@dataclass
class Condition:
    """A single condition (outcome) in the condition space.

    Each market has multiple conditions, e.g., "Trump wins PA" has YES and NO.
    """

    condition_id: str  # "market_A::YES"
    market_id: str  # "market_A"
    outcome_index: int  # 0 or 1 (index within market)
    outcome_name: str  # "YES" or "NO"


@dataclass
class ConditionSpace:
    """The full condition space across all markets.

    Maps markets to their conditions and provides utilities for indexing.
    """

    conditions: list[Condition]
    market_to_conditions: dict[str, list[int]] = field(default_factory=dict)
    _market_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_market_data(
        cls,
        market_ids: list[str],
        market_outcomes: dict[str, list[str]] | None = None,
    ) -> ConditionSpace:
        """Build condition space from market definitions.

        Args:
            market_ids: List of market identifiers
            market_outcomes: Optional mapping of market_id -> outcome names.
                           Defaults to ["YES", "NO"] for each market.

        Returns:
            ConditionSpace with all conditions indexed
        """
        if market_outcomes is None:
            market_outcomes = {m: ["YES", "NO"] for m in market_ids}

        conditions = []
        market_to_conditions = {}
        idx = 0

        for market_id in market_ids:
            outcomes = market_outcomes.get(market_id, ["YES", "NO"])
            market_to_conditions[market_id] = []

            for outcome_idx, outcome_name in enumerate(outcomes):
                condition = Condition(
                    condition_id=f"{market_id}::{outcome_name}",
                    market_id=market_id,
                    outcome_index=outcome_idx,
                    outcome_name=outcome_name,
                )
                conditions.append(condition)
                market_to_conditions[market_id].append(idx)
                idx += 1

        space = cls(
            conditions=conditions,
            market_to_conditions=market_to_conditions,
            _market_ids=list(market_ids),
        )
        return space

    def get_yes_index(self, market_id: str) -> int:
        """Get the condition index for YES outcome of a market."""
        indices = self.market_to_conditions[market_id]
        # Assume first outcome is YES
        return indices[0]

    def get_no_index(self, market_id: str) -> int:
        """Get the condition index for NO outcome of a market."""
        indices = self.market_to_conditions[market_id]
        # Assume second outcome is NO
        return indices[1]

    def get_condition_indices(self, market_id: str) -> list[int]:
        """Get all condition indices for a market."""
        return self.market_to_conditions[market_id]

    def n_conditions(self) -> int:
        """Total number of conditions."""
        return len(self.conditions)

    def n_markets(self) -> int:
        """Number of markets."""
        return len(self._market_ids)

    @property
    def market_ids(self) -> list[str]:
        """List of market IDs in order."""
        return self._market_ids


class RelationshipType(str, Enum):
    """Types of logical relationships between markets."""

    IMPLIES = "implies"  # B=YES implies A=YES
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # At most one can be YES
    EQUIVALENT = "equivalent"  # Same outcome
    OPPOSITE = "opposite"  # Opposite outcomes
    # Additional types for backward compatibility
    AND = "and"
    OR = "or"
    PREREQUISITE = "prerequisite"
    EXHAUSTIVE = "exhaustive"
    INCOMPATIBLE = "incompatible"


class MarketRelationship(BaseModel):
    """A logical relationship between two markets."""

    type: str  # Allow string for backward compatibility
    from_market: str
    to_market: str | None = None  # None for unary constraints
    confidence: float = 1.0
    reasoning: str | None = None  # Optional LLM explanation


class MarketCluster(BaseModel):
    """A cluster of related markets with their relationships."""

    cluster_id: str
    market_ids: list[str]
    relationships: list[MarketRelationship] = Field(default_factory=list)
    is_partition: bool = False  # If True, markets must sum to 1
    theme: str | None = None  # Optional cluster description

    @property
    def size(self) -> int:
        return len(self.market_ids)


class RelationshipGraph(BaseModel):
    """Graph of market relationships."""

    clusters: list[MarketCluster] = Field(default_factory=list)
    model_used: str | None = None
    total_markets: int = 0
    total_relationships: int = 0

    def get_relationships(self, market_ids: list[str]) -> list[MarketRelationship]:
        """Get all relationships involving the given markets."""
        market_set = set(market_ids)
        relationships = []
        for cluster in self.clusters:
            for rel in cluster.relationships:
                if rel.from_market in market_set or (
                    rel.to_market and rel.to_market in market_set
                ):
                    relationships.append(rel)
        return relationships

    def get_all_relationships(self) -> list[MarketRelationship]:
        """Get all relationships across all clusters."""
        rels = []
        for cluster in self.clusters:
            rels.extend(cluster.relationships)
        return rels

    def get_all_market_ids(self) -> set[str]:
        """Get all unique market IDs across clusters."""
        ids = set()
        for cluster in self.clusters:
            ids.update(cluster.market_ids)
        return ids

    @property
    def computed_relationships(self) -> int:
        return sum(len(c.relationships) for c in self.clusters)


class OptimizationConfig(BaseModel):
    """Configuration for Frank-Wolfe optimization."""

    max_iterations: int = 1000
    tolerance: float = 1e-6
    epsilon_init: float = 0.1  # Initial barrier contraction
    epsilon_min: float = 0.001  # Minimum barrier contraction
    epsilon_decay: float = 0.9  # Decay rate per iteration

    # Step size configuration
    step_mode: Literal["adaptive", "line_search", "fixed"] = "line_search"
    smoothness_alpha: float = 0.1  # EMA decay for smoothness estimation
    initial_smoothness: float = 1.0  # Initial L estimate for adaptive mode
    min_smoothness: float = 1e-6  # Minimum L to prevent division by zero
    fixed_step_size: float = 0.5  # Step size for fixed mode

    # Backward compatibility aliases
    initial_barrier: float = 0.1
    min_barrier: float = 0.001
    barrier_decay: float = 0.9
    line_search: bool = True
    verbose: bool = False


# =============================================================================
# Backward Compatibility Types (for existing backtest/simulator code)
# =============================================================================


class ConstraintViolation(BaseModel):
    """Details about a violated constraint."""

    constraint_type: str
    from_market: str
    to_market: str | None
    violation_amount: float
    description: str


class ArbitrageResult(BaseModel):
    """Result from the Frank-Wolfe arbitrage solver.

    Backward-compatible format using dict[str, float] for prices.
    """

    market_prices: dict[str, float]  # Original prices (YES probability)
    coherent_prices: dict[str, float]  # Arbitrage-free prices
    kl_divergence: float
    constraints_violated: list[ConstraintViolation] = Field(default_factory=list)
    converged: bool
    iterations: int
    final_gap: float = 0.0

    @property
    def has_arbitrage(self) -> bool:
        return self.kl_divergence > 1e-6 or len(self.constraints_violated) > 0

    def get_price_adjustments(self) -> dict[str, float]:
        return {
            mid: self.coherent_prices[mid] - self.market_prices[mid]
            for mid in self.market_prices
        }


class MarginalArbitrageResult(BaseModel):
    """Result of marginal polytope arbitrage detection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input prices per condition (condition_id -> price)
    condition_prices: dict[str, float]

    # Projected coherent prices per condition
    coherent_condition_prices: dict[str, float]

    # Aggregated by market (market_id -> [p_yes, p_no, ...])
    market_prices: dict[str, list[float]]

    # Coherent prices aggregated by market
    coherent_market_prices: dict[str, list[float]]

    # KL divergence between market and coherent prices
    kl_divergence: float

    # Duality gap at termination
    duality_gap: float = 0.0

    # Whether optimization converged
    converged: bool

    # Number of iterations run
    iterations: int

    # Active vertices found during optimization (list of binary vectors)
    active_vertices: list[list[int]] = Field(default_factory=list)

    # Guaranteed profit (KL - gap)
    @property
    def guaranteed_profit(self) -> float:
        return max(0.0, self.kl_divergence - self.duality_gap)

    def has_arbitrage(self, threshold: float = 0.01) -> bool:
        """Check if there's significant arbitrage opportunity."""
        return self.kl_divergence > threshold
