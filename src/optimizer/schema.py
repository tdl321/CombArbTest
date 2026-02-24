"""Schema definitions for the Optimization Engine.

OPT-05 Output Contract: Defines the ArbitrageResult model that captures
the output of the Frank-Wolfe solver.
"""

from typing import Literal
from pydantic import BaseModel, Field


class MarketRelationship(BaseModel):
    """A relationship between markets as identified by the LLM.
    
    This is the INPUT from the LLM agent that identifies logical
    relationships between markets.
    """
    type: Literal["implies", "mutually_exclusive", "and", "or", "prerequisite"]
    from_market: str
    to_market: str | None = None  # None for unary constraints
    confidence: float = Field(ge=0.0, le=1.0)


class MarketCluster(BaseModel):
    """A cluster of related markets.
    
    Contains market IDs and the relationships between them.
    """
    cluster_id: str
    market_ids: list[str]
    relationships: list[MarketRelationship]


class RelationshipGraph(BaseModel):
    """Full graph of market relationships.
    
    This is the complete INPUT from the LLM agent.
    """
    clusters: list[MarketCluster]
    
    def get_all_market_ids(self) -> set[str]:
        """Get all unique market IDs across clusters."""
        ids = set()
        for cluster in self.clusters:
            ids.update(cluster.market_ids)
        return ids
    
    def get_all_relationships(self) -> list[MarketRelationship]:
        """Get all relationships across clusters."""
        rels = []
        for cluster in self.clusters:
            rels.extend(cluster.relationships)
        return rels


class ConstraintViolation(BaseModel):
    """Details about a violated constraint."""
    constraint_type: str
    from_market: str
    to_market: str | None
    violation_amount: float  # How much the constraint is violated by
    description: str


class ArbitrageResult(BaseModel):
    """Result from the Frank-Wolfe arbitrage solver.
    
    This is the OUTPUT of the optimization engine.
    """
    market_prices: dict[str, float]  # Original market prices (p)
    coherent_prices: dict[str, float]  # Arbitrage-free prices (q)
    kl_divergence: float  # KL(p || q) - distance from coherent
    constraints_violated: list[ConstraintViolation]  # Violated constraints
    converged: bool  # Did the solver converge?
    iterations: int  # Number of FW iterations
    final_gap: float = 0.0  # Final duality gap (FW gap)
    
    @property
    def has_arbitrage(self) -> bool:
        """True if the original prices allow arbitrage."""
        return self.kl_divergence > 1e-6 or len(self.constraints_violated) > 0
    
    def get_price_adjustments(self) -> dict[str, float]:
        """Get the adjustment needed for each market."""
        return {
            market_id: self.coherent_prices[market_id] - self.market_prices[market_id]
            for market_id in self.market_prices
        }


class OptimizationConfig(BaseModel):
    """Configuration for the Frank-Wolfe solver."""
    max_iterations: int = 1000
    tolerance: float = 1e-6  # Convergence tolerance for duality gap
    initial_barrier: float = 0.1  # Initial barrier parameter epsilon
    min_barrier: float = 0.001  # Minimum barrier as we converge
    barrier_decay: float = 0.9  # Rate at which barrier shrinks
    line_search: bool = True  # Use line search for step size
    verbose: bool = False
