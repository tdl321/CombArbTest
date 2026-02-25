"""Schema definitions for the Optimization Engine.

OPT-05 Output Contract: Defines the ArbitrageResult model that captures
the output of the Frank-Wolfe solver.

NOTE: This schema is designed to be COMPATIBLE with src/llm/schema.py.
The LLM schema has additional fields (theme, reasoning, metadata) that
are optional here for minimal optimizer usage.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# Relationship types - MUST match src/llm/schema.py RelationshipType enum
RELATIONSHIP_TYPES = Literal[
    "implies",
    "mutually_exclusive", 
    "and",
    "or",
    "prerequisite",
    "exhaustive",
    "incompatible"  # Added for compatibility with LLM schema
]


class MarketRelationship(BaseModel):
    """A relationship between markets as identified by the LLM.
    
    This is the INPUT from the LLM agent that identifies logical
    relationships between markets.
    
    Compatible with src/llm/schema.MarketRelationship (LLM schema adds 'reasoning').
    """
    type: RELATIONSHIP_TYPES
    from_market: str
    to_market: str | None = None  # None for unary constraints (e.g., exhaustive)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None  # Optional: LLM explanation (present in llm/schema.py)


class MarketCluster(BaseModel):
    """A cluster of related markets.
    
    Contains market IDs and the relationships between them.
    
    Compatible with src/llm/schema.MarketCluster (LLM schema adds 'theme').
    """
    cluster_id: str
    market_ids: list[str]
    relationships: list[MarketRelationship] = Field(default_factory=list)
    is_partition: bool = False  # If True, all markets must sum to 1
    theme: str | None = None  # Optional: cluster description (present in llm/schema.py)


class RelationshipGraph(BaseModel):
    """Full graph of market relationships.
    
    This is the complete INPUT from the LLM agent.
    
    Compatible with src/llm/schema.RelationshipGraph (LLM schema adds metadata).
    """
    clusters: list[MarketCluster]
    # Optional metadata fields (present in llm/schema.py)
    model_used: str | None = None
    total_markets: int = 0
    total_relationships: int = 0
    
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
