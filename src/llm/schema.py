"""LLM Analysis Output Schema (LLM-04).

Pydantic models for market clustering and relationship extraction results.
These schemas are consumed by the Optimizer agent to build constraints.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of logical relationships between markets."""
    IMPLIES = "implies"           # A implies B (if A happens, B must happen)
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # A and B cannot both happen
    AND = "and"                   # Both A and B must happen together
    OR = "or"                     # At least one of A or B must happen
    PREREQUISITE = "prerequisite" # A must happen before B can happen


class MarketInfo(BaseModel):
    """Minimal market info for clustering input."""
    id: str
    question: str
    outcomes: list[str]
    
    class Config:
        frozen = True


class MarketRelationship(BaseModel):
    """A logical relationship between two markets.
    
    Used by the optimizer to build constraints:
    - IMPLIES: p(from) <= p(to) constraint
    - MUTUALLY_EXCLUSIVE: p(from) + p(to) <= 1 constraint  
    - AND: joint probability constraint
    - OR: p(from) + p(to) - p(from AND to) >= min constraint
    - PREREQUISITE: temporal/causal ordering constraint
    """
    type: Literal["implies", "mutually_exclusive", "and", "or", "prerequisite"]
    from_market: str  # market_id
    to_market: str | None = None  # market_id (None for unary relations)
    confidence: float = Field(ge=0.0, le=1.0)  # 0-1 confidence score
    reasoning: str | None = None  # LLM explanation for the relationship
    
    def __hash__(self) -> int:
        return hash((self.type, self.from_market, self.to_market))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarketRelationship):
            return False
        return (
            self.type == other.type
            and self.from_market == other.from_market
            and self.to_market == other.to_market
        )


class MarketCluster(BaseModel):
    """A cluster of semantically related markets.
    
    Markets in the same cluster share a common theme and may have
    logical relationships that create arbitrage constraints.
    """
    cluster_id: str
    theme: str  # e.g., "2024 US Presidential Election"
    market_ids: list[str]  # List of market IDs in this cluster
    relationships: list[MarketRelationship] = Field(default_factory=list)
    
    @property
    def size(self) -> int:
        """Number of markets in this cluster."""
        return len(self.market_ids)
    
    def get_relationships_for_market(self, market_id: str) -> list[MarketRelationship]:
        """Get all relationships involving a specific market."""
        return [
            r for r in self.relationships
            if r.from_market == market_id or r.to_market == market_id
        ]


class RelationshipGraph(BaseModel):
    """Complete graph of market clusters and relationships.
    
    This is the main output consumed by the Optimizer agent.
    """
    clusters: list[MarketCluster]
    generated_at: datetime = Field(default_factory=datetime.now)
    model_used: str | None = None  # LLM model that generated this
    total_markets: int = 0
    total_relationships: int = 0
    
    def model_post_init(self, __context) -> None:
        """Compute summary stats after initialization."""
        object.__setattr__(
            self, 
            "total_markets", 
            sum(c.size for c in self.clusters)
        )
        object.__setattr__(
            self,
            "total_relationships",
            sum(len(c.relationships) for c in self.clusters)
        )
    
    def get_cluster_for_market(self, market_id: str) -> MarketCluster | None:
        """Find the cluster containing a specific market."""
        for cluster in self.clusters:
            if market_id in cluster.market_ids:
                return cluster
        return None
    
    def get_all_relationships(self) -> list[MarketRelationship]:
        """Get all relationships across all clusters."""
        relationships = []
        for cluster in self.clusters:
            relationships.extend(cluster.relationships)
        return relationships
    
    def get_relationships_by_type(
        self, 
        rel_type: RelationshipType
    ) -> list[MarketRelationship]:
        """Get all relationships of a specific type."""
        return [
            r for r in self.get_all_relationships()
            if r.type == rel_type.value
        ]
    
    def to_constraint_dict(self) -> dict:
        """Convert to format expected by the optimizer.
        
        Returns a dict with:
        - market_ids: list of all market IDs
        - constraints: list of constraint dicts with type, markets, confidence
        """
        market_ids = []
        for cluster in self.clusters:
            market_ids.extend(cluster.market_ids)
        
        constraints = []
        for rel in self.get_all_relationships():
            constraint = {
                "type": rel.type,
                "from_market": rel.from_market,
                "to_market": rel.to_market,
                "confidence": rel.confidence,
            }
            constraints.append(constraint)
        
        return {
            "market_ids": market_ids,
            "constraints": constraints,
            "generated_at": self.generated_at.isoformat(),
        }
