"""LLM Analysis Output Schema (LLM-04).

Pydantic models for market clustering and relationship extraction results.
These schemas are consumed by the Optimizer agent to build constraints.

Implements formal logical constraints for combinatorial arbitrage:
- Mutual Exclusivity: Only one outcome can be true (Σ P ≤ 1)
- Exhaustiveness: At least one outcome must occur (Σ P ≥ 1)
- Logical Implication: Truth of A requires truth of B (P(A) ≤ P(B))
- Prerequisite: Temporal/causal dependency
- Incompatibility: Outcomes cannot coexist (P(A ∧ B) = 0)

NOTE: This schema is the CANONICAL source. The optimizer schema
(src/optimizer/schema.py) is a compatible subset for minimal usage.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of logical relationships between markets.

    These map to mathematical constraints used in arbitrage detection:

    MUTUALLY_EXCLUSIVE: P(A) + P(B) ≤ 1
        Only one outcome within a set can be true.
        Example: Trump wins vs Harris wins

    EXHAUSTIVE: Σ P(outcomes) ≥ 1
        At least one outcome in a set must occur.
        Example: In a binary market, YES or NO must be true.

    IMPLIES: P(A) ≤ P(B)
        If A is true, B must also be true.
        Example: "Wins state" implies "wins at least one state"

    PREREQUISITE: P(B) ≤ P(A), temporal ordering
        A must happen before/for B to be possible.
        Example: "Nominated" prerequisite for "wins general"

    INCOMPATIBLE: P(A ∧ B) = 0, so P(A) + P(B) ≤ 1
        Outcomes cannot coexist even if not direct competitors.
        Example: Two different Finals matchups

    AND: Strong positive correlation
        Outcomes tend to resolve the same way.
        Example: "Wins House" AND "Wins Senate" (sweep)

    OR: P(A ∨ B ∨ C) requirement
        At least one must occur for broader condition.
        Example: Victory path through swing states
    """
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    EXHAUSTIVE = "exhaustive"
    IMPLIES = "implies"
    PREREQUISITE = "prerequisite"
    INCOMPATIBLE = "incompatible"
    AND = "and"
    OR = "or"


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
    - MUTUALLY_EXCLUSIVE: p(from) + p(to) <= 1 constraint
    - EXHAUSTIVE: sum(p) >= 1 for the market set
    - IMPLIES: p(from) <= p(to) constraint
    - PREREQUISITE: p(to) <= p(from), temporal ordering
    - INCOMPATIBLE: p(from) + p(to) <= 1 (structural impossibility)
    - AND: joint probability / correlation constraint
    - OR: p(from) + p(to) - p(from AND to) >= min constraint
    """
    type: Literal[
        "mutually_exclusive",
        "exhaustive",
        "implies",
        "prerequisite",
        "incompatible",
        "and",
        "or"
    ]
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

    @property
    def constraint_formula(self) -> str:
        """Human-readable constraint formula."""
        formulas = {
            "mutually_exclusive": f"P({self.from_market}) + P({self.to_market}) ≤ 1",
            "exhaustive": f"Σ P(outcomes) ≥ 1",
            "implies": f"P({self.from_market}) ≤ P({self.to_market})",
            "prerequisite": f"P({self.to_market}) ≤ P({self.from_market})",
            "incompatible": f"P({self.from_market} ∧ {self.to_market}) = 0",
            "and": f"P({self.from_market}) ≈ P({self.to_market})",
            "or": f"P({self.from_market} ∨ {self.to_market}) ≥ threshold",
        }
        return formulas.get(self.type, "Unknown constraint")


class MarketCluster(BaseModel):
    """A cluster of semantically related markets.

    Markets in the same cluster share a common theme and may have
    logical relationships that create arbitrage constraints.
    """
    cluster_id: str
    theme: str  # e.g., "2024 US Presidential Election"
    market_ids: list[str]  # List of market IDs in this cluster
    relationships: list[MarketRelationship] = Field(default_factory=list)
    is_partition: bool = False  # If True, all markets must sum to 1

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

    def get_mutual_exclusion_sets(self) -> list[set[str]]:
        """Get sets of mutually exclusive market IDs."""
        # Build graph of mutual exclusions
        exclusions = {}
        for rel in self.relationships:
            if rel.type in ("mutually_exclusive", "incompatible"):
                if rel.from_market not in exclusions:
                    exclusions[rel.from_market] = set()
                if rel.to_market not in exclusions:
                    exclusions[rel.to_market] = set()
                exclusions[rel.from_market].add(rel.to_market)
                exclusions[rel.to_market].add(rel.from_market)

        # Find connected components (mutual exclusion sets)
        visited = set()
        sets = []
        for market_id in exclusions:
            if market_id not in visited:
                component = set()
                stack = [market_id]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        stack.extend(exclusions.get(current, set()) - visited)
                if len(component) > 1:
                    sets.append(component)

        return sets


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

    def get_all_market_ids(self) -> set[str]:
        """Get all unique market IDs across clusters.
        
        Added for compatibility with optimizer schema interface.
        """
        ids = set()
        for cluster in self.clusters:
            ids.update(cluster.market_ids)
        return ids

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

    def get_constraint_summary(self) -> dict:
        """Get summary of all constraint types."""
        summary = {
            "mutually_exclusive": 0,
            "exhaustive": 0,
            "implies": 0,
            "prerequisite": 0,
            "incompatible": 0,
            "and": 0,
            "or": 0,
        }
        for rel in self.get_all_relationships():
            if rel.type in summary:
                summary[rel.type] += 1
        return summary

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
                "formula": rel.constraint_formula,
            }
            constraints.append(constraint)

        return {
            "market_ids": market_ids,
            "constraints": constraints,
            "constraint_summary": self.get_constraint_summary(),
            "generated_at": self.generated_at.isoformat(),
        }
