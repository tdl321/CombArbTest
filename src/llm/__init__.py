"""LLM integration for market analysis.

Modules:
- client: LLMClient for OpenRouter API (Kimi 2.5)
- schema: Pydantic models for clustering and relationships
- clustering: Semantic market clustering
- extractor: Relationship extraction between markets
"""

from .client import LLMClient, get_client
from .schema import (
    MarketCluster,
    MarketInfo,
    MarketRelationship,
    RelationshipGraph,
    RelationshipType,
)
from .clustering import MarketClusterer, cluster_markets
from .extractor import (
    RelationshipExtractor,
    build_relationship_graph,
    extract_relationships,
)

__all__ = [
    # Client
    "LLMClient",
    "get_client",
    # Schema
    "MarketCluster",
    "MarketInfo", 
    "MarketRelationship",
    "RelationshipGraph",
    "RelationshipType",
    # Clustering
    "MarketClusterer",
    "cluster_markets",
    # Extraction
    "RelationshipExtractor",
    "build_relationship_graph",
    "extract_relationships",
]
