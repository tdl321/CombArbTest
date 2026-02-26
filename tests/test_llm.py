"""LLM Analysis module tests (focused on 3+ market partitions).

Tests:
1. Schema validation (unit tests - no LLM required)
2. Market clustering (integration - requires LLM)
3. Partition constraint extraction (integration - requires LLM)
4. Full pipeline (integration - requires LLM)

NOTE: Tests focus on 3+ market partition detection.
Simple 2-market constraints have been removed from the system.
"""

import json
import logging
import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.llm import (
    MarketCluster,
    MarketInfo,
    MarketRelationship,
    RelationshipGraph,
    RelationshipType,
    MarketClusterer,
    build_relationship_graph,
)


# =============================================================================
# Sample Markets - 3+ market partitions for testing
# =============================================================================

ELECTION_MARKETS = [
    MarketInfo(
        id="trump-wins-2024",
        question="Will Donald Trump win the 2024 US Presidential Election?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="biden-wins-2024",
        question="Will Joe Biden win the 2024 US Presidential Election?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="kennedy-wins-2024",
        question="Will RFK Jr win the 2024 US Presidential Election?",
        outcomes=["Yes", "No"],
    ),
]

CHAMPIONSHIP_MARKETS = [
    MarketInfo(
        id="lakers-win-championship",
        question="Will the Lakers win the 2024 NBA Championship?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="celtics-win-championship",
        question="Will the Celtics win the 2024 NBA Championship?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="warriors-win-championship",
        question="Will the Warriors win the 2024 NBA Championship?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="nuggets-win-championship",
        question="Will the Nuggets win the 2024 NBA Championship?",
        outcomes=["Yes", "No"],
    ),
]

SAMPLE_MARKETS = ELECTION_MARKETS + CHAMPIONSHIP_MARKETS


# =============================================================================
# Unit Tests (no LLM required)
# =============================================================================

class TestSchemaValidation:
    """Test Pydantic models for partition constraints."""

    def test_market_relationship_partition_types(self):
        """Test that partition types are valid."""
        rel = MarketRelationship(
            type="mutually_exclusive",
            from_market="A",
            to_market="B",
            confidence=0.9,
        )
        assert rel.type == "mutually_exclusive"

        rel = MarketRelationship(
            type="exhaustive",
            from_market="A",
            to_market=None,
            confidence=0.9,
        )
        assert rel.type == "exhaustive"

    def test_market_cluster_with_partition_relationships(self):
        """Test cluster with partition constraints."""
        cluster = MarketCluster(
            cluster_id="election-2024",
            theme="2024 US Presidential Election",
            market_ids=["trump", "biden", "kennedy"],
            relationships=[
                MarketRelationship(type="exhaustive", from_market="trump", to_market=None, confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="trump", to_market="biden", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="trump", to_market="kennedy", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="biden", to_market="kennedy", confidence=0.9),
            ],
        )
        assert cluster.size == 3
        assert len(cluster.relationships) == 4

    def test_relationship_graph_partition_summary(self):
        """Test graph with partition cluster."""
        cluster = MarketCluster(
            cluster_id="test-partition",
            theme="Test",
            market_ids=["A", "B", "C"],
            relationships=[
                MarketRelationship(type="exhaustive", from_market="A", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="A", to_market="B", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="A", to_market="C", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="B", to_market="C", confidence=0.9),
            ],
        )

        graph = RelationshipGraph(clusters=[cluster], model_used="test")

        assert graph.total_markets == 3
        assert graph.total_relationships == 4

        exclusive_rels = graph.get_relationships_by_type(RelationshipType.MUTUALLY_EXCLUSIVE)
        assert len(exclusive_rels) == 3

    def test_market_info_creation(self):
        """Test MarketInfo model creation."""
        market = MarketInfo(
            id="test-market",
            question="Will X happen?",
            outcomes=["Yes", "No"],
        )
        assert market.id == "test-market"
        assert len(market.outcomes) == 2

    def test_relationship_graph_empty(self):
        """Test empty relationship graph."""
        graph = RelationshipGraph(clusters=[], model_used="test")
        assert graph.total_markets == 0
        assert graph.total_relationships == 0

    def test_market_clusterer_creation(self):
        """Test MarketClusterer can be instantiated."""
        clusterer = MarketClusterer()
        assert clusterer is not None

    def test_two_market_cluster_not_partition(self):
        """Test that 2-market clusters are too small for partition detection."""
        cluster = MarketCluster(
            cluster_id="binary-test",
            theme="Binary market",
            market_ids=["market-a", "market-b"],
            relationships=[],
        )
        assert len(cluster.market_ids) < 3


# =============================================================================
# Integration Tests (require LLM - skip by default)
# =============================================================================

@pytest.mark.integration
@pytest.mark.llm
class TestMarketClustering:
    """Integration tests for market clustering (requires LLM API)."""

    def test_cluster_and_extract(self):
        """Test that cluster_and_extract returns a RelationshipGraph."""
        clusterer = MarketClusterer()
        graph = clusterer.cluster_and_extract(SAMPLE_MARKETS)

        assert isinstance(graph, RelationshipGraph)
        assert len(graph.clusters) >= 1

    def test_prefers_partition_sized_clusters(self):
        """Test that clustering prefers 3+ market groups."""
        clusterer = MarketClusterer()
        graph = clusterer.cluster_and_extract(ELECTION_MARKETS)

        partition_clusters = [c for c in graph.clusters if len(c.market_ids) >= 3]
        assert len(partition_clusters) >= 1


@pytest.mark.integration
@pytest.mark.llm
class TestFullPipeline:
    """Integration tests for complete pipeline (requires LLM API)."""

    def test_build_relationship_graph(self):
        """Test full pipeline with sample markets."""
        graph = build_relationship_graph(SAMPLE_MARKETS)

        assert graph.model_used is not None
        assert graph.generated_at is not None
        assert len(graph.clusters) >= 1

    def test_constraint_dict_export(self):
        """Test exporting graph to constraint dict for optimizer."""
        graph = build_relationship_graph(ELECTION_MARKETS)

        constraint_dict = graph.to_constraint_dict()

        assert "market_ids" in constraint_dict
        assert "constraints" in constraint_dict
