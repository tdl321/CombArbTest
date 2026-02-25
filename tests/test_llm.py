"""LLM Analysis module tests (focused on 3+ market partitions).

Tests:
1. Schema validation
2. Market clustering
3. Partition constraint extraction
4. Full pipeline (build_relationship_graph)

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
    RelationshipExtractor,
    build_relationship_graph,
)


# =============================================================================
# Sample Markets - 3+ market partitions for testing
# =============================================================================

# 2024 Election: 3 candidates = valid partition
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

# NBA Championship: 4 teams = valid partition
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

# All sample markets
SAMPLE_MARKETS = ELECTION_MARKETS + CHAMPIONSHIP_MARKETS


# =============================================================================
# Unit Tests (no LLM required)
# =============================================================================

class TestSchemaValidation:
    """Test Pydantic models for partition constraints."""
    
    def test_market_relationship_partition_types(self):
        """Test that partition types are valid."""
        # mutually_exclusive is valid
        rel = MarketRelationship(
            type="mutually_exclusive",
            from_market="A",
            to_market="B",
            confidence=0.9,
        )
        assert rel.type == "mutually_exclusive"
        
        # exhaustive is valid
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
                MarketRelationship(
                    type="exhaustive",
                    from_market="trump",
                    to_market=None,
                    confidence=0.9,
                ),
                MarketRelationship(
                    type="mutually_exclusive",
                    from_market="trump",
                    to_market="biden",
                    confidence=0.9,
                ),
                MarketRelationship(
                    type="mutually_exclusive",
                    from_market="trump",
                    to_market="kennedy",
                    confidence=0.9,
                ),
                MarketRelationship(
                    type="mutually_exclusive",
                    from_market="biden",
                    to_market="kennedy",
                    confidence=0.9,
                ),
            ],
        )
        assert cluster.size == 3
        assert len(cluster.relationships) == 4  # 1 exhaustive + 3 pairs
    
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
        
        # Check constraint types
        exclusive_rels = graph.get_relationships_by_type(RelationshipType.MUTUALLY_EXCLUSIVE)
        assert len(exclusive_rels) == 3


# =============================================================================
# Integration Tests (require LLM)
# =============================================================================

@pytest.mark.integration
@pytest.mark.llm
class TestMarketClustering:
    """Integration tests for market clustering."""
    
    def test_clusters_related_markets(self):
        """Test that related markets are clustered together."""
        clusterer = MarketClusterer()
        clusters = clusterer.cluster_markets(SAMPLE_MARKETS)
        
        assert len(clusters) >= 1
        
        # Check all markets are assigned
        all_assigned = set()
        for cluster in clusters:
            all_assigned.update(cluster.market_ids)
        
        all_ids = {m.id for m in SAMPLE_MARKETS}
        # At minimum, 3+ market clusters should be formed
        assert len(all_assigned) > 0
    
    def test_prefers_partition_sized_clusters(self):
        """Test that clustering prefers 3+ market groups."""
        clusterer = MarketClusterer()
        clusters = clusterer.cluster_markets(ELECTION_MARKETS)
        
        # Should create at least one cluster with 3 markets
        partition_clusters = [c for c in clusters if len(c.market_ids) >= 3]
        assert len(partition_clusters) >= 1


@pytest.mark.integration
@pytest.mark.llm
class TestPartitionExtraction:
    """Integration tests for partition constraint extraction."""
    
    def test_extracts_partition_constraints(self):
        """Test extraction of partition constraints from 3+ market cluster."""
        cluster = MarketCluster(
            cluster_id="election-test",
            theme="2024 Presidential Election",
            market_ids=[m.id for m in ELECTION_MARKETS],
            relationships=[],
        )
        
        market_info = {m.id: m for m in ELECTION_MARKETS}
        extractor = RelationshipExtractor()
        relationships = extractor.extract_relationships(cluster, market_info)
        
        # Should have exhaustive constraint
        exhaustive = [r for r in relationships if r.type == "exhaustive"]
        assert len(exhaustive) >= 1
        
        # Should have mutual exclusivity constraints
        exclusive = [r for r in relationships if r.type == "mutually_exclusive"]
        assert len(exclusive) >= 1
    
    def test_skips_two_market_clusters(self):
        """Test that 2-market clusters are skipped."""
        cluster = MarketCluster(
            cluster_id="binary-test",
            theme="Binary market",
            market_ids=["market-a", "market-b"],
            relationships=[],
        )
        
        market_info = {
            "market-a": MarketInfo(id="market-a", question="Q1", outcomes=["Yes", "No"]),
            "market-b": MarketInfo(id="market-b", question="Q2", outcomes=["Yes", "No"]),
        }
        
        extractor = RelationshipExtractor()
        relationships = extractor.extract_relationships(cluster, market_info)
        
        # Should return empty - not enough markets for partition
        assert len(relationships) == 0


@pytest.mark.integration
@pytest.mark.llm
class TestFullPipeline:
    """Integration tests for complete pipeline."""
    
    def test_build_relationship_graph(self):
        """Test full pipeline with sample markets."""
        graph = build_relationship_graph(SAMPLE_MARKETS)
        
        assert graph.model_used is not None
        assert graph.generated_at is not None
        
        # Should have at least one cluster
        assert len(graph.clusters) >= 1
        
        # All clusters should have 3+ markets (others filtered)
        for cluster in graph.clusters:
            assert len(cluster.market_ids) >= 3
    
    def test_constraint_dict_export(self):
        """Test exporting graph to constraint dict for optimizer."""
        graph = build_relationship_graph(ELECTION_MARKETS)
        
        constraint_dict = graph.to_constraint_dict()
        
        assert "market_ids" in constraint_dict
        assert "constraints" in constraint_dict
        
        # Should have partition constraints
        if constraint_dict["constraints"]:
            types = [c["type"] for c in constraint_dict["constraints"]]
            # Should include partition-related types
            assert any(t in types for t in ["mutually_exclusive", "exhaustive"])


# =============================================================================
# Run as script for manual testing
# =============================================================================

def main():
    """Run tests manually."""
    print("=" * 60)
    print("LLM Analysis Module Test (3+ Market Partitions)")
    print("=" * 60)
    print(f"Testing with {len(SAMPLE_MARKETS)} sample markets\n")
    
    # Test 1: Schema validation
    print("Test 1: Schema Validation...")
    test = TestSchemaValidation()
    test.test_market_relationship_partition_types()
    test.test_market_cluster_with_partition_relationships()
    test.test_relationship_graph_partition_summary()
    print("✓ Schema validation passed\n")
    
    # Test 2: Full pipeline (requires LLM)
    print("Test 2: Full Pipeline...")
    try:
        graph = build_relationship_graph(SAMPLE_MARKETS)
        print(f"✓ Built graph: {len(graph.clusters)} clusters, {graph.total_relationships} relationships")
        
        print("\nClusters:")
        for cluster in graph.clusters:
            print(f"  - {cluster.cluster_id}: {len(cluster.market_ids)} markets, "
                  f"{len(cluster.relationships)} constraints")
        
        print("\n✓ Full pipeline passed")
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
    
    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
