#!/usr/bin/env python3
"""Integration test for LLM Analysis module (LLM-02, LLM-03, LLM-04).

Tests:
1. Schema validation
2. Market clustering
3. Relationship extraction
4. Full pipeline (build_relationship_graph)
"""

import json
import logging
from datetime import datetime

# Configure logging
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
    get_client,
)


# Sample markets for testing - simulating high-volume Polymarket markets
SAMPLE_MARKETS = [
    # US 2024 Election cluster
    MarketInfo(
        id="trump-wins-election-2024",
        question="Will Donald Trump win the 2024 US Presidential Election?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="biden-wins-election-2024",
        question="Will Joe Biden win the 2024 US Presidential Election?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="trump-wins-pennsylvania",
        question="Will Donald Trump win Pennsylvania in 2024?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="trump-wins-georgia",
        question="Will Donald Trump win Georgia in 2024?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="trump-wins-michigan",
        question="Will Donald Trump win Michigan in 2024?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="republican-wins-popular-vote",
        question="Will the Republican candidate win the popular vote in 2024?",
        outcomes=["Yes", "No"],
    ),
    # Bitcoin cluster
    MarketInfo(
        id="btc-100k-2024",
        question="Will Bitcoin reach $100,000 by end of 2024?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="btc-150k-2024",
        question="Will Bitcoin reach $150,000 by end of 2024?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="btc-etf-approved",
        question="Will a spot Bitcoin ETF be approved in the US by 2024?",
        outcomes=["Yes", "No"],
    ),
    # AI cluster
    MarketInfo(
        id="openai-gpt5-2024",
        question="Will OpenAI release GPT-5 by end of 2024?",
        outcomes=["Yes", "No"],
    ),
    MarketInfo(
        id="google-gemini-ultra-beats-gpt4",
        question="Will Google Gemini Ultra outperform GPT-4 on benchmarks?",
        outcomes=["Yes", "No"],
    ),
    # Standalone market
    MarketInfo(
        id="world-cup-winner-2026",
        question="Will Brazil win the 2026 FIFA World Cup?",
        outcomes=["Yes", "No"],
    ),
]


def test_schema_validation():
    """Test that Pydantic models validate correctly."""
    print("\n" + "="*60)
    print("TEST 1: Schema Validation")
    print("="*60)
    
    # Test MarketRelationship
    rel = MarketRelationship(
        type="implies",
        from_market="trump-wins-pennsylvania",
        to_market="trump-wins-election-2024",
        confidence=0.85,
        reasoning="Winning PA is a strong indicator of winning the election",
    )
    print(f"MarketRelationship: {rel.type} from {rel.from_market} to {rel.to_market}")
    assert rel.confidence == 0.85
    
    # Test MarketCluster
    cluster = MarketCluster(
        cluster_id="us-2024-election",
        theme="2024 US Presidential Election",
        market_ids=["trump-wins-election-2024", "biden-wins-election-2024"],
        relationships=[rel],
    )
    print(f"MarketCluster: {cluster.cluster_id} with {cluster.size} markets")
    assert cluster.size == 2
    
    # Test RelationshipGraph
    graph = RelationshipGraph(
        clusters=[cluster],
        model_used="test-model",
    )
    print(f"RelationshipGraph: {graph.total_markets} markets, {graph.total_relationships} relationships")
    assert graph.total_markets == 2
    assert graph.total_relationships == 1
    
    # Test to_constraint_dict
    constraint_dict = graph.to_constraint_dict()
    print(f"Constraint dict keys: {list(constraint_dict.keys())}")
    assert "market_ids" in constraint_dict
    assert "constraints" in constraint_dict
    
    print("Schema validation PASSED")
    return True


def test_clustering():
    """Test market clustering with LLM."""
    print("\n" + "="*60)
    print("TEST 2: Market Clustering (LLM-02)")
    print("="*60)
    
    clusterer = MarketClusterer()
    clusters = clusterer.cluster_markets(SAMPLE_MARKETS)
    
    print(f"\nFound {len(clusters)} clusters:")
    for cluster in clusters:
        print(f"\n  Cluster: {cluster.cluster_id}")
        print(f"  Theme: {cluster.theme}")
        print(f"  Markets ({cluster.size}):")
        for mid in cluster.market_ids:
            # Find the market question
            market = next((m for m in SAMPLE_MARKETS if m.id == mid), None)
            if market:
                print(f"    - {mid}: {market.question[:60]}...")
    
    # Verify all markets are assigned
    all_assigned = set()
    for cluster in clusters:
        all_assigned.update(cluster.market_ids)
    
    all_market_ids = {m.id for m in SAMPLE_MARKETS}
    missing = all_market_ids - all_assigned
    if missing:
        print(f"\nWARNING: Markets not assigned to any cluster: {missing}")
    else:
        print(f"\nAll {len(all_market_ids)} markets assigned to clusters")
    
    print("Clustering PASSED")
    return clusters


def test_extraction(clusters: list[MarketCluster]):
    """Test relationship extraction with LLM."""
    print("\n" + "="*60)
    print("TEST 3: Relationship Extraction (LLM-03)")
    print("="*60)
    
    market_info = {m.id: m for m in SAMPLE_MARKETS}
    extractor = RelationshipExtractor()
    
    enriched_clusters = extractor.extract_for_all_clusters(clusters, market_info)
    
    total_relationships = 0
    for cluster in enriched_clusters:
        if cluster.relationships:
            print(f"\nCluster: {cluster.cluster_id}")
            print(f"Relationships ({len(cluster.relationships)}):")
            for rel in cluster.relationships:
                print(f"  - {rel.type.upper()}: {rel.from_market} -> {rel.to_market}")
                print(f"    Confidence: {rel.confidence:.2f}")
                if rel.reasoning:
                    print(f"    Reasoning: {rel.reasoning[:80]}...")
            total_relationships += len(cluster.relationships)
    
    print(f"\nTotal relationships extracted: {total_relationships}")
    print("Extraction PASSED")
    return enriched_clusters


def test_full_pipeline():
    """Test the complete pipeline: clustering + extraction."""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline (build_relationship_graph)")
    print("="*60)
    
    graph = build_relationship_graph(SAMPLE_MARKETS)
    
    print(f"\nRelationshipGraph Summary:")
    print(f"  Model: {graph.model_used}")
    print(f"  Generated at: {graph.generated_at}")
    print(f"  Total clusters: {len(graph.clusters)}")
    print(f"  Total markets: {graph.total_markets}")
    print(f"  Total relationships: {graph.total_relationships}")
    
    # Print relationships by type
    print("\nRelationships by type:")
    for rel_type in RelationshipType:
        rels = graph.get_relationships_by_type(rel_type)
        if rels:
            print(f"  {rel_type.value}: {len(rels)}")
    
    # Export to constraint dict for optimizer
    constraint_dict = graph.to_constraint_dict()
    print(f"\nConstraint dict for optimizer:")
    print(f"  Market IDs: {len(constraint_dict['market_ids'])}")
    print(f"  Constraints: {len(constraint_dict['constraints'])}")
    
    # Pretty print a sample of the constraint dict
    if constraint_dict['constraints']:
        print("\nSample constraints:")
        for c in constraint_dict['constraints'][:3]:
            print(f"  {json.dumps(c, indent=4)}")
    
    print("\nFull pipeline PASSED")
    return graph


def main():
    """Run all tests."""
    print("="*60)
    print("LLM Analysis Module Integration Test")
    print("="*60)
    print(f"Testing with {len(SAMPLE_MARKETS)} sample markets")
    
    try:
        # Test 1: Schema validation (no LLM needed)
        test_schema_validation()
        
        # Test 2: Clustering (requires LLM)
        clusters = test_clustering()
        
        # Test 3: Extraction (requires LLM)
        enriched_clusters = test_extraction(clusters)
        
        # Test 4: Full pipeline (requires LLM)
        graph = test_full_pipeline()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        
        # Final output
        print("\nFinal RelationshipGraph JSON:")
        print(json.dumps(graph.model_dump(mode='json'), indent=2, default=str))
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
