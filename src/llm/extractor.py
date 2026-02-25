"""Dependency Extraction (LLM-03).

Extracts logical relationships between markets within clusters.
Focuses on PARTITION constraints (3+ markets) for combinatorial arbitrage detection.

NOTE: Simple 2-market constraints (implies, prerequisite, pairwise exclusive) are
no longer extracted. Focus is exclusively on 3+ market partitions where Σ P = 1.
"""

import json
import logging
import time
from typing import Any

from .client import LLMClient, get_client
from .schema import (
    MarketCluster,
    MarketInfo,
    MarketRelationship,
    RelationshipGraph,
    RelationshipType,
)

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are an expert in logical reasoning and prediction markets. Your task is to identify PARTITION constraints between prediction markets for combinatorial arbitrage.

## PARTITION CONSTRAINTS (3+ Markets)

A PARTITION is a set of markets that are BOTH exhaustive AND mutually exclusive:
- EXHAUSTIVE: At least one outcome MUST occur
- MUTUALLY EXCLUSIVE: Only one outcome CAN occur
- Mathematical constraint: Σ P(outcomes) = 1 exactly

### Requirements for Valid Partitions:
1. Must have 3 or more markets (NOT binary Yes/No markets)
2. Must be BOTH exhaustive AND mutually exclusive
3. Outcomes must cover ALL possibilities for the event

### Examples of Valid Partitions:
- Election winners (3+ candidates): "Trump wins", "Biden wins", "Kennedy wins"
- Championship outcomes (3+ teams): "Lakers win", "Celtics win", "Warriors win"
- Score ranges: "0-5 points", "6-10 points", "11+ points"
- Date ranges: "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"

### What is NOT a Valid Partition:
- Binary Yes/No markets (only 2 outcomes)
- Simple pairwise constraints between 2 markets
- Correlated but not mutually exclusive markets

## OUTPUT REQUIREMENTS

For each partition found:
1. Mark ALL pairwise relationships as mutually_exclusive
2. Add ONE exhaustive relationship for the set

Only return partitions with 3+ markets. Binary markets should be IGNORED.

CRITICAL: Only identify relationships you are confident about. These constraints will be used to detect arbitrage - false positives waste computation.

IMPORTANT: Respond with valid JSON only. No markdown, no extra text."""


EXTRACTION_USER_PROMPT_TEMPLATE = """Analyze these related prediction markets and identify PARTITION constraints.

A PARTITION is a set of 3+ markets that are BOTH:
- EXHAUSTIVE: Exactly one outcome must occur
- MUTUALLY EXCLUSIVE: No two outcomes can both be true

Cluster Theme: {theme}

Markets:
{markets_json}

IMPORTANT FILTERS:
- Only extract partitions with 3 or more markets
- Binary (Yes/No) markets should NOT form partitions
- Focus on true combinatorial arbitrage opportunities

Return a JSON object with this structure:
{{
    "relationships": [
        {{
            "type": "mutually_exclusive",
            "from_market": "market_id",
            "to_market": "market_id",
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }}
    ],
    "constraint_summary": {{
        "exhaustive_sets": [
            {{
                "market_ids": ["id1", "id2", "id3"],
                "is_partition": true,
                "reasoning": "Why these form a complete partition"
            }}
        ]
    }},
    "analysis_notes": "Any observations about the partition structure"
}}

REMEMBER:
- exhaustive_sets with fewer than 3 markets should NOT be returned
- Each partition needs ALL pairwise mutual exclusivity relationships

Return ONLY valid JSON, no markdown formatting."""


class RelationshipExtractor:
    """Extracts logical relationships between markets using LLM analysis."""

    def __init__(
        self,
        client: LLMClient | None = None,
        min_confidence: float = 0.6,
        temperature: float = 0.2,
        max_retries: int = 2,
    ):
        self._client = client
        self.min_confidence = min_confidence
        self.temperature = temperature
        self.max_retries = max_retries
        logger.debug("[EXTRACT] RelationshipExtractor initialized: min_conf=%.2f, temp=%.1f",
                     min_confidence, temperature)

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client()
        return self._client

    def extract_relationships(
        self,
        cluster: MarketCluster,
        market_info: dict[str, MarketInfo],
    ) -> list[MarketRelationship]:
        """Extract relationships between markets in a cluster.
        
        Only processes clusters with 3+ markets for partition detection.
        """
        # Require 3+ markets for partition detection
        if len(cluster.market_ids) < 3:
            logger.debug("[EXTRACT] Cluster %s has < 3 markets, skipping (partition requires 3+)", 
                        cluster.cluster_id)
            return []

        markets_data = []
        for market_id in cluster.market_ids:
            info = market_info.get(market_id)
            if info:
                markets_data.append({
                    "id": info.id,
                    "question": info.question,
                    "outcomes": info.outcomes,
                })

        if len(markets_data) < 3:
            logger.debug("[EXTRACT] Cluster %s has < 3 valid markets, skipping", cluster.cluster_id)
            return []

        markets_json = json.dumps(markets_data, indent=2)
        prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            theme=cluster.theme,
            markets_json=markets_json,
        )

        logger.info("[EXTRACT] Extracting partition constraints for cluster %s with %d markets",
                    cluster.cluster_id, len(markets_data))
        start_time = time.time()

        current_temperature = self.temperature
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat_json(
                    prompt=prompt,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    temperature=current_temperature,
                )
                relationships = self._parse_extraction_response(response, cluster.market_ids)
                elapsed = time.time() - start_time
                logger.info("[EXTRACT] Extracted %d relationships in %.3fs", len(relationships), elapsed)
                return relationships
            except json.JSONDecodeError as e:
                logger.warning("[EXTRACT] JSON parse error on attempt %d for cluster %s: %s",
                              attempt + 1, cluster.cluster_id, e)
                if attempt < self.max_retries:
                    current_temperature = min(0.5, current_temperature + 0.1)
                    continue
                else:
                    logger.error("[EXTRACT] All attempts failed for cluster %s", cluster.cluster_id)
                    return []
            except Exception as e:
                logger.error("[EXTRACT] Extraction failed for cluster %s: %s", cluster.cluster_id, e)
                return []

        return []

    def _parse_extraction_response(
        self,
        response: dict[str, Any],
        valid_market_ids: list[str],
    ) -> list[MarketRelationship]:
        """Parse LLM response into MarketRelationship objects.
        
        Only accepts partition-related constraints (mutually_exclusive, exhaustive).
        """
        relationships = []
        valid_ids_set = set(valid_market_ids)
        
        # Only accept partition-related types
        partition_types = {"mutually_exclusive", "exhaustive"}

        for rel_data in response.get("relationships", []):
            rel_type = rel_data.get("type", "").lower()
            from_market = rel_data.get("from_market")
            to_market = rel_data.get("to_market")
            confidence = rel_data.get("confidence", 0.0)
            reasoning = rel_data.get("reasoning")

            # Only accept partition-related constraint types
            if rel_type not in partition_types:
                logger.debug("[EXTRACT] Skipping non-partition type: %s", rel_type)
                continue

            if from_market not in valid_ids_set:
                logger.debug("[EXTRACT] Invalid from_market: %s", from_market)
                continue

            if to_market and to_market not in valid_ids_set:
                logger.debug("[EXTRACT] Invalid to_market: %s", to_market)
                continue

            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.5

            if confidence < self.min_confidence:
                logger.debug("[EXTRACT] Low confidence (%.2f) constraint skipped", confidence)
                continue

            relationships.append(MarketRelationship(
                type=rel_type,
                from_market=from_market,
                to_market=to_market,
                confidence=confidence,
                reasoning=reasoning,
            ))

        # Process exhaustive_sets from constraint_summary
        constraint_summary = response.get("constraint_summary", {})
        for ex_set in constraint_summary.get("exhaustive_sets", []):
            market_ids = ex_set.get("market_ids", [])
            is_partition = ex_set.get("is_partition", False)
            reasoning = ex_set.get("reasoning", "")
            
            # Filter to valid market IDs and require 3+ markets
            valid_set_ids = [mid for mid in market_ids if mid in valid_ids_set]
            
            if len(valid_set_ids) < 3:
                logger.debug("[EXTRACT] Skipping exhaustive set with < 3 markets")
                continue
            
            # Add exhaustive relationship
            relationships.append(MarketRelationship(
                type="exhaustive",
                from_market=valid_set_ids[0],
                to_market=None,
                confidence=0.8,
                reasoning=f"Partition: {reasoning}" if is_partition else reasoning,
            ))
            
            # Add pairwise mutual exclusivity if marked as partition
            if is_partition:
                for i, id1 in enumerate(valid_set_ids):
                    for id2 in valid_set_ids[i+1:]:
                        relationships.append(MarketRelationship(
                            type="mutually_exclusive",
                            from_market=id1,
                            to_market=id2,
                            confidence=0.8,
                            reasoning=f"Part of partition: {reasoning}",
                        ))

        return relationships

    def extract_for_all_clusters(
        self,
        clusters: list[MarketCluster],
        market_info: dict[str, MarketInfo],
    ) -> list[MarketCluster]:
        """Extract relationships for all clusters with 3+ markets."""
        # Filter to only process clusters with 3+ markets
        valid_clusters = [c for c in clusters if len(c.market_ids) >= 3]
        logger.info("[EXTRACT] Processing %d clusters with 3+ markets (skipping %d smaller clusters)",
                    len(valid_clusters), len(clusters) - len(valid_clusters))
        
        enriched_clusters = []

        for i, cluster in enumerate(valid_clusters):
            logger.debug("[EXTRACT] Processing cluster %d/%d: %s", 
                        i + 1, len(valid_clusters), cluster.cluster_id)
            relationships = self.extract_relationships(cluster, market_info)

            enriched_cluster = MarketCluster(
                cluster_id=cluster.cluster_id,
                theme=cluster.theme,
                market_ids=cluster.market_ids,
                relationships=relationships,
            )
            enriched_clusters.append(enriched_cluster)

        logger.info("[EXTRACT] Completed: %d clusters enriched with partition constraints", 
                    len(enriched_clusters))
        return enriched_clusters


def extract_relationships(
    cluster: MarketCluster,
    market_info: dict[str, MarketInfo],
    client: LLMClient | None = None,
) -> list[MarketRelationship]:
    """Convenience function to extract relationships for a cluster."""
    logger.debug("[EXTRACT] extract_relationships called for cluster %s", cluster.cluster_id)
    extractor = RelationshipExtractor(client=client)
    return extractor.extract_relationships(cluster, market_info)


def build_relationship_graph(
    markets: list[MarketInfo],
    client: LLMClient | None = None,
) -> RelationshipGraph:
    """Build a complete relationship graph from markets.
    
    Only includes clusters with 3+ markets for partition-based arbitrage.
    """
    from .clustering import MarketClusterer

    logger.info("[EXTRACT] Building relationship graph for %d markets", len(markets))
    start_time = time.time()

    if client is None:
        client = get_client()

    market_info = {m.id: m for m in markets}

    # Step 1: Cluster markets
    logger.info("[EXTRACT] Step 1: Clustering markets")
    clusterer = MarketClusterer(client=client)
    clusters = clusterer.cluster_markets(markets)

    # Step 2: Extract relationships for clusters with 3+ markets
    logger.info("[EXTRACT] Step 2: Extracting partition constraints (3+ market clusters only)")
    extractor = RelationshipExtractor(client=client)
    enriched_clusters = extractor.extract_for_all_clusters(clusters, market_info)

    # Step 3: Build the graph
    graph = RelationshipGraph(
        clusters=enriched_clusters,
        model_used=client.model,
    )

    elapsed = time.time() - start_time
    logger.info("[EXTRACT] Graph built: %d clusters, %d relationships in %.3fs",
                len(graph.clusters), graph.total_relationships, elapsed)

    return graph
