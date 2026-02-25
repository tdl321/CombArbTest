"""Two-Stage Market Clustering with LLM.

Stage 1: Group semantically related markets
Stage 2: Extract partition constraints for clusters with 3+ markets

Focus on PARTITION detection for combinatorial arbitrage:
- A partition is a set of 3+ markets that are BOTH exhaustive AND mutually exclusive
- Partitions have the constraint: Σ P(markets) = 1 exactly
- Binary (Yes/No) markets are NOT valid partitions
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

# Minimum markets required for partition-based arbitrage
MIN_PARTITION_MARKETS = 3


# =============================================================================
# PROMPTS
# =============================================================================

CLUSTERING_SYSTEM = """You are analyzing prediction markets to group semantically related ones.

Group markets that:
1. Reference the same event (e.g., "2024 Presidential Election")
2. Have competing outcomes (e.g., different candidates winning)
3. Share the same subject (e.g., Bitcoin price targets)

IMPORTANT: Prioritize creating groups of 3 or more markets that could form PARTITIONS
(exhaustive and mutually exclusive sets where exactly one outcome must occur).

Markets without clear relationships go in individual clusters.
Respond with valid JSON only."""

CLUSTERING_USER = """Group these prediction markets by semantic relationships.

IMPORTANT: Prioritize grouping markets that form PARTITIONS (3+ markets where
exactly one outcome must occur, like election candidates or championship winners).

Markets:
{markets_json}

Return JSON:
{{
    "clusters": [
        {{
            "cluster_id": "descriptive-slug",
            "theme": "Description",
            "market_ids": ["id1", "id2", "id3"]
        }}
    ]
}}"""

CONSTRAINTS_SYSTEM = """Identify PARTITION constraints between prediction markets for combinatorial arbitrage.

## PARTITION REQUIREMENTS:
- Must have 3 or more markets (NOT binary Yes/No markets)
- Must be BOTH exhaustive AND mutually exclusive
- Mathematical constraint: Σ P(outcomes) = 1 exactly

## Examples of Valid Partitions:
- Election winners (3+ candidates)
- Championship outcomes (3+ teams)  
- Score ranges (3+ buckets)
- Date/time ranges (3+ periods)

## What is NOT a Valid Partition:
- Binary Yes/No markets (only 2 outcomes)
- Simple pairwise constraints between 2 markets
- Markets that are correlated but not mutually exclusive

Be CONSERVATIVE. Only high-confidence partitions. Valid JSON only."""

CONSTRAINTS_USER = """Identify PARTITION constraints in this market cluster.

PARTITION REQUIREMENTS:
- Must have 3 or more markets
- Must be BOTH exhaustive AND mutually exclusive
- Binary (Yes/No) markets should NOT be extracted as partitions

Theme: {theme}

Markets:
{markets_json}

Return JSON:
{{
    "constraints": [
        {{
            "type": "mutually_exclusive",
            "from_market": "id",
            "to_market": "id",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation"
        }}
    ],
    "exhaustive_sets": [
        {{
            "market_ids": ["id1", "id2", "id3"],
            "is_partition": true,
            "confidence": 0.0-1.0,
            "reasoning": "Why these form a complete partition with 3+ outcomes"
        }}
    ]
}}

CRITICAL FILTERS:
- exhaustive_sets with fewer than 3 markets should NOT be returned
- Only return sets where EXACTLY ONE outcome must occur
- Set "is_partition": true when the set is both exhaustive AND mutually exclusive"""


class MarketClusterer:
    """Two-stage LLM clustering: grouping then partition extraction."""

    def __init__(
        self,
        client: LLMClient | None = None,
        max_batch_size: int = 30,
        temperature: float = 0.2,
        max_retries: int = 2,
        min_confidence: float = 0.7,
    ):
        self._client = client
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        logger.debug("[CLUSTER] MarketClusterer initialized: batch_size=%d, temp=%.1f, retries=%d",
                     max_batch_size, temperature, max_retries)

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client()
        return self._client

    def cluster_and_extract(self, markets: list[MarketInfo]) -> RelationshipGraph:
        """Main entry: cluster markets and extract partition constraints."""
        if not markets:
            logger.warning("[CLUSTER] No markets provided")
            return RelationshipGraph(clusters=[])

        logger.info("[CLUSTER] Starting with %d markets", len(markets))
        start_time = time.time()
        
        # Stage 1: Cluster markets
        clusters = self._stage1_cluster(markets)
        
        # Stage 2: Extract partition constraints for clusters with 3+ markets
        enriched = []
        skipped_count = 0
        for cluster in clusters:
            if len(cluster.market_ids) >= MIN_PARTITION_MARKETS:
                constraints = self._stage2_constraints(cluster, markets)
                cluster = MarketCluster(
                    cluster_id=cluster.cluster_id,
                    theme=cluster.theme,
                    market_ids=cluster.market_ids,
                    relationships=constraints,
                )
                enriched.append(cluster)
            else:
                skipped_count += 1
        
        graph = RelationshipGraph(
            clusters=enriched,
            model_used=getattr(self.client, "model", "unknown"),
        )
        
        elapsed = time.time() - start_time
        logger.info("[CLUSTER] Done: %d partition clusters, %d constraints in %.3fs (skipped %d small clusters)",
                    len(graph.clusters), graph.total_relationships, elapsed, skipped_count)
        
        # Log partition detection summary
        partition_count = sum(1 for c in graph.clusters if self._is_partition_cluster(c))
        if partition_count > 0:
            logger.info("[CLUSTER] Detected %d valid partition clusters for arbitrage", partition_count)
        
        return graph
    
    def _is_partition_cluster(self, cluster: MarketCluster) -> bool:
        """Check if a cluster represents a valid partition (3+ markets, exhaustive + exclusive)."""
        if len(cluster.market_ids) < MIN_PARTITION_MARKETS:
            return False
        
        if not cluster.relationships:
            return False
        
        has_exhaustive = any(r.type == "exhaustive" for r in cluster.relationships)
        exclusive_pairs = sum(
            1 for r in cluster.relationships 
            if r.type in ("mutually_exclusive", "incompatible") and r.to_market is not None
        )
        
        n = len(cluster.market_ids)
        expected_pairs = n * (n - 1) // 2
        
        return has_exhaustive and expected_pairs > 0 and exclusive_pairs >= expected_pairs * 0.5

    def _stage1_cluster(self, markets: list[MarketInfo]) -> list[MarketCluster]:
        """Stage 1: Group semantically related markets."""
        logger.info("[STAGE1] Semantic clustering for %d markets", len(markets))
        
        if len(markets) <= self.max_batch_size:
            return self._cluster_batch(markets)
        
        all_clusters = []
        batch_num = 0
        for i in range(0, len(markets), self.max_batch_size):
            batch_num += 1
            batch = markets[i:i + self.max_batch_size]
            logger.info("[STAGE1] Batch %d: %d markets", batch_num, len(batch))
            clusters = self._cluster_batch(batch)
            all_clusters.extend(clusters)
        
        logger.info("[STAGE1] All batches complete: %d total clusters", len(all_clusters))
        return all_clusters

    def _cluster_batch(self, markets: list[MarketInfo]) -> list[MarketCluster]:
        """Cluster a batch via LLM."""
        markets_data = [
            {"id": m.id, "question": m.question, "outcomes": m.outcomes}
            for m in markets
        ]
        
        prompt = CLUSTERING_USER.format(markets_json=json.dumps(markets_data, indent=2))
        
        logger.info("[STAGE1] LLM call: %d markets", len(markets))
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat_json(
                    prompt=prompt,
                    system=CLUSTERING_SYSTEM,
                    temperature=self.temperature,
                )
                clusters = self._parse_clusters(response, markets)
                elapsed = time.time() - start_time
                logger.info("[STAGE1] Result: %d clusters in %.3fs", len(clusters), elapsed)
                return clusters
                
            except Exception as e:
                logger.warning("[STAGE1] Attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    logger.error("[STAGE1] All attempts failed, using fallback")
                    return self._fallback_clusters(markets)
        
        return self._fallback_clusters(markets)

    def _parse_clusters(self, response: dict[str, Any], markets: list[MarketInfo]) -> list[MarketCluster]:
        """Parse LLM clustering response."""
        clusters = []
        seen_ids = set()
        valid_ids = {m.id for m in markets}
        
        for c in response.get("clusters", []):
            market_ids = [
                mid for mid in c.get("market_ids", [])
                if mid in valid_ids and mid not in seen_ids
            ]
            if market_ids:
                clusters.append(MarketCluster(
                    cluster_id=c.get("cluster_id", "unknown"),
                    theme=c.get("theme", "Unknown"),
                    market_ids=market_ids,
                ))
                seen_ids.update(market_ids)
        
        # Handle unclustered markets - but don't create single-market clusters
        # since they can't form partitions anyway
        unclustered = [m for m in markets if m.id not in seen_ids]
        if unclustered:
            logger.debug("[STAGE1] %d markets unclustered (will be skipped - no partition possible)", 
                        len(unclustered))
        
        return clusters

    def _fallback_clusters(self, markets: list[MarketInfo]) -> list[MarketCluster]:
        """Fallback: try to create one big cluster if possible."""
        logger.warning("[STAGE1] Using fallback clustering")
        if len(markets) >= MIN_PARTITION_MARKETS:
            return [
                MarketCluster(
                    cluster_id="fallback-all",
                    theme="All markets (fallback)",
                    market_ids=[m.id for m in markets],
                )
            ]
        return []

    def _stage2_constraints(self, cluster: MarketCluster, all_markets: list[MarketInfo]) -> list[MarketRelationship]:
        """Stage 2: Extract partition constraints for a cluster with 3+ markets."""
        theme_short = cluster.theme[:40] + "..." if len(cluster.theme) > 40 else cluster.theme
        logger.info("[STAGE2] Extracting partition constraints for: %s (%d markets)", 
                    theme_short, len(cluster.market_ids))
        
        market_map = {m.id: m for m in all_markets}
        cluster_markets = [market_map[mid] for mid in cluster.market_ids if mid in market_map]
        
        if len(cluster_markets) < MIN_PARTITION_MARKETS:
            logger.debug("[STAGE2] Cluster has < %d markets, skipping", MIN_PARTITION_MARKETS)
            return []
        
        markets_data = [
            {"id": m.id, "question": m.question, "outcomes": m.outcomes}
            for m in cluster_markets
        ]
        
        prompt = CONSTRAINTS_USER.format(
            theme=cluster.theme,
            markets_json=json.dumps(markets_data, indent=2),
        )
        
        logger.info("[STAGE2] LLM call: %d markets", len(cluster_markets))
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat_json(
                    prompt=prompt,
                    system=CONSTRAINTS_SYSTEM,
                    temperature=self.temperature,
                )
                constraints = self._parse_constraints(response, cluster.market_ids)
                
                elapsed = time.time() - start_time
                by_type = {}
                for c in constraints:
                    by_type[c.type] = by_type.get(c.type, 0) + 1
                logger.info("[STAGE2] Result: %d constraints %s in %.3fs", len(constraints), dict(by_type), elapsed)
                
                return constraints
                
            except Exception as e:
                logger.warning("[STAGE2] Attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    logger.error("[STAGE2] All attempts failed, no constraints extracted")
                    return []
        
        return []

    def _parse_constraints(self, response: dict[str, Any], valid_ids: list[str]) -> list[MarketRelationship]:
        """Parse LLM constraints response for partition constraints only."""
        relationships = []
        valid_set = set(valid_ids)
        
        # Only accept partition-related constraint types
        partition_types = {"mutually_exclusive", "exhaustive"}
        
        # Parse individual constraints
        for c in response.get("constraints", []):
            rel_type = c.get("type", "")
            from_id = c.get("from_market")
            to_id = c.get("to_market")
            confidence = float(c.get("confidence", 0))
            
            if rel_type not in partition_types:
                logger.debug("[STAGE2] Skipping non-partition type: %s", rel_type)
                continue
            if from_id not in valid_set:
                logger.debug("[STAGE2] Skipping invalid from_market: %s", from_id)
                continue
            if to_id and to_id not in valid_set:
                logger.debug("[STAGE2] Skipping invalid to_market: %s", to_id)
                continue
            if confidence < self.min_confidence:
                logger.debug("[STAGE2] Skipping low confidence constraint: %.2f", confidence)
                continue
            
            relationships.append(MarketRelationship(
                type=rel_type,
                from_market=from_id,
                to_market=to_id,
                confidence=confidence,
                reasoning=c.get("reasoning"),
            ))
        
        # Parse exhaustive sets (require 3+ markets for partitions)
        for ex in response.get("exhaustive_sets", []):
            ids = [mid for mid in ex.get("market_ids", []) if mid in valid_set]
            confidence = float(ex.get("confidence", 0))
            reasoning = ex.get("reasoning", "")
            is_partition = ex.get("is_partition", False)
            
            # CRITICAL: Require 3+ markets for valid partitions
            if len(ids) < MIN_PARTITION_MARKETS:
                logger.debug("[STAGE2] Skipping exhaustive set with < %d markets", MIN_PARTITION_MARKETS)
                continue
                
            if confidence < self.min_confidence:
                logger.debug("[STAGE2] Skipping low confidence exhaustive set: %.2f", confidence)
                continue
            
            # Add pairwise mutual exclusivity constraints
            for i, id1 in enumerate(ids):
                for id2 in ids[i+1:]:
                    relationships.append(MarketRelationship(
                        type="mutually_exclusive",
                        from_market=id1,
                        to_market=id2,
                        confidence=confidence,
                        reasoning=reasoning,
                    ))
            
            # Add exhaustive constraint
            relationships.append(MarketRelationship(
                type="exhaustive",
                from_market=ids[0],
                to_market=None,
                confidence=confidence,
                reasoning="Partition (%d markets): %s" % (len(ids), reasoning),
            ))
            
            logger.info("[STAGE2] Detected partition: %d markets (%s)", len(ids), reasoning[:50])
        
        return relationships


def cluster_markets(markets: list[MarketInfo], client: LLMClient | None = None) -> list[MarketCluster]:
    """Cluster markets (Stage 1 only)."""
    logger.debug("[CLUSTER] cluster_markets called with %d markets", len(markets))
    clusterer = MarketClusterer(client=client)
    graph = clusterer.cluster_and_extract(markets)
    return graph.clusters


def build_relationship_graph(markets: list[MarketInfo], client: LLMClient | None = None) -> RelationshipGraph:
    """Full pipeline: cluster + extract partition constraints.
    
    Only includes clusters with 3+ markets for combinatorial arbitrage.
    """
    logger.debug("[CLUSTER] build_relationship_graph called with %d markets", len(markets))
    clusterer = MarketClusterer(client=client)
    return clusterer.cluster_and_extract(markets)


# =============================================================================
# COMPLEX CONSTRAINT PROMPTS (for non-partition logical relationships)
# =============================================================================

COMPLEX_CONSTRAINTS_SYSTEM = """Identify LOGICAL CONSTRAINTS between prediction markets for combinatorial arbitrage.

## CONSTRAINT TYPES TO DETECT:

### 1. IMPLIES (A → B)
- If A is true, B must also be true
- Constraint: P(A) ≤ P(B)
- Example: "Trump wins presidency" IMPLIES "Trump is inaugurated"
- Example: "Team wins championship" IMPLIES "Team made playoffs"

### 2. PREREQUISITE (B requires A)
- B cannot happen without A happening first
- Constraint: P(B) ≤ P(A)
- Example: "Bill becomes law" REQUIRES "Bill passes Senate"
- Example: "Wins finals" REQUIRES "Wins semifinals"

### 3. MUTUALLY_EXCLUSIVE (A XOR B)
- A and B cannot both be true
- Constraint: P(A) + P(B) ≤ 1
- Example: "Trump wins" and "Harris wins" cannot both happen

### 4. INCOMPATIBLE (structural impossibility)
- Events are structurally impossible together
- Constraint: P(A) + P(B) ≤ 1
- Example: Two teams meeting in semifinals cannot BOTH win 6 games

IMPORTANT: Binary Yes/No markets CAN have logical relationships with OTHER markets.
Focus on cross-market implications, not internal Yes/No structure.

Return high-confidence constraints only. Valid JSON only."""

COMPLEX_CONSTRAINTS_USER = """Identify LOGICAL CONSTRAINTS between these prediction markets.

Theme: {theme}

Markets:
{markets_json}

Return JSON:
{{
    "constraints": [
        {{
            "type": "implies|prerequisite|mutually_exclusive|incompatible",
            "from_market": "market_id",
            "to_market": "market_id",
            "confidence": 0.0-1.0,
            "reasoning": "Why this logical relationship exists"
        }}
    ]
}}

Focus on:
- Cross-market implications (A winning implies B must also happen)
- Temporal prerequisites (A must happen before B can happen)
- Logical impossibilities (A and B cannot both be true)"""


class ComplexConstraintExtractor:
    """Extract complex logical constraints (implies, prerequisite, etc.) between markets.
    
    Unlike the partition-focused MarketClusterer, this class extracts constraints
    that require the Frank-Wolfe solver to evaluate (not simple sum = 1 checks).
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        temperature: float = 0.2,
        max_retries: int = 2,
        min_confidence: float = 0.7,
    ):
        self._client = client
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        logger.debug("[COMPLEX] ComplexConstraintExtractor initialized")

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client()
        return self._client

    def extract_constraints(
        self,
        markets: list[MarketInfo],
        theme: str = "Related markets",
    ) -> list[MarketRelationship]:
        """Extract complex logical constraints from a set of markets.
        
        Returns constraints that require the solver (implies, prerequisite, etc.),
        NOT partition constraints (which can be checked algebraically).
        """
        if len(markets) < 2:
            logger.debug("[COMPLEX] Need at least 2 markets for constraints")
            return []

        logger.info("[COMPLEX] Extracting complex constraints: %d markets, theme=%s",
                    len(markets), theme[:40])

        markets_data = [
            {"id": m.id, "question": m.question, "outcomes": m.outcomes}
            for m in markets
        ]

        prompt = COMPLEX_CONSTRAINTS_USER.format(
            theme=theme,
            markets_json=json.dumps(markets_data, indent=2),
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat_json(
                    prompt=prompt,
                    system=COMPLEX_CONSTRAINTS_SYSTEM,
                    temperature=self.temperature,
                )
                constraints = self._parse_complex_constraints(response, [m.id for m in markets])
                logger.info("[COMPLEX] Extracted %d constraints", len(constraints))
                return constraints

            except Exception as e:
                logger.warning("[COMPLEX] Attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    logger.error("[COMPLEX] All attempts failed")
                    return []

        return []

    def _parse_complex_constraints(
        self,
        response: dict[str, Any],
        valid_ids: list[str],
    ) -> list[MarketRelationship]:
        """Parse LLM response for complex constraints.
        
        Accepts: implies, prerequisite, mutually_exclusive, incompatible, and, or
        (NOT exhaustive - that is for partitions)
        """
        relationships = []
        valid_set = set(valid_ids)

        # Valid complex constraint types (not partition types)
        complex_types = {"implies", "prerequisite", "mutually_exclusive", "incompatible", "and", "or"}

        for c in response.get("constraints", []):
            rel_type = c.get("type", "")
            from_id = c.get("from_market")
            to_id = c.get("to_market")
            confidence = float(c.get("confidence", 0))
            reasoning = c.get("reasoning", "")

            if rel_type not in complex_types:
                logger.debug("[COMPLEX] Skipping non-complex type: %s", rel_type)
                continue
            if from_id not in valid_set:
                logger.debug("[COMPLEX] Skipping invalid from_market: %s", from_id)
                continue
            if to_id not in valid_set:
                logger.debug("[COMPLEX] Skipping invalid to_market: %s", to_id)
                continue
            if confidence < self.min_confidence:
                logger.debug("[COMPLEX] Skipping low confidence constraint: %.2f", confidence)
                continue

            relationships.append(MarketRelationship(
                type=rel_type,
                from_market=from_id,
                to_market=to_id,
                confidence=confidence,
                reasoning=reasoning,
            ))
            logger.debug("[COMPLEX] Added %s: %s -> %s (conf=%.2f)",
                        rel_type, from_id, to_id, confidence)

        return relationships


def extract_complex_constraints(
    markets: list[MarketInfo],
    theme: str = "Related markets",
    client: LLMClient | None = None,
) -> RelationshipGraph:
    """Extract complex logical constraints and build a RelationshipGraph.
    
    This is the main entry point for complex constraint extraction.
    Returns a graph with is_partition=False clusters that require solver evaluation.
    """
    logger.info("[COMPLEX] extract_complex_constraints called with %d markets", len(markets))

    extractor = ComplexConstraintExtractor(client=client)
    constraints = extractor.extract_constraints(markets, theme)

    if not constraints:
        logger.warning("[COMPLEX] No complex constraints found")
        return RelationshipGraph(clusters=[])

    # Build a single cluster with all markets and constraints
    # is_partition=False ensures solver is used (not algebraic check)
    cluster = MarketCluster(
        cluster_id="complex-constraints",
        theme=theme,
        market_ids=[m.id for m in markets],
        relationships=constraints,
        is_partition=False,  # CRITICAL: Forces solver path
    )

    return RelationshipGraph(
        clusters=[cluster],
        model_used=getattr(extractor.client, "model", "unknown"),
    )
