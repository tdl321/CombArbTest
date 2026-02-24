"""Dependency Extraction (LLM-03).

Extracts logical relationships between markets within clusters.
Relationship types: IMPLIES, MUTUALLY_EXCLUSIVE, AND, OR, PREREQUISITE
"""

import json
import logging
import re
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


EXTRACTION_SYSTEM_PROMPT = """You are an expert in logical reasoning and prediction markets. Your task is to identify logical relationships between prediction markets.

Relationship types:
1. IMPLIES: If market A resolves Yes, market B must also resolve Yes
   Example: "Trump wins Pennsylvania" IMPLIES "Trump wins at least one swing state"

2. MUTUALLY_EXCLUSIVE: Markets A and B cannot both resolve Yes
   Example: "Trump wins election" and "Biden wins election" are mutually exclusive

3. AND: Both markets must resolve the same way (strongly correlated)
   Example: "Democrats win House" AND "Democrats win Senate" (for a sweep scenario)

4. OR: At least one market must resolve Yes
   Example: "Trump wins PA" OR "Trump wins GA" OR "Trump wins AZ" (for victory path)

5. PREREQUISITE: Market A must happen before/for market B to be possible
   Example: "Trump nominated" is PREREQUISITE for "Trump wins general election"

For each relationship, provide:
- type: One of [implies, mutually_exclusive, and, or, prerequisite]
- from_market: The source market ID
- to_market: The target market ID (for directional relationships)
- confidence: How confident you are (0.0-1.0)
- reasoning: Brief explanation

Only identify relationships you are confident about. Do not force relationships.

IMPORTANT: Respond with valid JSON only. No markdown, no extra text."""


EXTRACTION_USER_PROMPT_TEMPLATE = """Analyze these related prediction markets and identify logical relationships between them.

Cluster Theme: {theme}

Markets:
{markets_json}

Return a JSON object with this structure:
{{
    "relationships": [
        {{
            "type": "implies|mutually_exclusive|and|or|prerequisite",
            "from_market": "market_id",
            "to_market": "market_id",
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }}
    ],
    "analysis_notes": "Any overall observations about this cluster"
}}

Only include relationships you are reasonably confident about (confidence >= 0.6).
If no strong relationships exist, return an empty relationships array.

Return ONLY valid JSON, no markdown formatting."""


class RelationshipExtractor:
    """Extracts logical relationships between markets using LLM analysis.
    
    Usage:
        extractor = RelationshipExtractor()
        
        # With a cluster and market info
        markets = {
            "1": MarketInfo(id="1", question="Trump wins PA?", outcomes=["Yes", "No"]),
            "2": MarketInfo(id="2", question="Trump wins election?", outcomes=["Yes", "No"]),
        }
        cluster = MarketCluster(
            cluster_id="trump-2024",
            theme="Trump 2024 Election",
            market_ids=["1", "2"],
        )
        
        relationships = extractor.extract_relationships(cluster, markets)
    """
    
    def __init__(
        self,
        client: LLMClient | None = None,
        min_confidence: float = 0.6,
        temperature: float = 0.2,
        max_retries: int = 2,
    ):
        """Initialize the extractor.
        
        Args:
            client: LLM client instance (uses singleton if not provided)
            min_confidence: Minimum confidence threshold for relationships
            temperature: LLM temperature (lower = more deterministic)
            max_retries: Number of retries on JSON parse failure
        """
        self._client = client
        self.min_confidence = min_confidence
        self.temperature = temperature
        self.max_retries = max_retries
    
    @property
    def client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = get_client()
        return self._client
    
    def extract_relationships(
        self,
        cluster: MarketCluster,
        market_info: dict[str, MarketInfo],
    ) -> list[MarketRelationship]:
        """Extract relationships between markets in a cluster.
        
        Args:
            cluster: The market cluster to analyze
            market_info: Dict mapping market_id to MarketInfo
            
        Returns:
            List of MarketRelationship objects
        """
        # Skip clusters with only one market
        if len(cluster.market_ids) < 2:
            return []
        
        # Build market data for the prompt
        markets_data = []
        for market_id in cluster.market_ids:
            info = market_info.get(market_id)
            if info:
                markets_data.append({
                    "id": info.id,
                    "question": info.question,
                    "outcomes": info.outcomes,
                })
        
        if len(markets_data) < 2:
            return []
        
        markets_json = json.dumps(markets_data, indent=2)
        prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            theme=cluster.theme,
            markets_json=markets_json,
        )
        
        logger.info(
            f"Extracting relationships for cluster '{cluster.cluster_id}' "
            f"with {len(markets_data)} markets..."
        )
        
        # Try extraction with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat_json(
                    prompt=prompt,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    temperature=self.temperature,
                )
                return self._parse_extraction_response(response, cluster.market_ids)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse error on attempt {attempt + 1} for cluster "
                    f"{cluster.cluster_id}: {e}"
                )
                if attempt < self.max_retries:
                    # Try with a slightly different prompt or temperature
                    self.temperature = min(0.5, self.temperature + 0.1)
                    continue
                else:
                    logger.error(
                        f"LLM extraction failed after {self.max_retries + 1} attempts "
                        f"for cluster {cluster.cluster_id}"
                    )
                    return []
            except Exception as e:
                logger.error(f"LLM extraction failed for cluster {cluster.cluster_id}: {e}")
                return []
        
        return []
    
    def _extract_json_from_response(self, raw_response: str) -> dict[str, Any]:
        """Extract JSON from potentially markdown-wrapped response."""
        # Try direct parse first
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_pattern, raw_response)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object directly
        json_obj_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_obj_pattern, raw_response)
        if matches:
            # Try the last match (often the complete one)
            for match in reversed(matches):
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        raise json.JSONDecodeError("Could not extract JSON from response", raw_response, 0)
    
    def _parse_extraction_response(
        self,
        response: dict[str, Any],
        valid_market_ids: list[str],
    ) -> list[MarketRelationship]:
        """Parse LLM response into MarketRelationship objects."""
        relationships = []
        valid_ids_set = set(valid_market_ids)
        valid_types = {t.value for t in RelationshipType}
        
        for rel_data in response.get("relationships", []):
            rel_type = rel_data.get("type", "").lower()
            from_market = rel_data.get("from_market")
            to_market = rel_data.get("to_market")
            confidence = rel_data.get("confidence", 0.0)
            reasoning = rel_data.get("reasoning")
            
            # Validate relationship type
            if rel_type not in valid_types:
                logger.warning(f"Invalid relationship type: {rel_type}")
                continue
            
            # Validate market IDs
            if from_market not in valid_ids_set:
                logger.warning(f"Invalid from_market: {from_market}")
                continue
            
            if to_market and to_market not in valid_ids_set:
                logger.warning(f"Invalid to_market: {to_market}")
                continue
            
            # Validate confidence
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.5
            
            # Filter by minimum confidence
            if confidence < self.min_confidence:
                continue
            
            relationships.append(MarketRelationship(
                type=rel_type,
                from_market=from_market,
                to_market=to_market,
                confidence=confidence,
                reasoning=reasoning,
            ))
        
        return relationships
    
    def extract_for_all_clusters(
        self,
        clusters: list[MarketCluster],
        market_info: dict[str, MarketInfo],
    ) -> list[MarketCluster]:
        """Extract relationships for all clusters.
        
        Args:
            clusters: List of clusters to analyze
            market_info: Dict mapping market_id to MarketInfo
            
        Returns:
            List of clusters with relationships populated
        """
        enriched_clusters = []
        
        for cluster in clusters:
            relationships = self.extract_relationships(cluster, market_info)
            
            # Create a new cluster with relationships
            enriched_cluster = MarketCluster(
                cluster_id=cluster.cluster_id,
                theme=cluster.theme,
                market_ids=cluster.market_ids,
                relationships=relationships,
            )
            enriched_clusters.append(enriched_cluster)
        
        return enriched_clusters


def extract_relationships(
    cluster: MarketCluster,
    market_info: dict[str, MarketInfo],
    client: LLMClient | None = None,
) -> list[MarketRelationship]:
    """Convenience function to extract relationships for a cluster.
    
    Args:
        cluster: The market cluster to analyze
        market_info: Dict mapping market_id to MarketInfo
        client: Optional LLM client
        
    Returns:
        List of MarketRelationship objects
    """
    extractor = RelationshipExtractor(client=client)
    return extractor.extract_relationships(cluster, market_info)


def build_relationship_graph(
    markets: list[MarketInfo],
    client: LLMClient | None = None,
) -> RelationshipGraph:
    """Build a complete relationship graph from markets.
    
    This is the main entry point that combines clustering and extraction.
    
    Args:
        markets: List of MarketInfo objects
        client: Optional LLM client (shared for clustering and extraction)
        
    Returns:
        RelationshipGraph with clusters and relationships
    """
    from .clustering import MarketClusterer
    
    if client is None:
        client = get_client()
    
    # Build market info dict
    market_info = {m.id: m for m in markets}
    
    # Step 1: Cluster markets
    clusterer = MarketClusterer(client=client)
    clusters = clusterer.cluster_markets(markets)
    
    # Step 2: Extract relationships for each cluster
    extractor = RelationshipExtractor(client=client)
    enriched_clusters = extractor.extract_for_all_clusters(clusters, market_info)
    
    # Step 3: Build the graph
    return RelationshipGraph(
        clusters=enriched_clusters,
        model_used=client.model,
    )
