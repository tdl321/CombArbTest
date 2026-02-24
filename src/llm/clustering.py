"""Semantic Market Clustering (LLM-02).

Groups semantically related Polymarket markets using LLM analysis.
Example: "Trump wins PA", "Trump wins election", "Republican wins" -> same cluster
"""

import json
import logging
from typing import Any

from .client import LLMClient, get_client
from .schema import MarketCluster, MarketInfo, RelationshipGraph


logger = logging.getLogger(__name__)


# System prompt for clustering task
CLUSTERING_SYSTEM_PROMPT = """You are an expert analyst for prediction markets. Your task is to group semantically related markets into clusters.

Markets are related if they:
1. Refer to the same event or outcome (e.g., election results)
2. Have logical dependencies (e.g., "Trump wins PA" relates to "Trump wins election")
3. Share the same subject matter (e.g., all about 2024 US election)
4. Have mutually exclusive outcomes (e.g., "Trump wins" vs "Biden wins")

For each cluster, provide:
- A unique cluster_id (slug format, e.g., "us-2024-presidential")
- A descriptive theme (e.g., "2024 US Presidential Election")
- List of market IDs that belong together

Markets that don't fit any cluster should go in individual clusters.

Respond with valid JSON only."""


CLUSTERING_USER_PROMPT_TEMPLATE = """Analyze these prediction markets and group semantically related ones into clusters.

Markets to analyze:
{markets_json}

Return a JSON object with this structure:
{{
    "clusters": [
        {{
            "cluster_id": "unique-slug-id",
            "theme": "Human readable theme description",
            "market_ids": ["market_id_1", "market_id_2", ...]
        }}
    ],
    "reasoning": "Brief explanation of clustering logic"
}}

Group markets that are semantically related. Each market should appear in exactly one cluster."""


class MarketClusterer:
    """Groups semantically related markets using LLM analysis.
    
    Usage:
        clusterer = MarketClusterer()
        markets = [
            MarketInfo(id="1", question="Will Trump win Pennsylvania?", outcomes=["Yes", "No"]),
            MarketInfo(id="2", question="Will Trump win the 2024 election?", outcomes=["Yes", "No"]),
            MarketInfo(id="3", question="Will Bitcoin hit 100k?", outcomes=["Yes", "No"]),
        ]
        clusters = clusterer.cluster_markets(markets)
    """
    
    def __init__(
        self,
        client: LLMClient | None = None,
        max_markets_per_batch: int = 30,
        temperature: float = 0.2,
    ):
        """Initialize the clusterer.
        
        Args:
            client: LLM client instance (uses singleton if not provided)
            max_markets_per_batch: Max markets to send in one LLM call
            temperature: LLM temperature (lower = more deterministic)
        """
        self._client = client
        self.max_markets_per_batch = max_markets_per_batch
        self.temperature = temperature
    
    @property
    def client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = get_client()
        return self._client
    
    def cluster_markets(
        self,
        markets: list[MarketInfo],
    ) -> list[MarketCluster]:
        """Group markets into semantic clusters.
        
        Args:
            markets: List of markets to cluster
            
        Returns:
            List of MarketCluster objects
        """
        if not markets:
            return []
        
        # For small batches, cluster directly
        if len(markets) <= self.max_markets_per_batch:
            return self._cluster_batch(markets)
        
        # For larger sets, batch and merge clusters
        all_clusters = []
        for i in range(0, len(markets), self.max_markets_per_batch):
            batch = markets[i:i + self.max_markets_per_batch]
            batch_clusters = self._cluster_batch(batch)
            all_clusters.extend(batch_clusters)
        
        # Merge similar clusters across batches
        return self._merge_similar_clusters(all_clusters)
    
    def _cluster_batch(self, markets: list[MarketInfo]) -> list[MarketCluster]:
        """Cluster a single batch of markets."""
        # Format markets for the prompt
        markets_data = [
            {
                "id": m.id,
                "question": m.question,
                "outcomes": m.outcomes,
            }
            for m in markets
        ]
        markets_json = json.dumps(markets_data, indent=2)
        
        prompt = CLUSTERING_USER_PROMPT_TEMPLATE.format(markets_json=markets_json)
        
        logger.info(f"Clustering {len(markets)} markets...")
        
        try:
            response = self.client.chat_json(
                prompt=prompt,
                system=CLUSTERING_SYSTEM_PROMPT,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"LLM clustering failed: {e}")
            # Fallback: each market in its own cluster
            return self._fallback_individual_clusters(markets)
        
        return self._parse_clustering_response(response, markets)
    
    def _parse_clustering_response(
        self,
        response: dict[str, Any],
        markets: list[MarketInfo],
    ) -> list[MarketCluster]:
        """Parse LLM response into MarketCluster objects."""
        clusters = []
        seen_market_ids = set()
        valid_market_ids = {m.id for m in markets}
        
        for cluster_data in response.get("clusters", []):
            cluster_id = cluster_data.get("cluster_id", "unknown")
            theme = cluster_data.get("theme", "Unknown Theme")
            market_ids = cluster_data.get("market_ids", [])
            
            # Filter to only valid market IDs
            valid_ids = [
                mid for mid in market_ids 
                if mid in valid_market_ids and mid not in seen_market_ids
            ]
            
            if valid_ids:
                clusters.append(MarketCluster(
                    cluster_id=cluster_id,
                    theme=theme,
                    market_ids=valid_ids,
                ))
                seen_market_ids.update(valid_ids)
        
        # Add unclustered markets to individual clusters
        unclustered = [m for m in markets if m.id not in seen_market_ids]
        for market in unclustered:
            clusters.append(MarketCluster(
                cluster_id=f"single-{market.id[:8]}",
                theme=market.question[:50],
                market_ids=[market.id],
            ))
        
        return clusters
    
    def _fallback_individual_clusters(
        self,
        markets: list[MarketInfo],
    ) -> list[MarketCluster]:
        """Fallback: put each market in its own cluster."""
        return [
            MarketCluster(
                cluster_id=f"single-{m.id[:8]}",
                theme=m.question[:50],
                market_ids=[m.id],
            )
            for m in markets
        ]
    
    def _merge_similar_clusters(
        self,
        clusters: list[MarketCluster],
    ) -> list[MarketCluster]:
        """Merge clusters with similar themes.
        
        TODO: Use LLM to identify similar clusters across batches.
        For now, just returns the clusters as-is.
        """
        # Simple implementation: no merging
        # A more sophisticated version would use embeddings or another
        # LLM call to identify similar cluster themes
        return clusters


def cluster_markets(
    markets: list[MarketInfo],
    client: LLMClient | None = None,
) -> list[MarketCluster]:
    """Convenience function to cluster markets.
    
    Args:
        markets: List of MarketInfo objects
        client: Optional LLM client
        
    Returns:
        List of MarketCluster objects
    """
    clusterer = MarketClusterer(client=client)
    return clusterer.cluster_markets(markets)
