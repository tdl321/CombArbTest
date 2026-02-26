"""LLM-based semantic market grouper.

Wraps the existing MarketClusterer from src/llm/clustering.py
as a MarketGrouper protocol implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.types import (
    Constraint,
    ConstraintType,
    GroupingType,
    MarketGroup,
    MarketMeta,
)
from src.core.protocols import MarketDataSource

logger = logging.getLogger(__name__)


class LLMSemanticGrouper:
    """Group markets by semantic similarity using LLM clustering.

    Delegates to the existing MarketClusterer.cluster_and_extract()
    pipeline for LLM-based grouping and constraint extraction.
    """

    grouping_type = GroupingType.SEMANTIC

    def __init__(self, model: str | None = None, cache_db: str | None = None):
        self._model = model
        self._cache_db = cache_db

    def group(
        self,
        markets: list[MarketMeta],
        data_source: MarketDataSource | None = None,
    ) -> list[MarketGroup]:
        """Group markets using LLM semantic clustering."""
        from src.llm.clustering import MarketClusterer
        from src.llm.schema import MarketInfo

        if len(markets) < 2:
            return []

        # Convert MarketMeta to MarketInfo (what the clusterer expects)
        market_infos = [
            MarketInfo(id=m.id, question=m.question, slug=m.slug)
            for m in markets
        ]

        # Run the existing clustering pipeline
        clusterer = MarketClusterer()
        try:
            graph = clusterer.cluster_and_extract(market_infos)
        except Exception as e:
            logger.error("[GROUPER] LLM clustering failed: %s", e)
            return []

        # Convert RelationshipGraph clusters to MarketGroup
        groups = []
        for cluster in graph.clusters:
            constraints = [
                Constraint(
                    type=self._map_relationship_type(rel.type),
                    from_market=rel.from_market,
                    to_market=getattr(rel, "to_market", None),
                    confidence=getattr(rel, "confidence", 1.0),
                )
                for rel in cluster.relationships
            ]

            groups.append(MarketGroup(
                group_id=cluster.cluster_id,
                name=getattr(cluster, "theme", "") or f"cluster_{cluster.cluster_id}",
                market_ids=cluster.market_ids,
                group_type=GroupingType.SEMANTIC,
                constraints=constraints,
                is_partition=getattr(cluster, "is_partition", False),
                metadata={
                    "source": "llm",
                    "num_relationships": len(cluster.relationships),
                },
            ))

        logger.info(
            "[GROUPER] LLM grouping: %d markets -> %d groups",
            len(markets), len(groups),
        )
        return groups

    @staticmethod
    def _map_relationship_type(rel_type: str) -> ConstraintType:
        """Map LLM relationship type string to ConstraintType enum."""
        mapping = {
            "mutually_exclusive": ConstraintType.MUTUALLY_EXCLUSIVE,
            "exhaustive": ConstraintType.EXHAUSTIVE,
            "implies": ConstraintType.IMPLIES,
            "prerequisite": ConstraintType.PREREQUISITE,
            "incompatible": ConstraintType.INCOMPATIBLE,
            "equivalent": ConstraintType.EQUIVALENT,
            "opposite": ConstraintType.OPPOSITE,
        }
        return mapping.get(rel_type, ConstraintType.MUTUALLY_EXCLUSIVE)
