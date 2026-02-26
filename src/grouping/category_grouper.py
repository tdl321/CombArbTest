"""Category-based market grouper.

Wraps the existing RuleBasedCategorizer from src/llm/categorizer.py
as a MarketGrouper protocol implementation. Groups markets by their
rule-based category assignment.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.core.types import (
    GroupingType,
    MarketGroup,
    MarketMeta,
)
from src.core.protocols import MarketDataSource

logger = logging.getLogger(__name__)


class CategoryGrouper:
    """Group markets by rule-based category assignment.

    Delegates categorization to the existing RuleBasedCategorizer,
    then groups markets that share the same category+subcategory.
    """

    grouping_type = GroupingType.CATEGORY

    def __init__(self, min_group_size: int = 2):
        self.min_group_size = min_group_size

    def group(
        self,
        markets: list[MarketMeta],
        data_source: MarketDataSource | None = None,
    ) -> list[MarketGroup]:
        """Group markets by category using RuleBasedCategorizer."""
        from src.llm.categorizer import RuleBasedCategorizer, MarketInfo

        categorizer = RuleBasedCategorizer()

        # Convert to MarketInfo format expected by categorizer
        market_infos = [
            MarketInfo(id=m.id, question=m.question, slug=m.slug)
            for m in markets
        ]

        results = categorizer.categorize_batch(market_infos)

        # Group by category + subcategory
        groups_map: dict[str, list[str]] = defaultdict(list)
        category_names: dict[str, tuple[str, str]] = {}

        for result in results:
            key = f"{result.category}/{result.subcategory}"
            groups_map[key].append(result.market_id)
            category_names[key] = (result.category, result.subcategory)

        # Build MarketGroup objects
        groups = []
        for key, market_ids in groups_map.items():
            if len(market_ids) < self.min_group_size:
                continue

            category, subcategory = category_names[key]
            groups.append(MarketGroup(
                group_id=f"cat_{key.replace(chr(47), chr(95))}",
                name=f"{category} / {subcategory}",
                market_ids=market_ids,
                group_type=GroupingType.CATEGORY,
                constraints=[],  # No logical constraints for category grouping
                is_partition=False,
                metadata={
                    "category": category,
                    "subcategory": subcategory,
                },
            ))

        logger.info(
            "[GROUPER] Category grouping: %d markets -> %d groups",
            len(markets), len(groups),
        )
        return groups
