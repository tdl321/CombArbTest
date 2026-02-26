"""Manual market grouper for hardcoded tournament/partition definitions.

Used when market groupings are known in advance (e.g., NBA MVP candidates,
presidential election candidates) and don't require LLM clustering.
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


class ManualGrouper:
    """Group markets using hardcoded definitions.

    Accepts a list of pre-defined groups (e.g., tournament partitions)
    and returns them as MarketGroup objects.
    """

    grouping_type = GroupingType.MANUAL

    def __init__(self, group_definitions: list[dict[str, Any]] | None = None):
        """Initialize with pre-defined group definitions.

        Args:
            group_definitions: List of dicts with keys:
                - group_id: str
                - name: str
                - market_ids: list[str]
                - is_partition: bool (default True)
                - constraints: list[dict] (optional)
        """
        self._definitions = group_definitions or []

    def group(
        self,
        markets: list[MarketMeta],
        data_source: MarketDataSource | None = None,
    ) -> list[MarketGroup]:
        """Return pre-defined groups, filtering to only include available markets."""
        available_ids = {m.id for m in markets}
        groups = []

        for defn in self._definitions:
            # Filter to only markets that exist in the input
            group_ids = [
                mid for mid in defn["market_ids"]
                if mid in available_ids
            ]

            if len(group_ids) < 2:
                continue

            is_partition = defn.get("is_partition", True)

            # Build constraints
            constraints = []
            if is_partition:
                # Add exhaustive + mutually_exclusive for all pairs
                for i, mid_a in enumerate(group_ids):
                    for mid_b in group_ids[i + 1:]:
                        constraints.append(Constraint(
                            type=ConstraintType.MUTUALLY_EXCLUSIVE,
                            from_market=mid_a,
                            to_market=mid_b,
                            confidence=1.0,
                        ))
                constraints.append(Constraint(
                    type=ConstraintType.EXHAUSTIVE,
                    from_market=group_ids[0],
                    confidence=1.0,
                    reasoning="Partition: exactly one outcome must occur",
                ))

            # Add any custom constraints from definition
            for c_dict in defn.get("constraints", []):
                constraints.append(Constraint(**c_dict))

            groups.append(MarketGroup(
                group_id=defn["group_id"],
                name=defn.get("name", defn["group_id"]),
                market_ids=group_ids,
                group_type=GroupingType.MANUAL,
                constraints=constraints,
                is_partition=is_partition,
                metadata=defn.get("metadata", {}),
            ))

        logger.info(
            "[GROUPER] Manual grouping: %d definitions -> %d groups (with available markets)",
            len(self._definitions), len(groups),
        )
        return groups

    def add_group(
        self,
        group_id: str,
        name: str,
        market_ids: list[str],
        is_partition: bool = True,
        **metadata,
    ) -> None:
        """Add a group definition."""
        self._definitions.append({
            "group_id": group_id,
            "name": name,
            "market_ids": market_ids,
            "is_partition": is_partition,
            "metadata": metadata,
        })
