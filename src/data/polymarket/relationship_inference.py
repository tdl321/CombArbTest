"""Derive constraints from Polymarket event metadata.

Two strategies:
  A) negRisk=True events -> automatic MUTUALLY_EXCLUSIVE + EXHAUSTIVE
  B) negRisk=False events -> fall back to LLM-based grouping
"""

from __future__ import annotations

import logging
from itertools import combinations

from src.core.types import (
    Constraint,
    ConstraintType,
    GroupingType,
    MarketGroup,
    MarketMeta,
)
from src.optimizer.schema import (
    MarketCluster,
    MarketRelationship,
    RelationshipGraph,
)

from .types import PolymarketEvent

logger = logging.getLogger(__name__)

# Constraint types that the solver's LMO can handle.
# EXHAUSTIVE is excluded because the LMO treats it as a no-op
# and there is a pre-existing AttributeError bug in the elif chain
# (references RelationshipType.EQUIVALENT which doesn't exist).
_SOLVER_SAFE_TYPES = {
    ConstraintType.MUTUALLY_EXCLUSIVE,
    ConstraintType.IMPLIES,
    ConstraintType.PREREQUISITE,
    ConstraintType.INCOMPATIBLE,
}


class RelationshipInferrer:
    """Infer constraints from Polymarket event structure.

    For negRisk=True events, all active markets form a mutually exclusive
    partition — this is guaranteed by the Polymarket protocol (CTF exchange).
    No LLM calls needed.

    For negRisk=False events, we delegate to the existing LLMSemanticGrouper.
    """

    def infer_from_event(
        self,
        event: PolymarketEvent,
        market_ids: list[str] | None = None,
    ) -> tuple[list[Constraint], bool]:
        """Infer constraints from a single event.

        Args:
            event: The Polymarket event
            market_ids: Specific market IDs to consider (defaults to all active)

        Returns:
            (constraints, is_partition) tuple
        """
        if market_ids is None:
            active = event.active_markets
            market_ids = [m.condition_id for m in active]

        if event.neg_risk and len(market_ids) > 1:
            return self._infer_neg_risk(market_ids), True

        return [], False

    def build_market_group(
        self,
        event: PolymarketEvent,
        market_ids: list[str] | None = None,
    ) -> MarketGroup:
        """Build a MarketGroup with inferred constraints.

        Args:
            event: The Polymarket event
            market_ids: Specific market IDs (defaults to all active)

        Returns:
            MarketGroup with constraints and partition flag
        """
        if market_ids is None:
            market_ids = [m.condition_id for m in event.active_markets]

        constraints, is_partition = self.infer_from_event(event, market_ids)

        group_type = GroupingType.PARTITION if is_partition else GroupingType.SEMANTIC

        return MarketGroup(
            group_id=event.event_id,
            name=event.title,
            market_ids=market_ids,
            group_type=group_type,
            constraints=constraints,
            is_partition=is_partition,
            metadata={
                "event_slug": event.slug,
                "neg_risk": event.neg_risk,
                "source": "polymarket_api",
            },
        )

    def build_relationship_graph(
        self,
        event: PolymarketEvent,
        market_ids: list[str] | None = None,
    ) -> RelationshipGraph:
        """Build a RelationshipGraph for the solver.

        Only includes constraint types the solver's LMO can handle.
        EXHAUSTIVE constraints are excluded (the LMO treats them as no-ops
        and partition semantics are captured via is_partition=True on the cluster).

        Args:
            event: The Polymarket event
            market_ids: Specific market IDs (defaults to all active)

        Returns:
            RelationshipGraph compatible with find_marginal_arbitrage()
        """
        if market_ids is None:
            market_ids = [m.condition_id for m in event.active_markets]

        constraints, is_partition = self.infer_from_event(event, market_ids)

        # Only include solver-safe relationship types in the graph
        relationships = []
        for c in constraints:
            if c.type not in _SOLVER_SAFE_TYPES:
                continue
            rel = MarketRelationship(
                type=c.type.value,
                from_market=c.from_market,
                to_market=c.to_market,
                confidence=c.confidence,
            )
            relationships.append(rel)

        cluster = MarketCluster(
            cluster_id=event.event_id,
            theme=event.title,
            market_ids=market_ids,
            relationships=relationships,
            is_partition=is_partition,
        )

        return RelationshipGraph(clusters=[cluster])

    def infer_with_llm_fallback(
        self,
        event: PolymarketEvent,
        market_metas: list[MarketMeta],
    ) -> tuple[MarketGroup, RelationshipGraph]:
        """Infer constraints, falling back to LLM for non-negRisk events.

        Args:
            event: The Polymarket event
            market_metas: MarketMeta objects for the markets

        Returns:
            (MarketGroup, RelationshipGraph) tuple
        """
        market_ids = [m.id for m in market_metas]

        if event.neg_risk and len(market_ids) > 1:
            logger.info(
                "Event '%s' has negRisk=True with %d markets — auto-inferring partition",
                event.title, len(market_ids),
            )
            group = self.build_market_group(event, market_ids)
            graph = self.build_relationship_graph(event, market_ids)
            return group, graph

        # Fall back to LLM grouping
        logger.info(
            "Event '%s' has negRisk=False — using LLM grouper for %d markets",
            event.title, len(market_ids),
        )
        return self._llm_fallback(event, market_metas)

    # ── Private Methods ────────────────────────────────────

    def _infer_neg_risk(self, market_ids: list[str]) -> list[Constraint]:
        """Generate MUTUALLY_EXCLUSIVE constraints for all pairs + EXHAUSTIVE.

        negRisk=True means exactly one outcome resolves YES — this is
        both mutually exclusive AND exhaustive (a partition).
        """
        constraints: list[Constraint] = []

        # All pairs are mutually exclusive
        for a, b in combinations(market_ids, 2):
            constraints.append(Constraint(
                type=ConstraintType.MUTUALLY_EXCLUSIVE,
                from_market=a,
                to_market=b,
                confidence=1.0,
                reasoning="negRisk=True: Polymarket CTF mutual exclusion",
            ))

        # The full set is exhaustive (at least one must be YES)
        for mid in market_ids:
            constraints.append(Constraint(
                type=ConstraintType.EXHAUSTIVE,
                from_market=mid,
                to_market=None,
                confidence=1.0,
                reasoning="negRisk=True: partition constraint (exactly one resolves YES)",
            ))

        logger.info(
            "negRisk partition: %d markets -> %d mutex pairs + %d exhaustive",
            len(market_ids),
            len(market_ids) * (len(market_ids) - 1) // 2,
            len(market_ids),
        )

        return constraints

    def _llm_fallback(
        self,
        event: PolymarketEvent,
        market_metas: list[MarketMeta],
    ) -> tuple[MarketGroup, RelationshipGraph]:
        """Use LLM grouper for non-negRisk events."""
        from src.grouping.llm_grouper import LLMSemanticGrouper

        grouper = LLMSemanticGrouper()
        groups = grouper.group(market_metas)

        if groups:
            group = groups[0]  # Use the first group
        else:
            # No LLM constraints — create a bare group
            group = MarketGroup(
                group_id=event.event_id,
                name=event.title,
                market_ids=[m.id for m in market_metas],
                group_type=GroupingType.SEMANTIC,
                constraints=[],
                is_partition=False,
            )

        # Build relationship graph — only include solver-safe types
        relationships = [
            MarketRelationship(
                type=c.type.value,
                from_market=c.from_market,
                to_market=c.to_market,
                confidence=c.confidence,
            )
            for c in group.constraints
            if c.type in _SOLVER_SAFE_TYPES
        ]

        cluster = MarketCluster(
            cluster_id=group.group_id,
            theme=group.name,
            market_ids=group.market_ids,
            relationships=relationships,
            is_partition=group.is_partition,
        )

        graph = RelationshipGraph(clusters=[cluster])
        return group, graph
