"""Dataset specification, live dataset container, and builder.

User-facing API for building solver-ready datasets from Polymarket events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.core.types import ConstraintType, MarketGroup
from src.optimizer.schema import (
    MarketCluster,
    MarketRelationship,
    MarginalArbitrageResult,
    OptimizationConfig,
    RelationshipGraph,
)

from .clob_client import ClobClient
from .gamma_client import GammaClient
from .mapping import to_market_meta
from .relationship_inference import RelationshipInferrer, _SOLVER_SAFE_TYPES
from .types import PolymarketEvent, PolymarketMarket

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    """Declarative specification of what markets to include.

    Users describe what they want; DatasetBuilder resolves it.
    """

    name: str
    event_slugs: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    market_ids: list[str] = field(default_factory=list)
    search_query: str | None = None
    min_volume: float | None = None
    min_liquidity: float | None = None
    refresh_prices_from_clob: bool = True


@dataclass
class LiveDataset:
    """Container for a resolved dataset with markets, constraints, and solver input.

    This is the main output of DatasetBuilder.build().
    """

    name: str
    events: list[PolymarketEvent]
    markets: dict[str, PolymarketMarket]
    market_group: MarketGroup
    relationships: RelationshipGraph

    def to_solver_input(
        self,
    ) -> tuple[dict[str, list[float]], RelationshipGraph, dict[str, list[str]]]:
        """Build input for find_marginal_arbitrage().

        Returns:
            (market_prices, relationships, market_outcomes)

            market_prices: {market_id: [p_yes, p_no, ...]}
            relationships: RelationshipGraph with constraints
            market_outcomes: {market_id: ["Yes", "No", ...]}
        """
        market_prices: dict[str, list[float]] = {}
        market_outcomes: dict[str, list[str]] = {}

        for mid, market in self.markets.items():
            if mid not in self.market_group.market_ids:
                continue

            outcomes = market.outcomes if market.outcomes else ["Yes", "No"]
            prices = list(market.outcome_prices)

            # Ensure we have prices for all outcomes
            while len(prices) < len(outcomes):
                prices.append(0.5)

            # For binary markets, ensure prices are complementary
            if len(outcomes) == 2 and len(prices) >= 2:
                # Keep YES price, derive NO from it
                prices[1] = 1.0 - prices[0]

            market_prices[mid] = prices
            market_outcomes[mid] = outcomes

        return market_prices, self.relationships, market_outcomes

    def run_solver(
        self,
        config: OptimizationConfig | None = None,
    ) -> MarginalArbitrageResult:
        """Convenience: build input and run the solver.

        Returns:
            MarginalArbitrageResult from find_marginal_arbitrage()
        """
        from src.optimizer.frank_wolfe import find_marginal_arbitrage

        prices, relationships, outcomes = self.to_solver_input()
        return find_marginal_arbitrage(
            market_prices=prices,
            relationships=relationships,
            market_outcomes=outcomes,
            config=config,
        )

    def summary(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [
            f"Dataset: {self.name}",
            f"Events: {len(self.events)}",
            f"Markets: {len(self.markets)} ({len(self.market_group.market_ids)} in group)",
            f"Partition: {self.market_group.is_partition}",
            f"Constraints: {len(self.market_group.constraints)}",
            "",
        ]

        # Market prices
        yes_sum = 0.0
        for mid in self.market_group.market_ids:
            market = self.markets.get(mid)
            if market:
                label = market.group_item_title or market.question[:50]
                price = market.yes_price
                yes_sum += price
                lines.append(f"  {label}: Yes={price:.3f}")

        lines.append("")
        lines.append(f"  Sum(Yes prices): {yes_sum:.4f} (should be ~1.0 for partitions)")

        return "\n".join(lines)


class DatasetBuilder:
    """Orchestrator: resolves a DatasetSpec into a LiveDataset.

    1. Resolve events from slugs/IDs/search
    2. Extract markets from events
    3. Add directly-specified market IDs
    4. Filter by volume/liquidity/active
    5. Refresh prices from CLOB midpoint
    6. Infer relationships from event structure
    7. Return LiveDataset
    """

    def __init__(
        self,
        gamma: GammaClient | None = None,
        clob: ClobClient | None = None,
    ):
        self._gamma = gamma or GammaClient()
        self._clob = clob or ClobClient()
        self._inferrer = RelationshipInferrer()

    def build(self, spec: DatasetSpec) -> LiveDataset:
        """Build a LiveDataset from a DatasetSpec."""
        logger.info("Building dataset: %s", spec.name)

        # Step 1: Resolve events
        events = self._resolve_events(spec)
        logger.info("Resolved %d events", len(events))

        # Step 2: Extract markets from events
        markets: dict[str, PolymarketMarket] = {}
        for event in events:
            for market in event.markets:
                markets[market.condition_id] = market

        # Step 3: Add directly-specified market IDs
        for market_id in spec.market_ids:
            if market_id not in markets:
                market = self._gamma.get_market_by_id(market_id)
                if market:
                    markets[market.condition_id] = market
                else:
                    logger.warning("Market %s not found", market_id)

        # Step 4: Filter
        markets = self._filter_markets(markets, spec)
        logger.info("After filtering: %d markets", len(markets))

        # Step 5: Refresh prices from CLOB
        if spec.refresh_prices_from_clob:
            markets = self._refresh_prices(markets)

        # Step 6: Infer relationships
        market_group, relationships = self._infer_relationships(events, markets)

        # Step 7: Build and return
        dataset = LiveDataset(
            name=spec.name,
            events=events,
            markets=markets,
            market_group=market_group,
            relationships=relationships,
        )

        logger.info("Dataset built: %s", dataset.summary().split(chr(10))[0])
        return dataset

    # ── Private Methods ────────────────────────────────────

    def _resolve_events(self, spec: DatasetSpec) -> list[PolymarketEvent]:
        """Resolve events from slugs, IDs, and/or search."""
        events: list[PolymarketEvent] = []
        seen_ids: set[str] = set()

        # By slug (primary method)
        for slug in spec.event_slugs:
            event = self._gamma.get_event_by_slug(slug)
            if event and event.event_id not in seen_ids:
                events.append(event)
                seen_ids.add(event.event_id)
            else:
                logger.warning("Event slug '%s' not found", slug)

        # By ID
        for eid in spec.event_ids:
            if eid not in seen_ids:
                event = self._gamma.get_event_by_id(eid)
                if event:
                    events.append(event)
                    seen_ids.add(event.event_id)
                else:
                    logger.warning("Event ID '%s' not found", eid)

        # By search
        if spec.search_query:
            search_results = self._gamma.search_events(query=spec.search_query)
            for event in search_results:
                if event.event_id not in seen_ids:
                    events.append(event)
                    seen_ids.add(event.event_id)

        return events

    def _filter_markets(
        self,
        markets: dict[str, PolymarketMarket],
        spec: DatasetSpec,
    ) -> dict[str, PolymarketMarket]:
        """Filter markets by volume, liquidity, and active status."""
        filtered: dict[str, PolymarketMarket] = {}
        for mid, market in markets.items():
            # Must be active
            if not market.is_active:
                continue
            # Volume filter
            if spec.min_volume is not None and market.volume < spec.min_volume:
                continue
            # Liquidity filter
            if spec.min_liquidity is not None and market.liquidity < spec.min_liquidity:
                continue
            filtered[mid] = market
        return filtered

    def _refresh_prices(
        self,
        markets: dict[str, PolymarketMarket],
    ) -> dict[str, PolymarketMarket]:
        """Refresh market prices from CLOB midpoint (more accurate than Gamma snapshot)."""
        refreshed = 0
        for mid, market in markets.items():
            if not market.yes_token_id:
                continue
            midpoint = self._clob.get_midpoint(market.yes_token_id)
            if midpoint is not None:
                market.outcome_prices = [midpoint, 1.0 - midpoint]
                refreshed += 1

        logger.info("Refreshed %d/%d market prices from CLOB", refreshed, len(markets))
        return markets

    def _infer_relationships(
        self,
        events: list[PolymarketEvent],
        markets: dict[str, PolymarketMarket],
    ) -> tuple[MarketGroup, RelationshipGraph]:
        """Infer constraints from event structure."""
        market_ids = list(markets.keys())

        if len(events) == 1:
            event = events[0]
            # Filter market_ids to those in this event
            event_market_ids = {m.condition_id for m in event.markets}
            relevant_ids = [mid for mid in market_ids if mid in event_market_ids]

            if event.neg_risk and len(relevant_ids) > 1:
                # Fast path: auto-infer from negRisk
                group = self._inferrer.build_market_group(event, relevant_ids)
                graph = self._inferrer.build_relationship_graph(event, relevant_ids)
                return group, graph

            # Non-negRisk: try LLM fallback
            market_metas = [to_market_meta(markets[mid]) for mid in relevant_ids]
            return self._inferrer.infer_with_llm_fallback(event, market_metas)

        # Multiple events: merge all market IDs, build combined graph
        all_clusters: list[MarketCluster] = []
        all_constraints = []
        all_market_ids: list[str] = []
        is_partition = True

        for event in events:
            event_market_ids = [
                m.condition_id for m in event.markets
                if m.condition_id in markets
            ]
            if not event_market_ids:
                continue

            constraints, partition = self._inferrer.infer_from_event(event, event_market_ids)
            is_partition = is_partition and partition
            all_constraints.extend(constraints)
            all_market_ids.extend(event_market_ids)

            # Only include solver-safe relationship types in the graph
            relationships = [
                MarketRelationship(
                    type=c.type.value,
                    from_market=c.from_market,
                    to_market=c.to_market,
                    confidence=c.confidence,
                )
                for c in constraints
                if c.type in _SOLVER_SAFE_TYPES
            ]

            all_clusters.append(MarketCluster(
                cluster_id=event.event_id,
                theme=event.title,
                market_ids=event_market_ids,
                relationships=relationships,
                is_partition=partition,
            ))

        group = MarketGroup(
            group_id="_".join(e.event_id for e in events),
            name="Combined: " + ", ".join(e.title for e in events),
            market_ids=all_market_ids,
            group_type="partition" if is_partition else "semantic",
            constraints=all_constraints,
            is_partition=is_partition,
        )

        graph = RelationshipGraph(clusters=all_clusters)
        return group, graph
