"""Pipeline orchestrator for the modular architecture.

This replaces the monolithic runner scripts (run_backtest.py, etc.)
and provides a clean top-level entry point for running backtests
with multiple strategies.

Usage:
    pipeline = Pipeline(data_source=ParquetMarketSource(...))
    pipeline.add_strategy("partition_arb", StrategyConfig(...))
    pipeline.add_strategy("rebalancing_arb", StrategyConfig(...))
    pipeline.add_grouper(LLMSemanticGrouper())
    pipeline.add_grouper(ManualGrouper(tournaments))

    results = pipeline.run(
        market_ids=[...],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
    )
"""

import logging
from datetime import datetime
from typing import Any

from src.core.types import (
    GroupingType,
    MarketGroup,
    MarketMeta,
    StrategyConfig,
)
from src.core.protocols import (
    ArbitrageStrategy,
    MarketDataSource,
    MarketGrouper,
)
from src.strategies.registry import get_strategy, list_strategies

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the modular pipeline.

    Coordinates strategies, groupers, data sources, and backtest evaluators
    into a single execution flow.
    """

    def __init__(self, data_source: MarketDataSource):
        self.data_source = data_source
        self._strategies: list[ArbitrageStrategy] = []
        self._groupers: dict[GroupingType, MarketGrouper] = {}
        self._report_plugins: dict[str, Any] = {}

    def add_strategy(
        self,
        name: str,
        config: StrategyConfig | None = None,
    ) -> None:
        """Register a strategy by name with optional config."""
        strategy = get_strategy(name, config)
        self._strategies.append(strategy)

        # Ensure we have a grouper for this strategy's needs
        needed = strategy.required_grouping
        if needed not in self._groupers:
            logger.warning(
                "Strategy %s needs grouping type %s but no grouper registered",
                name, needed.value,
            )

    def add_strategy_instance(self, strategy: ArbitrageStrategy) -> None:
        """Register a pre-instantiated strategy."""
        self._strategies.append(strategy)

    def add_grouper(self, grouper: MarketGrouper) -> None:
        """Register a market grouper."""
        self._groupers[grouper.grouping_type] = grouper

    def run(
        self,
        market_ids: list[str] | None = None,
        category: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Execute the full pipeline.

        1. Load markets
        2. Group markets per strategy requirements
        3. Run each strategy against its groups
        4. Collect results

        Returns dict keyed by strategy name.
        """
        # Step 1: Load markets
        markets = self.data_source.get_markets(
            market_ids=market_ids,
            category=category,
        )
        logger.info("Loaded %d markets", len(markets))

        if not markets:
            logger.warning("No markets found, returning empty results")
            return {}

        # Step 2: Group markets by each required grouping type
        needed_types = {s.required_grouping for s in self._strategies}
        groups_by_type: dict[GroupingType, list[MarketGroup]] = {}

        for gtype in needed_types:
            grouper = self._groupers.get(gtype)
            if grouper is None:
                logger.warning("No grouper for type %s, skipping", gtype.value)
                continue
            groups_by_type[gtype] = grouper.group(markets, self.data_source)
            logger.info(
                "Grouped %d markets into %d groups (type=%s)",
                len(markets), len(groups_by_type[gtype]), gtype.value,
            )

        # Step 3: Run strategies
        all_results = {}
        for strategy in self._strategies:
            groups = groups_by_type.get(strategy.required_grouping, [])
            if not groups:
                logger.warning(
                    "No groups for strategy %s, skipping", strategy.name
                )
                continue

            logger.info(
                "Running strategy %s over %d groups",
                strategy.name, len(groups),
            )

            # Delegate to the appropriate backtest evaluator
            from src.backtest.engine import get_evaluator
            evaluator = get_evaluator(strategy)
            result = evaluator.evaluate(
                strategy=strategy,
                groups=groups,
                data_source=self.data_source,
                config=strategy.config if hasattr(strategy, "config") else None,
            )
            all_results[strategy.name] = result

        return all_results

    @property
    def registered_strategies(self) -> list[str]:
        """List names of currently registered strategies."""
        return [s.name for s in self._strategies]

    @property
    def registered_groupers(self) -> list[str]:
        """List grouping types with registered groupers."""
        return [gt.value for gt in self._groupers]
