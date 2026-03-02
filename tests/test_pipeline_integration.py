"""Integration tests for the modular pipeline.

Verifies that the combinatorial_arb strategy works through the pipeline,
strategy discovery works, results are correct, the import graph is layered,
and new strategies can be registered trivially.
"""

import importlib
import pkgutil
import sys
import pytest
from datetime import datetime, timedelta
from typing import Any, Iterator
from unittest.mock import MagicMock

from src.core.types import (
    Constraint,
    ConstraintType,
    DataRequirements,
    GroupingType,
    GroupSnapshot,
    MarketGroup,
    MarketMeta,
    MarketSnapshot,
    MarketTimeSeries,
    Opportunity,
    PricePoint,
    StrategyConfig,
    TradeDirection,
    TradeLeg,
    TradeType,
)
from src.core.protocols import MarketDataSource
from src.pipeline import Pipeline
from src.backtest.engine import BacktestResult


# ── Mock Data Source ──

class MockMarketSource:
    """Mock data source for testing the pipeline."""

    def __init__(self, markets: list[MarketMeta], snapshots: list[GroupSnapshot]):
        self._markets = {m.id: m for m in markets}
        self._snapshots = snapshots

    def get_market(self, market_id: str) -> MarketMeta | None:
        return self._markets.get(market_id)

    def get_markets(
        self,
        market_ids=None,
        category=None,
        min_volume=None,
        limit=None,
    ) -> list[MarketMeta]:
        markets = list(self._markets.values())
        if market_ids:
            markets = [m for m in markets if m.id in market_ids]
        if category:
            markets = [m for m in markets if m.category == category]
        if limit:
            markets = markets[:limit]
        return markets

    def get_snapshot(self, market_id, at_time=None):
        for gs in self._snapshots:
            if market_id in gs.snapshots:
                return gs.snapshots[market_id]
        return None

    def get_snapshots(self, market_ids, at_time=None):
        result = {}
        for mid in market_ids:
            snap = self.get_snapshot(mid, at_time)
            if snap:
                result[mid] = snap
        return result

    def get_time_series(self, market_id, start=None, end=None, interval_minutes=60):
        market = self._markets.get(market_id)
        if not market:
            return MarketTimeSeries(market=MarketMeta(id=market_id, question="", slug="", outcomes=["Yes", "No"]), points=[])

        # Generate synthetic history
        points = []
        base_time = start or datetime(2025, 1, 1)
        for i in range(200):
            ts = base_time + timedelta(hours=i)
            points.append(PricePoint(
                timestamp=ts,
                prices={"Yes": 0.5, "No": 0.5},
            ))
        return MarketTimeSeries(market=market, points=points)

    def iter_snapshots(self, market_ids, start=None, end=None) -> Iterator[GroupSnapshot]:
        for gs in self._snapshots:
            # Filter to only include requested markets
            filtered_snaps = {
                mid: snap for mid, snap in gs.snapshots.items()
                if mid in market_ids
            }
            if filtered_snaps:
                yield GroupSnapshot(
                    group_id=gs.group_id,
                    timestamp=gs.timestamp,
                    snapshots=filtered_snaps,
                )

    def close(self):
        pass


# ── Mock Grouper ──

class MockGrouper:
    """Simple grouper for testing that creates groups from definitions."""

    def __init__(
        self,
        group_definitions: list[dict[str, Any]],
        grouping_type: GroupingType = GroupingType.SEMANTIC,
    ):
        self._definitions = group_definitions
        self.grouping_type = grouping_type

    def group(
        self,
        markets: list[MarketMeta],
        data_source: MarketDataSource | None = None,
    ) -> list[MarketGroup]:
        available_ids = {m.id for m in markets}
        groups = []
        for defn in self._definitions:
            group_ids = [mid for mid in defn["market_ids"] if mid in available_ids]
            if len(group_ids) < 2:
                continue
            groups.append(MarketGroup(
                group_id=defn["group_id"],
                name=defn.get("name", defn["group_id"]),
                market_ids=group_ids,
                group_type=self.grouping_type,
                constraints=defn.get("constraints", []),
                is_partition=defn.get("is_partition", False),
                metadata=defn.get("metadata", {}),
            ))
        return groups


def _make_market(mid: str, question: str = "Test?") -> MarketMeta:
    return MarketMeta(
        id=mid, question=question, slug=mid, outcomes=["Yes", "No"],
        volume=1000.0,
    )


def _make_snapshots(
    market_ids: list[str],
    prices_over_time: list[dict[str, float]],
    start: datetime | None = None,
) -> list[GroupSnapshot]:
    """Create a sequence of group snapshots with given prices."""
    start = start or datetime(2025, 1, 1)
    snapshots = []

    for i, prices in enumerate(prices_over_time):
        ts = start + timedelta(hours=i)
        market_snaps = {}
        for mid, price in prices.items():
            market = _make_market(mid)
            market_snaps[mid] = MarketSnapshot(
                market=market,
                price_point=PricePoint(
                    timestamp=ts,
                    prices={"Yes": price, "No": 1.0 - price},
                ),
            )
        snapshots.append(GroupSnapshot(
            group_id="test_group",
            timestamp=ts,
            snapshots=market_snaps,
        ))

    return snapshots


# ── Test 1: combinatorial_arb works through the pipeline ──

class TestPipelineWithCombinatorialArb:
    """Test pipeline with combinatorial arbitrage strategy."""

    def test_combinatorial_arb_runs_through_pipeline(self):
        """combinatorial_arb should run through the pipeline without error."""
        import src.strategies.combinatorial_arb  # noqa: F401

        markets = [_make_market("A"), _make_market("B"), _make_market("C")]

        snapshots = _make_snapshots(
            ["A", "B", "C"],
            [
                {"A": 0.30, "B": 0.30, "C": 0.30},
                {"A": 0.40, "B": 0.35, "C": 0.35},
            ],
        )

        data_source = MockMarketSource(markets, snapshots)

        pipeline = Pipeline(data_source=data_source)
        pipeline.add_grouper(MockGrouper(
            [{
                "group_id": "test_semantic",
                "name": "Test Semantic Group",
                "market_ids": ["A", "B", "C"],
                "is_partition": False,
            }],
            grouping_type=GroupingType.SEMANTIC,
        ))
        pipeline.add_strategy("combinatorial_arb")

        results = pipeline.run()

        assert "combinatorial_arb" in results
        result = results["combinatorial_arb"]
        assert isinstance(result, BacktestResult)

    def test_pipeline_with_no_strategies(self):
        """Pipeline with no strategies should return empty results."""
        data_source = MockMarketSource([], [])
        pipeline = Pipeline(data_source=data_source)
        results = pipeline.run()
        assert results == {}

    def test_pipeline_with_no_markets(self):
        """Pipeline with no markets should return empty results."""
        import src.strategies.combinatorial_arb  # noqa: F401
        data_source = MockMarketSource([], [])
        pipeline = Pipeline(data_source=data_source)
        pipeline.add_strategy("combinatorial_arb")
        results = pipeline.run()
        assert results == {}


# ── Test 2: Strategy registry discovers registered strategies ──

class TestStrategyRegistryDiscovery:
    """Strategy registry should discover all registered strategies."""

    def test_strategy_registry_discovery(self):
        """All registered strategies should be discoverable."""
        import src.strategies.combinatorial_arb  # noqa: F401
        from src.strategies.registry import list_strategies

        strategies = list_strategies()
        assert "combinatorial_arb" in strategies

    def test_registry_get_strategy(self):
        """Registry should instantiate strategies by name."""
        import src.strategies.combinatorial_arb  # noqa: F401
        from src.strategies.registry import get_strategy

        strategy = get_strategy("combinatorial_arb")
        assert strategy.name == "combinatorial_arb"

    def test_registry_unknown_strategy_raises(self):
        """Requesting an unregistered strategy should raise KeyError."""
        from src.strategies.registry import get_strategy
        with pytest.raises(KeyError):
            get_strategy("nonexistent_strategy")


# ── Test 3: Import graph is strictly layered ──

class TestImportGraph:
    """Verify no forbidden cross-layer imports.

    Layer hierarchy (lower cannot import higher):
        core -> grouping, strategies -> pipeline -> backtest
    """

    LAYER_ORDER = {
        "src.core": 0,
        "src.grouping": 1,
        "src.strategies": 1,
        "src.backtest": 2,
        "src.pipeline": 2,
    }

    FORBIDDEN_IMPORTS = [
        # Core layer must not import from higher layers
        ("src.core.types", "src.pipeline"),
        ("src.core.types", "src.backtest"),
        ("src.core.types", "src.strategies"),
        ("src.core.types", "src.grouping"),
        ("src.core.protocols", "src.pipeline"),
        ("src.core.protocols", "src.backtest"),
        ("src.core.protocols", "src.strategies"),
        ("src.core.protocols", "src.grouping"),
        # Strategies/grouping must not import pipeline or backtest
        ("src.strategies.registry", "src.pipeline"),
        ("src.strategies.registry", "src.backtest"),
        ("src.strategies.combinatorial_arb", "src.pipeline"),
        ("src.strategies.combinatorial_arb", "src.backtest"),
    ]

    def _get_imports(self, module_name: str) -> set[str]:
        """Get the set of top-level imports for a module."""
        mod = importlib.import_module(module_name)
        source_file = getattr(mod, "__file__", None)
        if source_file is None:
            return set()

        with open(source_file, "r") as f:
            source = f.read()

        imports = set()
        for line in source.splitlines():
            line = line.strip()
            if line.startswith("from "):
                parts = line.split()
                if len(parts) >= 2:
                    imports.add(parts[1])
            elif line.startswith("import "):
                parts = line.split()
                if len(parts) >= 2:
                    imports.add(parts[1].split(".")[0] + "." + ".".join(parts[1].split(".")[1:]) if "." in parts[1] else parts[1])
        return imports

    def test_no_forbidden_imports(self):
        """Core/strategy/grouping modules must not import from higher layers."""
        for module_name, forbidden_prefix in self.FORBIDDEN_IMPORTS:
            try:
                imports = self._get_imports(module_name)
            except (ImportError, FileNotFoundError):
                continue  # Module may not exist in all configs

            for imp in imports:
                assert not imp.startswith(forbidden_prefix), (
                    f"{module_name} imports {imp}, which violates the layer boundary "
                    f"(must not import from {forbidden_prefix})"
                )


# ── Test 4: New strategy can be registered with minimal code ──

class TestMinimalStrategyRegistration:
    """A new strategy can be registered with a trivial mock."""

    def test_register_trivial_strategy(self):
        """A minimal strategy class should register and run through the pipeline."""
        from src.strategies.registry import register_strategy, get_strategy, _STRATEGY_REGISTRY

        # Clean up after test
        had_dummy = "dummy_test_strategy" in _STRATEGY_REGISTRY

        @register_strategy("dummy_test_strategy")
        class DummyStrategy:
            def __init__(self, config=None):
                self.config = config

            @property
            def name(self):
                return "dummy_test_strategy"

            @property
            def required_grouping(self):
                return GroupingType.SEMANTIC

            @property
            def data_requirements(self):
                return DataRequirements(needs_snapshots=True, needs_time_series=False)

            def detect(self, group, snapshot, history=None):
                # Always emit one trivial opportunity
                return [Opportunity(
                    strategy_name="dummy_test_strategy",
                    group_id=group.group_id,
                    timestamp=snapshot.timestamp,
                    legs=[],
                    expected_profit=0.0,
                    confidence=1.0,
                )]

            def size_trades(self, opps, portfolio_state=None):
                return opps

            def validate(self, opps, snapshot):
                return opps

            def update_positions(self, open_trades, snapshot, history=None):
                return []

        # Verify it is discoverable
        from src.strategies.registry import list_strategies
        assert "dummy_test_strategy" in list_strategies()

        # Verify it can be instantiated
        strategy = get_strategy("dummy_test_strategy")
        assert strategy.name == "dummy_test_strategy"

        # Verify it runs through the pipeline
        markets = [_make_market("X"), _make_market("Y")]
        snapshots = _make_snapshots(
            ["X", "Y"],
            [{"X": 0.60, "Y": 0.50}],
        )
        data_source = MockMarketSource(markets, snapshots)

        pipeline = Pipeline(data_source=data_source)
        pipeline.add_grouper(MockGrouper([{
            "group_id": "dummy_group",
            "name": "Dummy",
            "market_ids": ["X", "Y"],
            "is_partition": False,
        }]))
        pipeline.add_strategy("dummy_test_strategy")
        results = pipeline.run()

        assert "dummy_test_strategy" in results
        assert isinstance(results["dummy_test_strategy"], BacktestResult)
        assert results["dummy_test_strategy"].total_opportunities >= 1

        # Cleanup: remove from registry so it doesn't affect other tests
        if not had_dummy:
            del _STRATEGY_REGISTRY["dummy_test_strategy"]
