"""Integration tests for the modular pipeline.

Verifies that existing strategies work through the pipeline,
strategy discovery works, results match the old path, the import
graph is layered, and new strategies can be registered trivially.
"""

import importlib
import pkgutil
import sys
import pytest
from datetime import datetime, timedelta
from typing import Any, Iterator
from unittest.mock import MagicMock

from src.core.types import (
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


def _make_market(mid: str, question: str = "Test?") -> MarketMeta:
    return MarketMeta(
        id=mid, question=question, slug=mid, outcomes=["Yes", "No"],
        volume=1000.0,
    )


def _make_partition_snapshots(
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
            group_id="partition_test",
            timestamp=ts,
            snapshots=market_snaps,
        ))

    return snapshots


# ── Test 1: partition_arb and combinatorial_arb work through the pipeline ──

class TestPipelineWithPartitionArb:
    """Test pipeline with partition arbitrage strategy."""

    def test_partition_arb_detects_violations(self):
        """Pipeline should detect partition violations (sum != 1)."""
        import src.strategies.partition_arb  # noqa: F401
        from src.grouping.manual_grouper import ManualGrouper

        markets = [_make_market("A"), _make_market("B"), _make_market("C")]

        # Create snapshots where partition sum != 1
        snapshots = _make_partition_snapshots(
            ["A", "B", "C"],
            [
                {"A": 0.30, "B": 0.30, "C": 0.30},  # Sum=0.90 (underpriced!)
                {"A": 0.33, "B": 0.33, "C": 0.33},  # Sum=0.99 (close to 1)
                {"A": 0.40, "B": 0.35, "C": 0.35},  # Sum=1.10 (overpriced!)
            ],
        )

        data_source = MockMarketSource(markets, snapshots)

        pipeline = Pipeline(data_source=data_source)
        pipeline.add_grouper(ManualGrouper([{
            "group_id": "test_partition",
            "name": "Test Partition",
            "market_ids": ["A", "B", "C"],
            "is_partition": True,
        }]))
        pipeline.add_strategy("partition_arb")

        results = pipeline.run()

        assert "partition_arb" in results
        result = results["partition_arb"]
        assert isinstance(result, BacktestResult)
        assert result.total_opportunities >= 1

    def test_combinatorial_arb_runs_through_pipeline(self):
        """combinatorial_arb should run through the pipeline without error."""
        import src.strategies.combinatorial_arb  # noqa: F401
        from src.grouping.manual_grouper import ManualGrouper

        markets = [_make_market("A"), _make_market("B"), _make_market("C")]

        snapshots = _make_partition_snapshots(
            ["A", "B", "C"],
            [
                {"A": 0.30, "B": 0.30, "C": 0.30},
                {"A": 0.40, "B": 0.35, "C": 0.35},
            ],
        )

        data_source = MockMarketSource(markets, snapshots)

        pipeline = Pipeline(data_source=data_source)
        # combinatorial_arb needs SEMANTIC grouping; ManualGrouper can be
        # configured for that.  Use is_partition=False to avoid generating
        # MUTUALLY_EXCLUSIVE constraints (those exercise a separate optimizer
        # code path not under test here).
        pipeline.add_grouper(ManualGrouper(
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
        import src.strategies.partition_arb  # noqa: F401
        data_source = MockMarketSource([], [])
        pipeline = Pipeline(data_source=data_source)
        pipeline.add_strategy("partition_arb")
        results = pipeline.run()
        assert results == {}


# ── Test 2: Strategy registry discovers registered strategies ──

class TestStrategyRegistryDiscovery:
    """Strategy registry should discover all registered strategies."""

    def test_strategy_registry_discovery(self):
        """All registered strategies should be discoverable."""
        import src.strategies.partition_arb  # noqa: F401
        import src.strategies.combinatorial_arb  # noqa: F401
        from src.strategies.registry import list_strategies

        strategies = list_strategies()
        assert "partition_arb" in strategies
        assert "combinatorial_arb" in strategies

    def test_registry_get_strategy(self):
        """Registry should instantiate strategies by name."""
        import src.strategies.partition_arb  # noqa: F401
        import src.strategies.combinatorial_arb  # noqa: F401
        from src.strategies.registry import get_strategy

        for name in ("partition_arb", "combinatorial_arb"):
            strategy = get_strategy(name)
            assert strategy.name == name

    def test_registry_unknown_strategy_raises(self):
        """Requesting an unregistered strategy should raise KeyError."""
        from src.strategies.registry import get_strategy
        with pytest.raises(KeyError):
            get_strategy("nonexistent_strategy")


# ── Test 3: Pipeline results match old WalkForwardSimulator path ──

class TestPipelineEquivalence:
    """Pipeline should produce results identical to direct strategy execution."""

    def test_partition_arb_pipeline_matches_direct(self):
        """Results from the pipeline should match calling strategy.detect() directly."""
        import src.strategies.partition_arb  # noqa: F401
        from src.strategies.registry import get_strategy
        from src.grouping.manual_grouper import ManualGrouper

        markets = [_make_market("A"), _make_market("B"), _make_market("C")]

        snapshots = _make_partition_snapshots(
            ["A", "B", "C"],
            [
                {"A": 0.30, "B": 0.30, "C": 0.30},  # Sum=0.90
                {"A": 0.40, "B": 0.35, "C": 0.35},  # Sum=1.10
            ],
        )

        data_source = MockMarketSource(markets, snapshots)

        # Path A: through the pipeline
        pipeline = Pipeline(data_source=data_source)
        group_def = {
            "group_id": "equiv_test",
            "name": "Equivalence Test",
            "market_ids": ["A", "B", "C"],
            "is_partition": True,
        }
        pipeline.add_grouper(ManualGrouper([group_def]))
        pipeline.add_strategy("partition_arb")
        pipeline_results = pipeline.run()

        # Path B: direct strategy execution (simulating WalkForwardSimulator)
        strategy = get_strategy("partition_arb")
        grouper = ManualGrouper([group_def])
        groups = grouper.group(markets, data_source)
        assert len(groups) == 1
        group = groups[0]

        direct_opps = []
        for gs in data_source.iter_snapshots(["A", "B", "C"]):
            opps = strategy.detect(group, gs)
            direct_opps.extend(opps)

        # Both paths should find the same number of opportunities
        pipeline_result = pipeline_results["partition_arb"]
        assert pipeline_result.total_opportunities == len(direct_opps)

        # Both paths should compute the same expected profits
        pipeline_profits = sorted(o.expected_profit for o in pipeline_result.opportunities)
        direct_profits = sorted(o.expected_profit for o in direct_opps)
        assert len(pipeline_profits) == len(direct_profits)
        for pp, dp in zip(pipeline_profits, direct_profits):
            assert abs(pp - dp) < 1e-10


# ── Test 4: Import graph is strictly layered ──

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
        ("src.strategies.partition_arb", "src.pipeline"),
        ("src.strategies.partition_arb", "src.backtest"),
        ("src.grouping.manual_grouper", "src.pipeline"),
        ("src.grouping.manual_grouper", "src.backtest"),
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


# ── Test 5: New strategy can be registered with minimal code ──

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
                return GroupingType.PARTITION

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
        from src.grouping.manual_grouper import ManualGrouper

        markets = [_make_market("X"), _make_market("Y")]
        snapshots = _make_partition_snapshots(
            ["X", "Y"],
            [{"X": 0.60, "Y": 0.50}],
        )
        data_source = MockMarketSource(markets, snapshots)

        pipeline = Pipeline(data_source=data_source)
        pipeline.add_grouper(ManualGrouper([{
            "group_id": "dummy_group",
            "name": "Dummy",
            "market_ids": ["X", "Y"],
            "is_partition": True,
        }]))
        pipeline.add_strategy("dummy_test_strategy")
        results = pipeline.run()

        assert "dummy_test_strategy" in results
        assert isinstance(results["dummy_test_strategy"], BacktestResult)
        assert results["dummy_test_strategy"].total_opportunities >= 1

        # Cleanup: remove from registry so it doesn't affect other tests
        if not had_dummy:
            del _STRATEGY_REGISTRY["dummy_test_strategy"]
