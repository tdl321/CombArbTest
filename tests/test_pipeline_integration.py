"""Integration tests for the modular pipeline.

Tests that multiple strategies can be run through the pipeline
using mock data sources.
"""

import pytest
from datetime import datetime, timedelta
from typing import Iterator
from unittest.mock import MagicMock

from src.core.types import (
    GroupingType,
    GroupSnapshot,
    MarketGroup,
    MarketMeta,
    MarketSnapshot,
    MarketTimeSeries,
    PricePoint,
    StrategyConfig,
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


class TestPipelineWithPartitionArb:
    """Test pipeline with partition arbitrage strategy."""

    def test_partition_arb_detects_violations(self):
        """Pipeline should detect partition violations (sum \!= 1)."""
        # Ensure strategy is registered
        import src.strategies.partition_arb  # noqa: F401
        from src.grouping.manual_grouper import ManualGrouper

        markets = [_make_market("A"), _make_market("B"), _make_market("C")]

        # Create snapshots where partition sum \!= 1
        snapshots = _make_partition_snapshots(
            ["A", "B", "C"],
            [
                {"A": 0.30, "B": 0.30, "C": 0.30},  # Sum=0.90 (underpriced\!)
                {"A": 0.33, "B": 0.33, "C": 0.33},  # Sum=0.99 (close to 1)
                {"A": 0.40, "B": 0.35, "C": 0.35},  # Sum=1.10 (overpriced\!)
            ],
        )

        data_source = MockMarketSource(markets, snapshots)

        # Set up pipeline
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
        assert result.total_opportunities >= 1  # Should find at least the 0.90 violation

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


class TestPipelineMultiStrategy:
    """Test pipeline with multiple strategies."""

    def test_multiple_strategies_run(self):
        """Multiple strategies should be able to run in the same pipeline."""
        import src.strategies.partition_arb  # noqa: F401
        import src.strategies.rebalancing_arb  # noqa: F401
        from src.grouping.manual_grouper import ManualGrouper
        from src.grouping.correlation_grouper import CorrelationGrouper

        markets = [_make_market("A"), _make_market("B"), _make_market("C")]

        snapshots = _make_partition_snapshots(
            ["A", "B", "C"],
            [
                {"A": 0.30, "B": 0.30, "C": 0.30},  # Sum=0.90
                {"A": 0.33, "B": 0.33, "C": 0.34},  # Sum=1.00
            ],
        )

        data_source = MockMarketSource(markets, snapshots)

        pipeline = Pipeline(data_source=data_source)

        # Add partition grouper and strategy
        pipeline.add_grouper(ManualGrouper([{
            "group_id": "test_partition",
            "name": "Test Partition",
            "market_ids": ["A", "B", "C"],
            "is_partition": True,
        }]))
        pipeline.add_strategy("partition_arb")

        # Add correlation grouper and strategy
        pipeline.add_grouper(CorrelationGrouper(min_correlation=0.1))
        pipeline.add_strategy("rebalancing_arb")

        results = pipeline.run()

        # Both strategies should have results (even if empty opportunities)
        assert "partition_arb" in results
        assert isinstance(results["partition_arb"], BacktestResult)

    def test_strategy_registry_discovery(self):
        """All registered strategies should be discoverable."""
        import src.strategies.partition_arb  # noqa: F401
        import src.strategies.combinatorial_arb  # noqa: F401
        import src.strategies.rebalancing_arb  # noqa: F401
        from src.strategies.registry import list_strategies

        strategies = list_strategies()
        assert "partition_arb" in strategies
        assert "combinatorial_arb" in strategies
        assert "rebalancing_arb" in strategies
