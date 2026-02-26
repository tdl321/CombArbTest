"""Unit tests for the correlation-based market grouper."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.core.types import (
    GroupingType,
    MarketGroup,
    MarketMeta,
    MarketTimeSeries,
    PricePoint,
)
from src.grouping.correlation_grouper import CorrelationGrouper


def _make_market(mid: str) -> MarketMeta:
    return MarketMeta(
        id=mid, question=f"Market {mid}?", slug=mid, outcomes=["Yes", "No"]
    )


def _make_time_series(
    market: MarketMeta,
    prices: list[float],
    start: datetime | None = None,
) -> MarketTimeSeries:
    start = start or datetime(2024, 12, 1)
    points = [
        PricePoint(
            timestamp=start + timedelta(hours=i),
            prices={"Yes": p, "No": 1.0 - p},
        )
        for i, p in enumerate(prices)
    ]
    return MarketTimeSeries(market=market, points=points)


class TestCorrelationGrouper:
    """Tests for CorrelationGrouper."""

    def test_init_defaults(self):
        grouper = CorrelationGrouper()
        assert grouper.grouping_type == GroupingType.CORRELATION
        assert grouper.min_correlation == 0.5
        assert grouper.min_group_size == 2

    def test_group_no_data_source(self):
        grouper = CorrelationGrouper()
        markets = [_make_market("A"), _make_market("B")]
        result = grouper.group(markets, data_source=None)
        assert result == []

    def test_group_correlated_markets(self):
        """Markets with similar price movements should be grouped together."""
        grouper = CorrelationGrouper(min_correlation=0.5)

        markets = [_make_market("A"), _make_market("B"), _make_market("C")]

        # A and B are highly correlated (move together)
        # C is uncorrelated (random/opposite)
        import numpy as np
        np.random.seed(42)

        base = np.linspace(0.3, 0.7, 50)
        noise_a = np.random.normal(0, 0.01, 50)
        noise_b = np.random.normal(0, 0.01, 50)
        prices_a = list(np.clip(base + noise_a, 0.01, 0.99))
        prices_b = list(np.clip(base + noise_b, 0.01, 0.99))
        prices_c = list(np.clip(np.random.uniform(0.3, 0.7, 50), 0.01, 0.99))

        ts_a = _make_time_series(_make_market("A"), prices_a)
        ts_b = _make_time_series(_make_market("B"), prices_b)
        ts_c = _make_time_series(_make_market("C"), prices_c)

        # Mock data source
        data_source = MagicMock()
        def get_ts(market_id, start=None, end=None, interval_minutes=60):
            mapping = {"A": ts_a, "B": ts_b, "C": ts_c}
            return mapping.get(market_id)
        data_source.get_time_series = get_ts

        groups = grouper.group(markets, data_source)

        # A and B should be in the same group
        assert len(groups) >= 1

        # Find the group containing A
        group_with_a = next((g for g in groups if "A" in g.market_ids), None)
        if group_with_a:
            assert "B" in group_with_a.market_ids  # A and B correlated

    def test_group_insufficient_data(self):
        """Markets with too little data should be excluded."""
        grouper = CorrelationGrouper()
        markets = [_make_market("A"), _make_market("B")]

        # Only 5 data points (less than 24 threshold)
        ts_short = _make_time_series(_make_market("A"), [0.5] * 5)

        data_source = MagicMock()
        data_source.get_time_series.return_value = ts_short

        groups = grouper.group(markets, data_source)
        assert groups == []

    def test_group_single_market(self):
        """Single market cannot form a group."""
        grouper = CorrelationGrouper()
        markets = [_make_market("A")]
        data_source = MagicMock()
        groups = grouper.group(markets, data_source)
        assert groups == []

    def test_cluster_by_correlation(self):
        """Test the internal clustering algorithm."""
        import numpy as np

        grouper = CorrelationGrouper()
        market_ids = ["A", "B", "C", "D"]

        # A-B correlated, C-D correlated, but A/B not with C/D
        corr = np.array([
            [1.0, 0.8, 0.1, 0.2],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.9],
            [0.2, 0.1, 0.9, 1.0],
        ])

        clusters = grouper._cluster_by_correlation(market_ids, corr, 0.5)
        assert len(clusters) == 2

        # Find which cluster has A
        cluster_a = next(c for c in clusters if "A" in c)
        cluster_c = next(c for c in clusters if "C" in c)

        assert "B" in cluster_a
        assert "D" in cluster_c

    def test_protocol_conformance(self):
        from src.core.protocols import MarketGrouper
        grouper = CorrelationGrouper()
        assert isinstance(grouper, MarketGrouper)
