"""Tests for the partition-focused constraint_checker module.

Tests partition detection and violation checking for 3+ market
combinatorial arbitrage.
"""

import pytest
from dataclasses import dataclass
from typing import Optional

from src.backtest.constraint_checker import (
    PartitionViolation,
    check_partition,
    is_partition_constraint,
    get_partition_market_ids,
    compute_partition_coherent_prices,
    compute_partition_trades,
    format_partition_violation,
)


# =============================================================================
# Mock cluster for testing
# =============================================================================

@dataclass
class MockRelationship:
    type: str
    from_market: str
    to_market: Optional[str] = None
    confidence: float = 0.9


@dataclass
class MockCluster:
    cluster_id: str
    theme: str
    market_ids: list[str]
    relationships: list[MockRelationship]


# =============================================================================
# Tests for check_partition
# =============================================================================

class TestCheckPartition:
    """Tests for partition sum constraint: Σ P = 1."""
    
    def test_no_violation_when_sum_equals_one(self):
        """No violation when partition sums to exactly 1."""
        market_ids = ["A", "B", "C"]
        prices = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = check_partition(market_ids, prices)
        assert result is None
    
    def test_no_violation_within_tolerance(self):
        """No violation when sum is within tolerance of 1."""
        market_ids = ["A", "B", "C"]
        prices = {"A": 0.4, "B": 0.35, "C": 0.26}  # Sum = 1.01
        result = check_partition(market_ids, prices, tolerance=0.02)
        assert result is None
    
    def test_overpriced_violation_when_sum_exceeds_one(self):
        """Violation detected when partition sum > 1."""
        market_ids = ["A", "B", "C"]
        prices = {"A": 0.5, "B": 0.4, "C": 0.3}  # Sum = 1.2
        result = check_partition(market_ids, prices)
        
        assert result is not None
        assert result.direction == "overpriced"
        assert result.total == pytest.approx(1.2)
        assert result.violation_amount == pytest.approx(0.2)
    
    def test_underpriced_violation_when_sum_less_than_one(self):
        """Violation detected when partition sum < 1."""
        market_ids = ["A", "B", "C"]
        prices = {"A": 0.3, "B": 0.2, "C": 0.3}  # Sum = 0.8
        result = check_partition(market_ids, prices)
        
        assert result is not None
        assert result.direction == "underpriced"
        assert result.total == pytest.approx(0.8)
        assert result.violation_amount == pytest.approx(-0.2)
    
    def test_requires_three_or_more_markets(self):
        """Returns None for partitions with fewer than 3 markets."""
        market_ids = ["A", "B"]
        prices = {"A": 0.6, "B": 0.6}  # Sum = 1.2, would be violation
        result = check_partition(market_ids, prices)
        assert result is None  # But we require 3+ markets
    
    def test_returns_none_when_price_missing(self):
        """Returns None when not all prices are available."""
        market_ids = ["A", "B", "C"]
        prices = {"A": 0.5, "B": 0.4}  # C is missing
        result = check_partition(market_ids, prices)
        assert result is None
    
    def test_four_market_partition(self):
        """Works correctly with 4 markets."""
        market_ids = ["A", "B", "C", "D"]
        prices = {"A": 0.4, "B": 0.3, "C": 0.3, "D": 0.2}  # Sum = 1.2
        result = check_partition(market_ids, prices)
        
        assert result is not None
        assert result.direction == "overpriced"
        assert len(result.market_ids) == 4
    
    def test_coherent_prices_calculated(self):
        """Coherent prices are normalized to sum to 1."""
        market_ids = ["A", "B", "C"]
        prices = {"A": 0.5, "B": 0.4, "C": 0.3}  # Sum = 1.2
        result = check_partition(market_ids, prices)
        
        assert result is not None
        assert sum(result.coherent_prices.values()) == pytest.approx(1.0)
        # Each price should be price/total
        assert result.coherent_prices["A"] == pytest.approx(0.5 / 1.2)
        assert result.coherent_prices["B"] == pytest.approx(0.4 / 1.2)
        assert result.coherent_prices["C"] == pytest.approx(0.3 / 1.2)


# =============================================================================
# Tests for is_partition_constraint
# =============================================================================

class TestIsPartitionConstraint:
    """Tests for identifying valid partition clusters."""
    
    def test_valid_partition_with_exhaustive_and_exclusive(self):
        """Identifies valid partition with exhaustive + mutual exclusivity."""
        cluster = MockCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C"],
            relationships=[
                MockRelationship(type="exhaustive", from_market="A"),
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="B"),
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="C"),
                MockRelationship(type="mutually_exclusive", from_market="B", to_market="C"),
            ],
        )
        assert is_partition_constraint(cluster) is True
    
    def test_requires_exhaustive_constraint(self):
        """Requires exhaustive constraint to be a partition."""
        cluster = MockCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C"],
            relationships=[
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="B"),
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="C"),
                MockRelationship(type="mutually_exclusive", from_market="B", to_market="C"),
            ],
        )
        assert is_partition_constraint(cluster) is False
    
    def test_requires_sufficient_exclusive_pairs(self):
        """Requires at least 50% of expected exclusive pairs."""
        cluster = MockCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C"],  # Expected: 3 pairs
            relationships=[
                MockRelationship(type="exhaustive", from_market="A"),
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="B"),
                # Only 1 out of 3 pairs (33%) - below 50% threshold
            ],
        )
        assert is_partition_constraint(cluster) is False
    
    def test_requires_three_or_more_markets(self):
        """Requires 3+ markets for combinatorial arbitrage."""
        cluster = MockCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B"],  # Only 2 markets
            relationships=[
                MockRelationship(type="exhaustive", from_market="A"),
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="B"),
            ],
        )
        assert is_partition_constraint(cluster) is False
    
    def test_empty_relationships_not_partition(self):
        """Empty relationships means not a partition."""
        cluster = MockCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C"],
            relationships=[],
        )
        assert is_partition_constraint(cluster) is False


# =============================================================================
# Tests for get_partition_market_ids
# =============================================================================

class TestGetPartitionMarketIds:
    """Tests for extracting partition market IDs from cluster."""
    
    def test_gets_all_exclusive_markets(self):
        """Gets all markets connected by exclusive relationships."""
        cluster = MockCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C", "D"],
            relationships=[
                MockRelationship(type="exhaustive", from_market="A"),
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="B"),
                MockRelationship(type="mutually_exclusive", from_market="A", to_market="C"),
                MockRelationship(type="mutually_exclusive", from_market="B", to_market="C"),
            ],
        )
        result = get_partition_market_ids(cluster)
        assert set(result) == {"A", "B", "C"}
    
    def test_returns_cluster_ids_as_fallback(self):
        """Returns all cluster IDs if no relationships found."""
        cluster = MockCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C"],
            relationships=[],
        )
        result = get_partition_market_ids(cluster)
        assert result == ["A", "B", "C"]


# =============================================================================
# Tests for compute_partition_coherent_prices
# =============================================================================

class TestComputePartitionCoherentPrices:
    """Tests for computing arbitrage-free prices."""
    
    def test_normalizes_to_sum_one(self):
        """Coherent prices sum to exactly 1."""
        prices = {"A": 0.5, "B": 0.3, "C": 0.2}  # Sum = 1.0
        result = compute_partition_coherent_prices(prices)
        assert sum(result.values()) == pytest.approx(1.0)
    
    def test_normalizes_overpriced(self):
        """Correctly normalizes overpriced partition."""
        prices = {"A": 0.6, "B": 0.4, "C": 0.2}  # Sum = 1.2
        result = compute_partition_coherent_prices(prices)
        assert sum(result.values()) == pytest.approx(1.0)
        assert result["A"] == pytest.approx(0.5)  # 0.6/1.2
        assert result["B"] == pytest.approx(1/3)  # 0.4/1.2
        assert result["C"] == pytest.approx(1/6)  # 0.2/1.2
    
    def test_normalizes_underpriced(self):
        """Correctly normalizes underpriced partition."""
        prices = {"A": 0.3, "B": 0.3, "C": 0.2}  # Sum = 0.8
        result = compute_partition_coherent_prices(prices)
        assert sum(result.values()) == pytest.approx(1.0)


# =============================================================================
# Tests for compute_partition_trades
# =============================================================================

class TestComputePartitionTrades:
    """Tests for computing trade directions."""
    
    def test_underpriced_means_buy_all(self):
        """Underpriced partition: BUY all outcomes."""
        prices = {"A": 0.3, "B": 0.3, "C": 0.2}
        coherent = {"A": 0.375, "B": 0.375, "C": 0.25}
        result = compute_partition_trades(prices, coherent, "underpriced")
        assert all(d == "BUY" for d in result.values())
    
    def test_overpriced_means_sell_all(self):
        """Overpriced partition: SELL all outcomes."""
        prices = {"A": 0.5, "B": 0.4, "C": 0.3}
        coherent = {"A": 0.417, "B": 0.333, "C": 0.25}
        result = compute_partition_trades(prices, coherent, "overpriced")
        assert all(d == "SELL" for d in result.values())


# =============================================================================
# Tests for format_partition_violation
# =============================================================================

class TestFormatPartitionViolation:
    """Tests for formatting violation descriptions."""
    
    def test_formats_correctly(self):
        """Formats violation with key information."""
        violation = PartitionViolation(
            market_ids=["A", "B", "C"],
            prices={"A": 0.5, "B": 0.4, "C": 0.3},
            total=1.2,
            violation_amount=0.2,
            direction="overpriced",
        )
        result = format_partition_violation(violation)
        
        assert "Partition(3 markets)" in result
        assert "sum=1.2000" in result
        assert "overpriced" in result
        assert "0.2000" in result


# =============================================================================
# Tests for PartitionViolation properties
# =============================================================================

class TestPartitionViolation:
    """Tests for PartitionViolation dataclass."""
    
    def test_profit_potential_calculation(self):
        """Profit potential is |violation| / total."""
        violation = PartitionViolation(
            market_ids=["A", "B", "C"],
            prices={"A": 0.5, "B": 0.4, "C": 0.3},
            total=1.2,
            violation_amount=0.2,
            direction="overpriced",
        )
        # Profit potential = 0.2 / 1.2 = 0.1667
        assert violation.profit_potential == pytest.approx(0.2 / 1.2)
    
    def test_auto_calculates_coherent_prices(self):
        """Coherent prices auto-calculated in __post_init__."""
        violation = PartitionViolation(
            market_ids=["A", "B", "C"],
            prices={"A": 0.5, "B": 0.4, "C": 0.3},
            total=1.2,
            violation_amount=0.2,
            direction="overpriced",
        )
        assert sum(violation.coherent_prices.values()) == pytest.approx(1.0)
