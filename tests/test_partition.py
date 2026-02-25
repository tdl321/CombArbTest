"""Test partition constraint checking for 3+ market combinatorial arbitrage.

Tests for detecting arbitrage when multi-market probabilities violate
the partition constraint (sum must equal 1).

NOTE: Partitions require 3+ markets for true combinatorial arbitrage.
Binary (2-market) cases are not considered valid partitions.
"""

import pytest
import sys
sys.path.insert(0, "/root/combarbbot")

from src.backtest.constraint_checker import (
    check_partition,
    is_partition_constraint,
    get_partition_market_ids,
    compute_partition_coherent_prices,
    compute_partition_trades,
    format_partition_violation,
    PartitionViolation,
)
from src.llm.schema import MarketCluster, MarketRelationship


class TestCheckPartition:
    """Tests for check_partition function."""

    def test_underpriced_partition(self):
        """Test detection when sum < 1 (underpriced) with 3 markets."""
        prices = {"A": 0.30, "B": 0.25, "C": 0.20}
        market_ids = ["A", "B", "C"]
        
        violation = check_partition(market_ids, prices)
        
        assert violation is not None
        assert violation.direction == "underpriced"
        assert abs(violation.violation_amount - (-0.25)) < 0.01
        assert abs(violation.total - 0.75) < 0.01
        # Coherent prices should sum to 1
        assert abs(sum(violation.coherent_prices.values()) - 1.0) < 0.01

    def test_overpriced_partition(self):
        """Test detection when sum > 1 (overpriced) with 3 markets."""
        prices = {"Trump": 0.50, "Harris": 0.45, "RFK": 0.20}
        market_ids = ["Trump", "Harris", "RFK"]
        
        violation = check_partition(market_ids, prices)
        
        assert violation is not None
        assert violation.direction == "overpriced"
        assert abs(violation.total - 1.15) < 0.01

    def test_two_market_returns_none(self):
        """Test that 2-market partitions return None (require 3+)."""
        prices = {"Yes": 0.70, "No": 0.50}  # Would be violation
        market_ids = ["Yes", "No"]
        
        violation = check_partition(market_ids, prices)
        
        assert violation is None  # Binary markets not valid partitions

    def test_valid_partition_no_violation(self):
        """Test that valid partitions (sum = 1) do not trigger violations."""
        prices = {"A": 0.50, "B": 0.30, "C": 0.20}
        market_ids = ["A", "B", "C"]
        
        violation = check_partition(market_ids, prices)
        
        assert violation is None

    def test_within_tolerance(self):
        """Test that small deviations within tolerance are ignored."""
        prices = {"A": 0.505, "B": 0.300, "C": 0.200}  # sum = 1.005
        market_ids = ["A", "B", "C"]
        
        violation = check_partition(market_ids, prices, tolerance=0.01)
        
        assert violation is None

    def test_custom_tolerance(self):
        """Test with custom tolerance value."""
        prices = {"A": 0.505, "B": 0.300, "C": 0.200}  # sum = 1.005
        market_ids = ["A", "B", "C"]
        
        # Should trigger with tighter tolerance
        violation = check_partition(market_ids, prices, tolerance=0.001)
        
        assert violation is not None
        assert violation.direction == "overpriced"

    def test_missing_prices(self):
        """Test handling of missing price data."""
        prices = {"A": 0.30}  # Missing B and C
        market_ids = ["A", "B", "C"]
        
        violation = check_partition(market_ids, prices)
        
        assert violation is None  # Cannot compute without all prices


class TestPartitionConstraintDetection:
    """Tests for is_partition_constraint function."""

    def test_detects_partition(self):
        """Test detection of a true partition (exhaustive + exclusive) with 3+ markets."""
        cluster = MarketCluster(
            cluster_id="election-2024",
            theme="2024 Presidential Election",
            market_ids=["trump", "harris", "rfk"],
            relationships=[
                MarketRelationship(type="exhaustive", from_market="trump", confidence=0.95),
                MarketRelationship(type="mutually_exclusive", from_market="trump", to_market="harris", confidence=0.95),
                MarketRelationship(type="mutually_exclusive", from_market="trump", to_market="rfk", confidence=0.95),
                MarketRelationship(type="mutually_exclusive", from_market="harris", to_market="rfk", confidence=0.95),
            ],
        )
        
        assert is_partition_constraint(cluster) is True

    def test_rejects_binary_market(self):
        """Test that 2-market clusters are not detected as partitions."""
        cluster = MarketCluster(
            cluster_id="binary",
            theme="Binary market",
            market_ids=["yes", "no"],
            relationships=[
                MarketRelationship(type="exhaustive", from_market="yes", confidence=0.95),
                MarketRelationship(type="mutually_exclusive", from_market="yes", to_market="no", confidence=0.95),
            ],
        )
        
        assert is_partition_constraint(cluster) is False

    def test_rejects_non_exhaustive(self):
        """Test that non-exhaustive clusters are not partitions."""
        cluster = MarketCluster(
            cluster_id="incomplete",
            theme="Not exhaustive",
            market_ids=["A", "B", "C"],
            relationships=[
                # Only has exclusivity, no exhaustive
                MarketRelationship(type="mutually_exclusive", from_market="A", to_market="B", confidence=0.95),
                MarketRelationship(type="mutually_exclusive", from_market="A", to_market="C", confidence=0.95),
                MarketRelationship(type="mutually_exclusive", from_market="B", to_market="C", confidence=0.95),
            ],
        )
        
        assert is_partition_constraint(cluster) is False

    def test_rejects_insufficient_exclusivity(self):
        """Test that clusters without enough exclusivity pairs are rejected."""
        cluster = MarketCluster(
            cluster_id="weak",
            theme="Weak exclusivity",
            market_ids=["A", "B", "C"],
            relationships=[
                MarketRelationship(type="exhaustive", from_market="A", confidence=0.95),
                # Only 1 of 3 pairs - below 50% threshold
                MarketRelationship(type="mutually_exclusive", from_market="A", to_market="B", confidence=0.95),
            ],
        )
        
        assert is_partition_constraint(cluster) is False


class TestCoherentPrices:
    """Tests for compute_partition_coherent_prices."""

    def test_normalizes_overpriced(self):
        """Test normalization of overpriced partition."""
        prices = {"A": 0.60, "B": 0.40, "C": 0.20}  # sum = 1.2
        
        coherent = compute_partition_coherent_prices(prices)
        
        assert abs(sum(coherent.values()) - 1.0) < 0.001
        assert abs(coherent["A"] - 0.5) < 0.001  # 0.6/1.2
        assert abs(coherent["B"] - 1/3) < 0.001  # 0.4/1.2
        assert abs(coherent["C"] - 1/6) < 0.001  # 0.2/1.2

    def test_normalizes_underpriced(self):
        """Test normalization of underpriced partition."""
        prices = {"A": 0.40, "B": 0.30, "C": 0.10}  # sum = 0.8
        
        coherent = compute_partition_coherent_prices(prices)
        
        assert abs(sum(coherent.values()) - 1.0) < 0.001


class TestTradeDirections:
    """Tests for compute_partition_trades."""

    def test_underpriced_all_buy(self):
        """Test that underpriced partitions result in BUY all."""
        prices = {"A": 0.30, "B": 0.30, "C": 0.20}
        coherent = {"A": 0.375, "B": 0.375, "C": 0.25}
        
        trades = compute_partition_trades(prices, coherent, "underpriced")
        
        assert all(d == "BUY" for d in trades.values())

    def test_overpriced_all_sell(self):
        """Test that overpriced partitions result in SELL all."""
        prices = {"A": 0.50, "B": 0.40, "C": 0.30}
        coherent = {"A": 0.417, "B": 0.333, "C": 0.25}
        
        trades = compute_partition_trades(prices, coherent, "overpriced")
        
        assert all(d == "SELL" for d in trades.values())


class TestProfitPotential:
    """Tests for profit potential calculation."""

    def test_profit_potential_underpriced(self):
        """Test profit potential calculation for underpriced partition."""
        violation = PartitionViolation(
            market_ids=["A", "B", "C"],
            prices={"A": 0.30, "B": 0.30, "C": 0.20},
            total=0.8,
            violation_amount=-0.2,
            direction="underpriced",
        )
        
        # profit_potential = |violation| / total = 0.2 / 0.8 = 0.25
        assert abs(violation.profit_potential - 0.25) < 0.01

    def test_profit_potential_overpriced(self):
        """Test profit potential calculation for overpriced partition."""
        violation = PartitionViolation(
            market_ids=["A", "B", "C"],
            prices={"A": 0.50, "B": 0.40, "C": 0.30},
            total=1.2,
            violation_amount=0.2,
            direction="overpriced",
        )
        
        # profit_potential = |violation| / total = 0.2 / 1.2 = 0.167
        assert abs(violation.profit_potential - 0.167) < 0.01


class TestGetPartitionMarketIds:
    """Tests for get_partition_market_ids."""

    def test_extracts_all_exclusive_markets(self):
        """Test extraction of markets from exclusive relationships."""
        cluster = MarketCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C", "D"],
            relationships=[
                MarketRelationship(type="exhaustive", from_market="A", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="A", to_market="B", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="A", to_market="C", confidence=0.9),
                MarketRelationship(type="mutually_exclusive", from_market="B", to_market="C", confidence=0.9),
            ],
        )
        
        ids = get_partition_market_ids(cluster)
        
        assert set(ids) == {"A", "B", "C"}

    def test_returns_all_ids_when_no_relationships(self):
        """Test fallback when no relationships exist."""
        cluster = MarketCluster(
            cluster_id="test",
            theme="Test",
            market_ids=["A", "B", "C"],
            relationships=[],
        )
        
        ids = get_partition_market_ids(cluster)
        
        assert ids == ["A", "B", "C"]


class TestFormatViolation:
    """Tests for format_partition_violation."""

    def test_format_includes_key_info(self):
        """Test that format includes market count, sum, direction."""
        violation = PartitionViolation(
            market_ids=["A", "B", "C"],
            prices={"A": 0.50, "B": 0.40, "C": 0.30},
            total=1.2,
            violation_amount=0.2,
            direction="overpriced",
        )
        
        formatted = format_partition_violation(violation)
        
        assert "3 markets" in formatted
        assert "1.2000" in formatted
        assert "overpriced" in formatted


class TestRealWorldScenarios:
    """Tests simulating real Polymarket-like scenarios."""

    def test_election_three_candidates(self):
        """Test 3-candidate election partition."""
        prices = {
            "trump": 0.52,
            "harris": 0.48,
            "rfk": 0.05,
        }  # Sum = 1.05, overpriced
        
        violation = check_partition(list(prices.keys()), prices)
        
        assert violation is not None
        assert violation.direction == "overpriced"
        assert violation.total == pytest.approx(1.05, abs=0.01)

    def test_championship_four_teams(self):
        """Test 4-team championship partition."""
        prices = {
            "lakers": 0.25,
            "celtics": 0.30,
            "warriors": 0.20,
            "nuggets": 0.15,
        }  # Sum = 0.90, underpriced
        
        violation = check_partition(list(prices.keys()), prices)
        
        assert violation is not None
        assert violation.direction == "underpriced"
        assert violation.total == pytest.approx(0.90, abs=0.01)

    def test_valid_championship_market(self):
        """Test valid championship market (no arbitrage)."""
        prices = {
            "team_a": 0.35,
            "team_b": 0.30,
            "team_c": 0.25,
            "team_d": 0.10,
        }  # Sum = 1.0
        
        violation = check_partition(list(prices.keys()), prices)
        
        assert violation is None

    def test_date_range_partition(self):
        """Test date range partition (e.g., which quarter will event occur)."""
        prices = {
            "q1_2024": 0.25,
            "q2_2024": 0.30,
            "q3_2024": 0.35,
            "q4_2024": 0.25,
        }  # Sum = 1.15, overpriced
        
        violation = check_partition(list(prices.keys()), prices)
        
        assert violation is not None
        assert violation.direction == "overpriced"
