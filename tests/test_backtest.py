"""Tests for the Backtest module.

Tests the core backtest components:
1. Report generation from opportunities
2. ArbitrageExtractor trade extraction
3. Optimizer integration
4. Data loading (integration, requires data files)
"""

import logging
import pytest
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


# =============================================================================
# Unit Tests (no external dependencies)
# =============================================================================


class TestReportGeneration:
    """Test report generation from ArbitrageOpportunity objects."""

    def test_report_basic(self):
        """Test basic report generation."""
        from src.backtest.schema import ArbitrageOpportunity
        from src.arbitrage.extractor import ArbitrageTrade
        from src.backtest.report import generate_report

        trade = ArbitrageTrade(
            constraint_type="implies",
            positions={"A": "SELL", "B": "BUY"},
            violation_amount=0.1,
            locked_profit=0.1,
            market_prices={"A": 0.7, "B": 0.5},
            description="Test implies trade",
        )
        opp = ArbitrageOpportunity(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            block_number=18000000,
            cluster_id="test_cluster",
            trade=trade,
            detection_method="solver",
        )

        report = generate_report(
            opportunities=[opp],
            start_date=datetime(2024, 1, 1, 10, 0, 0),
            end_date=datetime(2024, 1, 1, 12, 0, 0),
            markets_analyzed=2,
            clusters_found=1,
            cluster_themes={"test_cluster": "Test Theme"},
            cluster_market_ids={"test_cluster": ["A", "B"]},
        )

        assert report.total_opportunities == 1
        assert report.gross_pnl > 0

    def test_report_multiple_opps(self):
        """Test report with multiple opportunities."""
        from src.backtest.schema import ArbitrageOpportunity
        from src.arbitrage.extractor import ArbitrageTrade
        from src.backtest.report import generate_report

        opportunities = []
        for i, (profit, constraint) in enumerate([(0.1, "implies"), (0.2, "mutex"), (0.01, "implies")]):
            trade = ArbitrageTrade(
                constraint_type=constraint,
                positions={"A": "SELL", "B": "BUY"},
                violation_amount=profit,
                locked_profit=profit,
                market_prices={"A": 0.6, "B": 0.4},
                description=f"Test trade {i}",
            )
            opp = ArbitrageOpportunity(
                timestamp=datetime(2024, 1, 1, 10 + i, 0, 0),
                block_number=18000000 + i * 1000,
                cluster_id="test_cluster",
                trade=trade,
                detection_method="solver",
            )
            opportunities.append(opp)

        report = generate_report(
            opportunities=opportunities,
            start_date=datetime(2024, 1, 1, 10, 0, 0),
            end_date=datetime(2024, 1, 1, 13, 0, 0),
            markets_analyzed=2,
            clusters_found=1,
            cluster_themes={"test_cluster": "Test"},
            cluster_market_ids={"test_cluster": ["A", "B"]},
        )

        assert report.total_opportunities == 3
        assert report.gross_pnl == pytest.approx(0.31, abs=0.01)

    def test_report_format(self):
        """Test report formatting produces string output."""
        from src.backtest.schema import ArbitrageOpportunity
        from src.arbitrage.extractor import ArbitrageTrade
        from src.backtest.report import generate_report, format_report

        trade = ArbitrageTrade(
            constraint_type="implies",
            positions={"A": "SELL", "B": "BUY"},
            violation_amount=0.05,
            locked_profit=0.05,
            market_prices={"A": 0.55, "B": 0.50},
            description="Test implies trade",
        )
        opp = ArbitrageOpportunity(
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            block_number=18000000,
            cluster_id="test_cluster",
            trade=trade,
            detection_method="solver",
        )

        report = generate_report(
            opportunities=[opp],
            start_date=datetime(2024, 1, 1, 10, 0, 0),
            end_date=datetime(2024, 1, 1, 12, 0, 0),
            markets_analyzed=3,
            clusters_found=1,
            cluster_themes={"test_cluster": "Test"},
            cluster_market_ids={"test_cluster": ["A", "B"]},
        )

        formatted = format_report(report)
        assert isinstance(formatted, str)
        assert len(formatted) > 0


class TestArbitrageExtractor:
    """Test ArbitrageExtractor trade extraction."""

    def test_extractor_creation(self):
        """Test extractor can be created with defaults."""
        from src.arbitrage.extractor import ArbitrageExtractor
        extractor = ArbitrageExtractor()
        assert extractor.min_profit_threshold == 0.001
        assert extractor.fee_per_leg == 0.01

    def test_trade_net_profit(self):
        """Test ArbitrageTrade net profit calculation."""
        from src.arbitrage.extractor import ArbitrageTrade
        trade = ArbitrageTrade(
            constraint_type="implies",
            positions={"A": "SELL", "B": "BUY"},
            violation_amount=0.1,
            locked_profit=0.1,
            market_prices={"A": 0.7, "B": 0.5},
            description="Test trade",
        )
        # 2 legs * 0.01 fee = 0.02 fees
        assert trade.net_profit(fee_per_leg=0.01) == pytest.approx(0.08)
        assert trade.num_legs == 2

    def test_trade_zero_profit(self):
        """Test trade with zero profit."""
        from src.arbitrage.extractor import ArbitrageTrade
        trade = ArbitrageTrade(
            constraint_type="binary",
            positions={"A": "BUY"},
            violation_amount=0.0,
            locked_profit=0.0,
            market_prices={"A": 0.5},
            description="No profit trade",
        )
        assert trade.locked_profit == 0.0
        assert trade.net_profit(fee_per_leg=0.01) == -0.01


class TestOptimizerIntegration:
    """Test optimizer integration with the backtest pipeline."""

    def test_find_arbitrage_violation(self):
        """Test optimizer detects implication violation."""
        from src.optimizer import (
            find_arbitrage,
            RelationshipGraph,
            MarketCluster,
            MarketRelationship,
        )

        relationships = [
            MarketRelationship(
                type="implies",
                from_market="A",
                to_market="B",
                confidence=0.9,
            ),
        ]

        cluster = MarketCluster(
            cluster_id="test",
            theme="Test Cluster",
            market_ids=["A", "B"],
            relationships=relationships,
        )

        graph = RelationshipGraph(clusters=[cluster])

        # Prices violate implication (A > B, but A implies B)
        prices_violation = {"A": 0.7, "B": 0.5}
        result = find_arbitrage(prices_violation, graph)

        assert result.kl_divergence > 0
        assert result.has_arbitrage

    def test_find_arbitrage_no_violation(self):
        """Test optimizer returns no arbitrage when prices are consistent."""
        from src.optimizer import (
            find_arbitrage,
            RelationshipGraph,
            MarketCluster,
            MarketRelationship,
        )

        relationships = [
            MarketRelationship(
                type="implies",
                from_market="A",
                to_market="B",
                confidence=0.9,
            ),
        ]

        cluster = MarketCluster(
            cluster_id="test",
            theme="Test Cluster",
            market_ids=["A", "B"],
            relationships=relationships,
        )

        graph = RelationshipGraph(clusters=[cluster])

        # Prices satisfy implication (A < B)
        prices_ok = {"A": 0.4, "B": 0.6}
        result = find_arbitrage(prices_ok, graph)

        # Should have very small or zero KL divergence
        assert result.kl_divergence < 0.01


# =============================================================================
# Integration Tests (require data files on server)
# =============================================================================


@pytest.mark.integration
class TestDataLoading:
    """Integration tests for data loading (requires Polymarket data files)."""

    def test_market_loader(self):
        """Test loading markets from parquet data."""
        from src.data import MarketLoader
        loader = MarketLoader(DATA_DIR)
        markets_df = loader.query_markets(min_volume=1_000_000, limit=5)
        assert len(markets_df) > 0

    def test_block_loader(self):
        """Test loading block timestamps."""
        from src.data import BlockLoader
        loader = BlockLoader(DATA_DIR)
        assert loader is not None
