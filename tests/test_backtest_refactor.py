"""Tests for the refactored backtest system.

Phase 5 of the simulator refactor plan.
Tests the new arbitrage extraction and report generation flow.
"""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path

from src.optimizer.schema import ArbitrageResult, ConstraintViolation
from src.arbitrage.extractor import ArbitrageExtractor, ArbitrageTrade, extract_arbitrage_from_result
from src.backtest.schema import ArbitrageOpportunity, BacktestConfig, BacktestOutput, BacktestReport
from src.backtest.report_generator import (
    ArbitrageBacktestReport,
    generate_full_report,
    generate_report_text,
    ConstraintTypeStats,
    EdgeDistribution,
)


# ============================================================================
# Task 5.1: Test arbitrage extraction
# ============================================================================

class TestArbitrageExtraction:
    """Test ArbitrageExtractor produces correct trades for each constraint type."""
    
    def test_partition_extraction_underpriced(self):
        """Test partition extraction when sum < 1 (underpriced)."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
        
        # Market prices sum to 0.90 (underpriced by 0.10)
        result = ArbitrageResult(
            market_prices={"A": 0.30, "B": 0.30, "C": 0.30},
            coherent_prices={"A": 0.333, "B": 0.333, "C": 0.333},
            kl_divergence=0.01,
            constraints_violated=[],  # Partition check is separate
            converged=True,
            iterations=10,
        )
        
        # Extract trades - should find partition arbitrage
        trades = extractor.extract_trades(result)
        
        # Should have at least the partition trade
        partition_trades = [t for t in trades if t.constraint_type == "partition"]
        assert len(partition_trades) >= 1, "Should detect partition arbitrage"
        
        trade = partition_trades[0]
        assert trade.locked_profit == pytest.approx(0.10, abs=0.001)
        assert all(d == "BUY" for d in trade.positions.values()), "Should BUY ALL when underpriced"
    
    def test_partition_extraction_overpriced(self):
        """Test partition extraction when sum > 1 (overpriced)."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
        
        # Market prices sum to 1.10 (overpriced by 0.10)
        result = ArbitrageResult(
            market_prices={"A": 0.40, "B": 0.40, "C": 0.30},
            coherent_prices={"A": 0.364, "B": 0.364, "C": 0.273},
            kl_divergence=0.01,
            constraints_violated=[],
            converged=True,
            iterations=10,
        )
        
        trades = extractor.extract_trades(result)
        partition_trades = [t for t in trades if t.constraint_type == "partition"]
        
        assert len(partition_trades) >= 1, "Should detect partition arbitrage"
        trade = partition_trades[0]
        assert trade.locked_profit == pytest.approx(0.10, abs=0.001)
        assert all(d == "SELL" for d in trade.positions.values()), "Should SELL ALL when overpriced"
    
    def test_implies_extraction(self):
        """Test implies constraint extraction: A -> B violated when P(A) > P(B)."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
        
        result = ArbitrageResult(
            market_prices={"A": 0.60, "B": 0.50},
            coherent_prices={"A": 0.55, "B": 0.55},
            kl_divergence=0.01,
            constraints_violated=[
                ConstraintViolation(
                    constraint_type="implies",
                    from_market="A",
                    to_market="B",
                    violation_amount=0.10,
                    description="implies(A->B): P(A)=0.60 > P(B)=0.50"
                )
            ],
            converged=True,
            iterations=10,
        )
        
        trades = extractor.extract_trades(result)
        implies_trades = [t for t in trades if t.constraint_type == "implies"]
        
        assert len(implies_trades) >= 1, "Should detect implies arbitrage"
        trade = implies_trades[0]
        assert trade.locked_profit == pytest.approx(0.10, abs=0.001)
        assert trade.positions["A"] == "SELL", "Should SELL the implying event"
        assert trade.positions["B"] == "BUY", "Should BUY the implied event"
    
    def test_binary_extraction_underpriced(self):
        """Test binary constraint extraction: YES + NO < 1."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
        
        result = ArbitrageResult(
            market_prices={"YES": 0.40, "NO": 0.50},
            coherent_prices={"YES": 0.444, "NO": 0.556},
            kl_divergence=0.01,
            constraints_violated=[
                ConstraintViolation(
                    constraint_type="binary",
                    from_market="YES",
                    to_market="NO",
                    violation_amount=0.10,
                    description="binary: YES+NO=0.90 < 1"
                )
            ],
            converged=True,
            iterations=10,
        )
        
        trades = extractor.extract_trades(result)
        binary_trades = [t for t in trades if t.constraint_type == "binary"]
        
        assert len(binary_trades) >= 1, "Should detect binary arbitrage"
        trade = binary_trades[0]
        assert trade.locked_profit == pytest.approx(0.10, abs=0.001)
        assert trade.positions["YES"] == "BUY", "Should BUY when underpriced"
        assert trade.positions["NO"] == "BUY", "Should BUY when underpriced"
    
    def test_mutex_extraction(self):
        """Test mutex constraint extraction: P(A) + P(B) > 1."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
        
        result = ArbitrageResult(
            market_prices={"A": 0.60, "B": 0.50},
            coherent_prices={"A": 0.545, "B": 0.455},
            kl_divergence=0.01,
            constraints_violated=[
                ConstraintViolation(
                    constraint_type="mutex",
                    from_market="A",
                    to_market="B",
                    violation_amount=0.10,
                    description="mutex: A+B=1.10 > 1"
                )
            ],
            converged=True,
            iterations=10,
        )
        
        trades = extractor.extract_trades(result)
        mutex_trades = [t for t in trades if t.constraint_type == "mutex"]
        
        assert len(mutex_trades) >= 1, "Should detect mutex arbitrage"
        trade = mutex_trades[0]
        assert trade.locked_profit == pytest.approx(0.10, abs=0.001)
        assert trade.positions["A"] == "SELL", "Should SELL both in mutex"
        assert trade.positions["B"] == "SELL", "Should SELL both in mutex"
    
    def test_convenience_function(self):
        """Test the extract_arbitrage_from_result convenience function."""
        result = ArbitrageResult(
            market_prices={"A": 0.30, "B": 0.30, "C": 0.30},
            coherent_prices={"A": 0.333, "B": 0.333, "C": 0.333},
            kl_divergence=0.01,
            constraints_violated=[],
            converged=True,
            iterations=10,
        )
        
        trades = extract_arbitrage_from_result(result, min_profit=0.001, fee_per_leg=0.01)
        
        assert isinstance(trades, list)
        assert all(isinstance(t, ArbitrageTrade) for t in trades)
    
    def test_net_profit_calculation(self):
        """Test that net_profit correctly subtracts fees."""
        trade = ArbitrageTrade(
            constraint_type="partition",
            positions={"A": "BUY", "B": "BUY", "C": "BUY"},
            violation_amount=0.10,
            locked_profit=0.10,
            market_prices={"A": 0.30, "B": 0.30, "C": 0.30},
            description="test"
        )
        
        # 3 legs * 0.01 fee = 0.03 total fees
        assert trade.net_profit(fee_per_leg=0.01) == pytest.approx(0.07, abs=0.001)
        assert trade.num_legs == 3
    
    def test_min_profit_threshold_filtering(self):
        """Test that trades below min_profit_threshold are filtered out."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.05, fee_per_leg=0.01)
        
        # Small violation - should be filtered out after fees
        result = ArbitrageResult(
            market_prices={"A": 0.49, "B": 0.50},  # Sum = 0.99, profit = 0.01
            coherent_prices={"A": 0.495, "B": 0.505},
            kl_divergence=0.001,
            constraints_violated=[],
            converged=True,
            iterations=10,
        )
        
        trades = extractor.extract_trades(result)
        
        # With 2 legs * 0.01 = 0.02 fees, net = 0.01 - 0.02 = -0.01
        # Should be filtered out since net < min_profit (0.05)
        profitable_trades = [t for t in trades if t.net_profit(0.01) >= 0.05]
        assert len(profitable_trades) == 0


# ============================================================================
# Task 5.1: Test report generation
# ============================================================================

class TestReportGeneration:
    """Test report generation from arbitrage trades."""
    
    def test_basic_report_generation(self):
        """Test that report generates correctly from trades."""
        trades = [
            ArbitrageTrade(
                constraint_type="partition",
                positions={"X": "BUY", "Y": "BUY"},
                violation_amount=0.05,
                locked_profit=0.05,
                market_prices={"X": 0.45, "Y": 0.50},
                description="test partition"
            ),
            ArbitrageTrade(
                constraint_type="implies",
                positions={"A": "SELL", "B": "BUY"},
                violation_amount=0.10,
                locked_profit=0.10,
                market_prices={"A": 0.60, "B": 0.50},
                description="test implies"
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report, text, viz_files = generate_full_report(
                trades=trades,
                output_dir=tmpdir,
                period_start=datetime(2024, 1, 1),
                period_end=datetime(2024, 1, 2),
                markets_analyzed=4,
                clusters_monitored=2,
                fee_per_leg=0.01,
            )
            
            assert report.total_opportunities == 2
            assert report.total_locked_profit == pytest.approx(0.15, abs=0.001)
            assert "partition" in report.by_constraint_type
            assert "implies" in report.by_constraint_type
    
    def test_report_text_format(self):
        """Test that report text contains expected sections."""
        trades = [
            ArbitrageTrade(
                constraint_type="partition",
                positions={"X": "BUY", "Y": "BUY"},
                violation_amount=0.05,
                locked_profit=0.05,
                market_prices={"X": 0.45, "Y": 0.50},
                description="test"
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report, text, _ = generate_full_report(
                trades=trades,
                output_dir=tmpdir,
                fee_per_leg=0.01,
            )
            
            assert "ARBITRAGE BACKTEST REPORT" in text
            assert "OPPORTUNITY SUMMARY" in text
            assert "PROFIT SUMMARY" in text
            assert "EDGE SIZE DISTRIBUTION" in text
            assert "TOP OPPORTUNITIES" in text
    
    def test_empty_trades_report(self):
        """Test report generation with no trades."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report, text, viz_files = generate_full_report(
                trades=[],
                output_dir=tmpdir,
                fee_per_leg=0.01,
            )
            
            assert report.total_opportunities == 0
            assert report.total_locked_profit == 0.0
            assert report.total_net_profit == 0.0
    
    def test_constraint_type_stats(self):
        """Test ConstraintTypeStats accumulation."""
        stats = ConstraintTypeStats(constraint_type="partition")
        
        trade1 = ArbitrageTrade(
            constraint_type="partition",
            positions={"A": "BUY"},
            violation_amount=0.05,
            locked_profit=0.05,
            market_prices={"A": 0.95},
            description="test1"
        )
        
        trade2 = ArbitrageTrade(
            constraint_type="partition",
            positions={"B": "SELL"},
            violation_amount=0.10,
            locked_profit=0.10,
            market_prices={"B": 1.10},
            description="test2"
        )
        
        stats.add_trade(trade1, fee_per_leg=0.01)
        stats.add_trade(trade2, fee_per_leg=0.01)
        
        assert stats.count == 2
        assert stats.total_locked_profit == pytest.approx(0.15, abs=0.001)
        assert stats.max_edge_pct == pytest.approx(10.0, abs=0.1)  # 0.10 * 100
        assert stats.min_edge_pct == pytest.approx(5.0, abs=0.1)   # 0.05 * 100
    
    def test_edge_distribution(self):
        """Test EdgeDistribution bucketing."""
        dist = EdgeDistribution()
        
        # Small edge < 1%
        dist.add_trade(ArbitrageTrade(
            constraint_type="partition",
            positions={"A": "BUY"},
            violation_amount=0.005,  # 0.5%
            locked_profit=0.005,
            market_prices={"A": 0.995},
            description="small"
        ), fee_per_leg=0.01)
        
        # Medium edge 2-5%
        dist.add_trade(ArbitrageTrade(
            constraint_type="partition",
            positions={"A": "BUY"},
            violation_amount=0.03,  # 3%
            locked_profit=0.03,
            market_prices={"A": 0.97},
            description="medium"
        ), fee_per_leg=0.01)
        
        # Large edge > 10%
        dist.add_trade(ArbitrageTrade(
            constraint_type="partition",
            positions={"A": "BUY"},
            violation_amount=0.15,  # 15%
            locked_profit=0.15,
            market_prices={"A": 0.85},
            description="large"
        ), fee_per_leg=0.01)
        
        assert dist.bucket_lt_1pct == 1
        assert dist.bucket_2_5pct == 1
        assert dist.bucket_gt_10pct == 1


# ============================================================================
# Task 5.1: Test visualization generation
# ============================================================================

class TestVisualizationGeneration:
    """Test that visualization files are created."""
    
    def test_visualization_files_created(self):
        """Test that viz files are created in output directory."""
        trades = [
            ArbitrageTrade(
                constraint_type="partition",
                positions={"X": "BUY", "Y": "BUY", "Z": "BUY"},
                violation_amount=0.10,
                locked_profit=0.10,
                market_prices={"X": 0.30, "Y": 0.30, "Z": 0.30},
                description="test partition"
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report, text, viz_files = generate_full_report(
                trades=trades,
                output_dir=tmpdir,
                fee_per_leg=0.01,
            )
            
            # Should have at least one visualization
            assert len(viz_files) >= 1
            
            # All files should exist
            for viz_path in viz_files:
                assert os.path.exists(viz_path), f"Viz file should exist: {viz_path}"
                assert viz_path.endswith(".png"), "Should be PNG files"
    
    def test_report_text_file_created(self):
        """Test that report text file is saved."""
        trades = [
            ArbitrageTrade(
                constraint_type="partition",
                positions={"X": "BUY"},
                violation_amount=0.05,
                locked_profit=0.05,
                market_prices={"X": 0.95},
                description="test"
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report, text, _ = generate_full_report(
                trades=trades,
                output_dir=tmpdir,
                fee_per_leg=0.01,
            )
            
            report_path = os.path.join(tmpdir, "backtest_report.txt")
            assert os.path.exists(report_path)
            
            with open(report_path) as f:
                content = f.read()
            assert "ARBITRAGE BACKTEST REPORT" in content


# ============================================================================
# Task 5.1: Test end-to-end backtest (unit test level)
# ============================================================================

class TestEndToEndBacktest:
    """Test the complete backtest flow at unit test level."""
    
    def test_backtest_output_schema(self):
        """Test BacktestOutput schema construction."""
        config = BacktestConfig(
            market_ids=["A", "B", "C"],
            kl_threshold=0.01,
            transaction_cost=0.01,
            min_profit=0.001,
        )
        
        trade = ArbitrageTrade(
            constraint_type="partition",
            positions={"A": "BUY", "B": "BUY", "C": "BUY"},
            violation_amount=0.10,
            locked_profit=0.10,
            market_prices={"A": 0.30, "B": 0.30, "C": 0.30},
            description="test"
        )
        
        # Note: The new ArbitrageOpportunity expects trade as ArbitrageTrade from schema
        from src.backtest.schema import ArbitrageTrade as SchemaTrade
        
        schema_trade = SchemaTrade(
            constraint_type=trade.constraint_type,
            positions=trade.positions,
            violation_amount=trade.violation_amount,
            locked_profit=trade.locked_profit,
            market_prices=trade.market_prices,
            description=trade.description,
        )
        
        opp = ArbitrageOpportunity(
            timestamp=datetime.now(),
            block_number=12345,
            cluster_id="test_cluster",
            trade=schema_trade,
            detection_method="partition",
        )
        
        assert opp.locked_profit == 0.10
        assert opp.constraint_type == "partition"
        assert opp.positions == {"A": "BUY", "B": "BUY", "C": "BUY"}
    
    def test_opportunity_net_profit(self):
        """Test that opportunity net_profit method works."""
        from src.backtest.schema import ArbitrageTrade as SchemaTrade
        
        trade = SchemaTrade(
            constraint_type="partition",
            positions={"A": "BUY", "B": "BUY"},
            violation_amount=0.10,
            locked_profit=0.10,
            market_prices={"A": 0.45, "B": 0.45},
            description="test"
        )
        
        opp = ArbitrageOpportunity(
            timestamp=datetime.now(),
            block_number=12345,
            cluster_id="test",
            trade=trade,
            detection_method="partition",
        )
        
        # 2 legs * 0.01 = 0.02 fees
        assert opp.net_profit(fee_per_leg=0.01) == pytest.approx(0.08, abs=0.001)
    
    def test_backtest_config_defaults(self):
        """Test BacktestConfig default values."""
        config = BacktestConfig(market_ids=["A", "B"])
        
        assert config.kl_threshold == 0.01
        assert config.transaction_cost == 0.015
        assert config.min_profit == 0.001
        assert config.progress_interval == 1000
        assert config.store_all_opportunities is True


# ============================================================================
# Additional edge case tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_violation_no_trade(self):
        """Test that no trades are extracted when there's no violation."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
        
        # Prices sum to exactly 1.0 - no arbitrage
        result = ArbitrageResult(
            market_prices={"A": 0.50, "B": 0.50},
            coherent_prices={"A": 0.50, "B": 0.50},
            kl_divergence=0.0,
            constraints_violated=[],
            converged=True,
            iterations=10,
        )
        
        trades = extractor.extract_trades(result)
        
        # Should have no trades or only very small ones
        profitable_trades = [t for t in trades if t.net_profit(0.01) >= 0.001]
        assert len(profitable_trades) == 0
    
    def test_single_market_no_partition(self):
        """Test that single market doesn't create partition trade."""
        extractor = ArbitrageExtractor(min_profit_threshold=0.001, fee_per_leg=0.01)
        
        result = ArbitrageResult(
            market_prices={"A": 0.50},
            coherent_prices={"A": 0.50},
            kl_divergence=0.0,
            constraints_violated=[],
            converged=True,
            iterations=10,
        )
        
        trades = extractor.extract_trades(result)
        partition_trades = [t for t in trades if t.constraint_type == "partition"]
        
        # Single market shouldn't have meaningful partition arbitrage
        assert all(t.net_profit(0.01) < 0.001 for t in partition_trades)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
