"""Tests for Bregman projection visualization."""
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from datetime import datetime
import os

from src.visualization.bregman_plot import (
    BregmanAnalysis,
    compute_bregman_analysis,
    signal_to_bregman,
    plot_bregman_dual_panel,
    plot_bregman_report,
    plot_single_cluster_summary,
)
from src.visualization.schema import ArbitrageSignal


class TestBregmanAnalysis:
    """Tests for BregmanAnalysis dataclass and computation."""

    def test_compute_bregman_analysis_basic(self):
        """Test basic analysis computation."""
        outcomes = ["Yes", "No"]
        market_prices = {"Yes": 0.60, "No": 0.50}  # Sum = 1.10 (overpriced)
        coherent_prices = {"Yes": 0.545, "No": 0.455}  # Normalized

        analysis = compute_bregman_analysis(
            cluster_id="#1",
            question="Test market?",
            outcomes=outcomes,
            market_prices=market_prices,
            coherent_prices=coherent_prices,
            kl_divergence=0.05,
            iterations=100,
        )

        assert analysis.cluster_id == "#1"
        assert analysis.question == "Test market?"
        assert analysis.outcomes == outcomes
        assert analysis.observed == [0.60, 0.50]
        assert analysis.projected == [0.545, 0.455]
        assert abs(analysis.sum_p - 1.10) < 1e-6
        assert abs(analysis.overround - 10.0) < 1e-6
        assert analysis.kl_divergence == 0.05
        assert analysis.iterations == 100

    def test_compute_bregman_analysis_underpriced(self):
        """Test analysis with underpriced market."""
        outcomes = ["A", "B", "C"]
        market_prices = {"A": 0.30, "B": 0.25, "C": 0.35}  # Sum = 0.90
        coherent_prices = {"A": 0.333, "B": 0.278, "C": 0.389}

        analysis = compute_bregman_analysis(
            cluster_id="#2",
            question="Underpriced test?",
            outcomes=outcomes,
            market_prices=market_prices,
            coherent_prices=coherent_prices,
        )

        assert abs(analysis.sum_p - 0.90) < 1e-6
        assert abs(analysis.overround - (-10.0)) < 1e-6  # Negative = underpriced

    def test_signal_to_bregman_conversion(self):
        """Test conversion from ArbitrageSignal to BregmanAnalysis."""
        signal = ArbitrageSignal(
            timestamp=datetime.now(),
            cluster_id="cluster_001",
            markets=["Outcome A", "Outcome B"],
            constraint_type="mutually_exclusive",
            market_prices={"Outcome A": 0.55, "Outcome B": 0.55},
            coherent_prices={"Outcome A": 0.50, "Outcome B": 0.50},
            edge_magnitude=0.05,
            kl_divergence=0.01,
            direction={"Outcome A": -0.05, "Outcome B": -0.05},
            constraint_violation="Sum exceeds 1.0",
        )

        analysis = signal_to_bregman(signal, question="Will this happen?")

        assert analysis.cluster_id == "cluster_001"
        assert analysis.question == "Will this happen?"
        assert analysis.outcomes == ["Outcome A", "Outcome B"]
        assert abs(analysis.sum_p - 1.10) < 1e-6
        assert analysis.kl_divergence == 0.01


class TestBregmanDualPanel:
    """Tests for dual-panel visualization."""

    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis for testing."""
        return BregmanAnalysis(
            cluster_id="#1",
            question="Will Trump family launch a token by the end of 2025?",
            outcomes=[
                "Token by Dec 31 2024",
                "Token by Mar 31 2025",
                "Token by Jun 30 2025",
                "Token by Sep 30 2025",
                "Token by Dec 31 2025",
            ],
            observed=[0.35, 0.25, 0.22, 0.20, 0.224],  # sum=1.244
            projected=[0.281, 0.201, 0.177, 0.161, 0.180],  # normalized
            sum_p=1.244,
            overround=24.4,
            kl_divergence=0.08,
            iterations=845,
        )

    def test_plot_bregman_dual_panel_creates_figure(self, sample_analysis):
        """Test that dual panel plot creates a valid figure."""
        fig = plot_bregman_dual_panel(sample_analysis)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Check that figure has expected structure
        axes = fig.get_axes()
        assert len(axes) == 2  # Bar chart + radar chart

        plt.close(fig)

    def test_plot_bregman_dual_panel_saves_file(self, sample_analysis, tmp_path):
        """Test that plot can be saved to file."""
        save_path = str(tmp_path / "test_bregman.png")

        fig = plot_bregman_dual_panel(sample_analysis, save_path=save_path)

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

        plt.close(fig)

    def test_plot_bregman_dual_panel_custom_figsize(self, sample_analysis):
        """Test plot with custom figure size."""
        fig = plot_bregman_dual_panel(sample_analysis, figsize=(20, 8))

        assert fig.get_figwidth() == 20
        assert fig.get_figheight() == 8

        plt.close(fig)


class TestBregmanReport:
    """Tests for multi-page report generation."""

    @pytest.fixture
    def multiple_analyses(self):
        """Create multiple analysis objects for testing."""
        analyses = []
        for i in range(6):
            analyses.append(BregmanAnalysis(
                cluster_id=f"#{i+1}",
                question=f"Test market question {i+1}?",
                outcomes=[f"Outcome {j}" for j in range(3 + i % 3)],
                observed=[0.4 - 0.05*j for j in range(3 + i % 3)],
                projected=[0.35 - 0.04*j for j in range(3 + i % 3)],
                sum_p=1.0 + 0.1 * (i % 3),
                overround=10.0 * (i % 3),
                kl_divergence=0.01 * (i + 1),
                iterations=100 * (i + 1),
            ))
        return analyses

    def test_plot_bregman_report_creates_pages(self, multiple_analyses):
        """Test that report creates correct number of pages."""
        figures = plot_bregman_report(multiple_analyses, max_per_page=2)

        assert len(figures) == 3  # 6 analyses / 2 per page = 3 pages

        for fig in figures:
            plt.close(fig)

    def test_plot_bregman_report_saves_files(self, multiple_analyses, tmp_path):
        """Test that report saves multiple page files."""
        save_path = str(tmp_path / "report.png")

        figures = plot_bregman_report(multiple_analyses, save_path=save_path, max_per_page=3)

        assert os.path.exists(str(tmp_path / "report_page1.png"))
        assert os.path.exists(str(tmp_path / "report_page2.png"))

        for fig in figures:
            plt.close(fig)


class TestSingleClusterSummary:
    """Tests for three-panel cluster summary."""

    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis for testing."""
        return BregmanAnalysis(
            cluster_id="#1",
            question="Presidential election outcome?",
            outcomes=["Trump", "Biden", "Other"],
            observed=[0.52, 0.45, 0.08],  # sum=1.05
            projected=[0.495, 0.429, 0.076],
            sum_p=1.05,
            overround=5.0,
            kl_divergence=0.002,
            iterations=50,
        )

    def test_plot_single_cluster_summary_creates_figure(self, sample_analysis):
        """Test that summary plot creates a valid figure."""
        fig = plot_single_cluster_summary(sample_analysis)

        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Should have 3 axes: bar, radar, table
        axes = fig.get_axes()
        assert len(axes) == 3

        plt.close(fig)

    def test_plot_single_cluster_summary_saves_file(self, sample_analysis, tmp_path):
        """Test that summary can be saved."""
        save_path = str(tmp_path / "summary.png")

        fig = plot_single_cluster_summary(sample_analysis, save_path=save_path)

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_two_outcome_market(self):
        """Test with minimal 2-outcome market."""
        analysis = BregmanAnalysis(
            cluster_id="#1",
            question="Yes/No question?",
            outcomes=["Yes", "No"],
            observed=[0.60, 0.50],
            projected=[0.545, 0.455],
            sum_p=1.10,
            overround=10.0,
            kl_divergence=0.03,
        )

        fig = plot_bregman_dual_panel(analysis)
        assert fig is not None
        plt.close(fig)

    def test_many_outcome_market(self):
        """Test with many outcomes (10+)."""
        n = 12
        analysis = BregmanAnalysis(
            cluster_id="#1",
            question="Which month will event occur?",
            outcomes=[f"Month {i+1}" for i in range(n)],
            observed=[1.0/n + 0.02 for _ in range(n)],
            projected=[1.0/n for _ in range(n)],
            sum_p=1.0 + 0.02 * n,
            overround=2.0 * n,
            kl_divergence=0.05,
        )

        fig = plot_bregman_dual_panel(analysis)
        assert fig is not None
        plt.close(fig)

    def test_very_long_outcome_names(self):
        """Test with very long outcome names."""
        analysis = BregmanAnalysis(
            cluster_id="#1",
            question="A very long question that might need truncation in the display?",
            outcomes=[
                "This is a very long outcome name that should be truncated",
                "Another extremely long outcome name for testing purposes",
                "Yet another lengthy outcome description here",
            ],
            observed=[0.40, 0.35, 0.30],
            projected=[0.38, 0.33, 0.29],
            sum_p=1.05,
            overround=5.0,
            kl_divergence=0.01,
        )

        fig = plot_bregman_dual_panel(analysis)
        assert fig is not None
        plt.close(fig)

    def test_zero_kl_divergence(self):
        """Test with zero KL divergence (perfect match)."""
        analysis = BregmanAnalysis(
            cluster_id="#1",
            question="Fair market?",
            outcomes=["A", "B", "C"],
            observed=[0.33, 0.33, 0.34],
            projected=[0.33, 0.33, 0.34],
            sum_p=1.0,
            overround=0.0,
            kl_divergence=0.0,
        )

        fig = plot_bregman_dual_panel(analysis)
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Test simplex visualization."""
import numpy as np
from datetime import datetime
from src.visualization import SimplexProjector, ArbitrageSignal, plot_arbitrage_signal

def test_simplex_projector_2d():
    proj = SimplexProjector(2)
    assert proj.vertices.shape == (2, 2)
    
    # Test projection
    pt = proj.to_2d(np.array([0.5, 0.5]))
    assert pt.shape == (2,)

def test_simplex_projector_3d():
    proj = SimplexProjector(3)
    assert proj.vertices.shape == (3, 2)
    
    # Center of triangle
    pt = proj.to_2d(np.array([1/3, 1/3, 1/3]))
    expected_center = proj.vertices.mean(axis=0)
    np.testing.assert_array_almost_equal(pt, expected_center)

def test_simplex_projector_4d():
    proj = SimplexProjector(4)
    assert proj.vertices.shape == (4, 2)
    
    # Test uniform projection
    pt = proj.to_2d(np.array([0.25, 0.25, 0.25, 0.25]))
    assert pt.shape == (2,)

def test_simplex_projector_feasibility():
    proj = SimplexProjector(3)
    
    # Valid probability vector
    assert proj.is_feasible(np.array([0.3, 0.3, 0.4]))
    
    # Invalid: sum > 1
    assert not proj.is_feasible(np.array([0.5, 0.5, 0.5]))
    
    # Invalid: negative
    assert not proj.is_feasible(np.array([0.5, 0.5, -0.1]))

def test_simplex_distance():
    proj = SimplexProjector(3)
    
    # Point on simplex has zero distance
    dist = proj.distance_to_simplex(np.array([0.3, 0.3, 0.4]))
    assert dist < 1e-6
    
    # Point outside has positive distance
    dist = proj.distance_to_simplex(np.array([0.5, 0.5, 0.5]))
    assert dist > 0

def test_plot_signal():
    signal = ArbitrageSignal(
        timestamp=datetime.now(),
        cluster_id="test-cluster",
        markets=["A", "B", "C"],
        constraint_type="mutually_exclusive",
        market_prices={"A": 0.40, "B": 0.35, "C": 0.35},  # sum=1.10
        coherent_prices={"A": 0.364, "B": 0.318, "C": 0.318},
        edge_magnitude=0.05,
        kl_divergence=0.012,
        direction={"A": -0.036, "B": -0.032, "C": -0.032},
        constraint_violation="sum=1.10 > 1.0 (partition)",
    )
    
    fig = plot_arbitrage_signal(signal)
    assert fig is not None
    # Save test output
    fig.savefig("/tmp/test_signal_plot.png")
    print("Plot saved to /tmp/test_signal_plot.png")

def test_arbitrage_signal_creation():
    """Test creating an ArbitrageSignal."""
    signal = ArbitrageSignal(
        timestamp=datetime.now(),
        cluster_id="cluster-1",
        markets=["M1", "M2"],
        constraint_type="implies",
        market_prices={"M1": 0.6, "M2": 0.3},
        coherent_prices={"M1": 0.45, "M2": 0.45},
        edge_magnitude=0.212,
        kl_divergence=0.05,
        direction={"M1": -0.15, "M2": 0.15},
        constraint_violation="P(M1) > P(M2) but M1 implies M2",
        block_number=12345,
        detection_method="simple",
    )
    
    assert signal.cluster_id == "cluster-1"
    assert len(signal.markets) == 2
    assert signal.edge_magnitude > 0
    assert signal.detection_method == "simple"

if __name__ == "__main__":
    test_simplex_projector_2d()
    print("test_simplex_projector_2d passed")
    
    test_simplex_projector_3d()
    print("test_simplex_projector_3d passed")
    
    test_simplex_projector_4d()
    print("test_simplex_projector_4d passed")
    
    test_simplex_projector_feasibility()
    print("test_simplex_projector_feasibility passed")
    
    test_simplex_distance()
    print("test_simplex_distance passed")
    
    test_arbitrage_signal_creation()
    print("test_arbitrage_signal_creation passed")
    
    test_plot_signal()
    print("test_plot_signal passed")
    
    print("\nAll tests passed!")
