#!/usr/bin/env python3
"""Unit tests for the Optimization Engine.

Tests OPT-01 through OPT-05 with known arbitrage cases.
"""

import sys
sys.path.insert(0, "/root/combarbbot")

import numpy as np
import pytest

from src.optimizer import (
    # Schema
    MarketRelationship,
    MarketCluster,
    RelationshipGraph,
    ArbitrageResult,
    OptimizationConfig,
    # Divergence
    kl_divergence,
    kl_gradient,
    # LMO
    ConstraintBuilder,
    LinearMinimizationOracle,
    # Frank-Wolfe
    frank_wolfe,
    barrier_frank_wolfe,
    find_arbitrage,
    find_arbitrage_simple,
)


class TestKLDivergence:
    """Test OPT-04: KL Divergence calculations."""
    
    def test_kl_identical(self):
        """KL(p || p) = 0."""
        p = np.array([0.3, 0.7])
        assert kl_divergence(p, p) < 1e-10
    
    def test_kl_positive(self):
        """KL is always non-negative."""
        p = np.array([0.2, 0.8])
        q = np.array([0.4, 0.6])
        assert kl_divergence(p, q) >= 0
    
    def test_kl_asymmetric(self):
        """KL(p || q) != KL(q || p) in general."""
        p = np.array([0.2, 0.8])
        q = np.array([0.4, 0.6])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert kl_pq != kl_qp
    
    def test_kl_gradient_direction(self):
        """Gradient should point toward reducing divergence."""
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        grad = kl_gradient(p, q)
        # Gradient points away from p, so where p < q, gradient is positive
        # (increase q makes it farther), where p > q, gradient is negative
        # p[0]=0.3 < q[0]=0.5: grad[0] should be positive
        # p[1]=0.7 > q[1]=0.5: grad[1] should be negative
        assert grad[0] > 0  # Moving q[0] up increases divergence
        assert grad[1] < 0  # Moving q[1] up decreases divergence


class TestConstraintBuilder:
    """Test OPT-02: Constraint building."""
    
    def test_implies_constraint(self):
        """Test A implies B: P(A) <= P(B)."""
        builder = ConstraintBuilder(["A", "B"])
        builder.add_implies("A", "B")
        constraints = builder.build()
        
        # Should have 1 constraint: x[0] - x[1] <= 0
        assert constraints.A.shape == (1, 2)
        assert constraints.A[0, 0] == 1.0
        assert constraints.A[0, 1] == -1.0
        assert constraints.b[0] == 0.0
    
    def test_mutual_exclusivity(self):
        """Test mutual exclusivity: P(A) + P(B) <= 1."""
        builder = ConstraintBuilder(["A", "B"])
        builder.add_mutually_exclusive("A", "B")
        constraints = builder.build()
        
        # Should have 1 constraint: x[0] + x[1] <= 1
        assert constraints.A.shape == (1, 2)
        assert constraints.A[0, 0] == 1.0
        assert constraints.A[0, 1] == 1.0
        assert constraints.b[0] == 1.0
    
    def test_binary_market(self):
        """Test binary market: P(YES) + P(NO) = 1."""
        builder = ConstraintBuilder(["YES", "NO"])
        builder.add_binary_market("YES", "NO")
        constraints = builder.build()
        
        # Should have 2 constraints for equality
        assert constraints.A.shape == (2, 2)


class TestLMO:
    """Test OPT-02: Linear Minimization Oracle."""
    
    def test_lmo_simple(self):
        """LMO should find vertex in direction of gradient."""
        builder = ConstraintBuilder(["A", "B"])
        builder.add_mutually_exclusive("A", "B")
        constraints = builder.build()
        
        lmo = LinearMinimizationOracle(constraints)
        
        # Gradient pointing toward A
        g = np.array([-1.0, 0.0])
        x, obj = lmo.solve(g)
        
        # Should maximize A (minimize -A), so A=1
        assert x[0] > 0.9
    
    def test_lmo_respects_constraints(self):
        """LMO solution should satisfy constraints."""
        builder = ConstraintBuilder(["A", "B"])
        builder.add_implies("A", "B")  # P(A) <= P(B)
        constraints = builder.build()
        
        lmo = LinearMinimizationOracle(constraints)
        
        g = np.array([-1.0, 1.0])  # Want high A, low B
        x, _ = lmo.solve(g)
        
        # Should have A <= B
        assert x[0] <= x[1] + 1e-6
    
    def test_lmo_extreme_points(self):
        """Should find multiple extreme points."""
        builder = ConstraintBuilder(["A", "B"])
        constraints = builder.build()  # Just box constraints
        
        lmo = LinearMinimizationOracle(constraints)
        points = lmo.get_extreme_points(10)
        
        # Should find vertices of [0,1]^2
        assert len(points) >= 2


class TestFrankWolfe:
    """Test OPT-01, OPT-03, OPT-05: Frank-Wolfe solver."""
    
    def test_implies_arbitrage(self):
        """Test: A implies B but P(A) > P(B) is arbitrage.
        
        Example: Market A at 0.6, Market B at 0.5
        A implies B means P(A) <= P(B), so this is violated.
        Coherent prices should satisfy P(A) <= P(B).
        """
        builder = ConstraintBuilder(["A", "B"])
        builder.add_implies("A", "B")
        constraints = builder.build()
        
        # Prices violating A implies B
        prices = {"A": 0.6, "B": 0.5}
        
        config = OptimizationConfig(max_iterations=5000, tolerance=1e-4)
        result = frank_wolfe(prices, constraints, config)
        
        # Coherent prices should satisfy A <= B
        assert result.coherent_prices["A"] <= result.coherent_prices["B"] + 1e-4
        
        # Original violated, so KL > 0
        assert result.kl_divergence > 1e-6
        
        print("Original: A=%s, B=%s" % (prices["A"], prices["B"]))
        print("Coherent: A=%.4f, B=%.4f" % (result.coherent_prices["A"], result.coherent_prices["B"]))
        print("KL divergence: %.6f" % result.kl_divergence)
    
    def test_already_coherent(self):
        """If prices already satisfy constraints, minimal change."""
        builder = ConstraintBuilder(["A", "B"])
        builder.add_implies("A", "B")
        constraints = builder.build()
        
        # Prices already satisfying A implies B
        prices = {"A": 0.4, "B": 0.6}
        
        result = frank_wolfe(prices, constraints)
        
        # Should converge quickly with low KL
        assert result.kl_divergence < 0.01
    
    def test_barrier_variant(self):
        """Test barrier Frank-Wolfe for numerical stability."""
        builder = ConstraintBuilder(["A", "B"])
        builder.add_implies("A", "B")
        builder.add_mutually_exclusive("A", "B")  # Extra constraint
        constraints = builder.build()
        
        prices = {"A": 0.6, "B": 0.5}
        
        result = barrier_frank_wolfe(prices, constraints)
        
        assert result.coherent_prices["A"] <= result.coherent_prices["B"] + 1e-4
        assert result.coherent_prices["A"] + result.coherent_prices["B"] <= 1.0 + 1e-4
    
    def test_three_markets(self):
        """Test with three markets and chain of implications."""
        # A implies B implies C
        builder = ConstraintBuilder(["A", "B", "C"])
        builder.add_implies("A", "B")
        builder.add_implies("B", "C")
        constraints = builder.build()
        
        # Violates chain: A > B and B > C
        prices = {"A": 0.7, "B": 0.5, "C": 0.3}
        
        config = OptimizationConfig(max_iterations=5000, tolerance=1e-4)
        result = frank_wolfe(prices, constraints, config)
        
        # Should have A <= B <= C
        assert result.coherent_prices["A"] <= result.coherent_prices["B"] + 1e-3
        assert result.coherent_prices["B"] <= result.coherent_prices["C"] + 1e-3
        
        print("Original: A=%s, B=%s, C=%s" % (prices["A"], prices["B"], prices["C"]))
        print("Coherent: A=%.4f, B=%.4f, C=%.4f" % (result.coherent_prices["A"], result.coherent_prices["B"], result.coherent_prices["C"]))


class TestHighLevelAPI:
    """Test the high-level find_arbitrage API."""
    
    def test_find_arbitrage_with_graph(self):
        """Test full pipeline with RelationshipGraph."""
        graph = RelationshipGraph(
            clusters=[
                MarketCluster(
                    cluster_id="test",
                    market_ids=["A", "B"],
                    relationships=[
                        MarketRelationship(
                            type="implies",
                            from_market="A",
                            to_market="B",
                            confidence=1.0,
                        )
                    ],
                )
            ]
        )
        
        prices = {"A": 0.6, "B": 0.5}
        
        result = find_arbitrage(prices, graph)
        
        assert result.has_arbitrage
        assert result.coherent_prices["A"] <= result.coherent_prices["B"] + 1e-4
    
    def test_mutual_exclusivity_arbitrage(self):
        """Test arbitrage from mutual exclusivity violation."""
        graph = RelationshipGraph(
            clusters=[
                MarketCluster(
                    cluster_id="test",
                    market_ids=["A", "B"],
                    relationships=[
                        MarketRelationship(
                            type="mutually_exclusive",
                            from_market="A",
                            to_market="B",
                            confidence=1.0,
                        )
                    ],
                )
            ]
        )
        
        # Violates P(A) + P(B) <= 1
        prices = {"A": 0.7, "B": 0.6}
        
        result = find_arbitrage(prices, graph)
        
        assert result.has_arbitrage
        assert result.coherent_prices["A"] + result.coherent_prices["B"] <= 1.0 + 1e-4
        
        print("Original sum: %s" % (prices["A"] + prices["B"]))
        print("Coherent sum: %.4f" % (result.coherent_prices["A"] + result.coherent_prices["B"]))


def run_demo():
    """Run a demonstration of the optimization engine."""
    print("=" * 60)
    print("Optimization Engine Demo")
    print("=" * 60)
    
    # Example: Election market with implication
    # "Trump wins" implies "Republican wins presidency"
    print("\n1. Implication Constraint Demo")
    print("-" * 40)
    
    graph = RelationshipGraph(
        clusters=[
            MarketCluster(
                cluster_id="election",
                market_ids=["trump_wins", "republican_wins"],
                relationships=[
                    MarketRelationship(
                        type="implies",
                        from_market="trump_wins",
                        to_market="republican_wins",
                        confidence=1.0,
                    )
                ],
            )
        ]
    )
    
    # Arbitrage: Trump winning is priced higher than Republican winning
    prices = {"trump_wins": 0.45, "republican_wins": 0.40}
    
    result = find_arbitrage(prices, graph)
    
    print("Market prices: trump_wins=%s, republican_wins=%s" % (prices["trump_wins"], prices["republican_wins"]))
    print("Constraint: trump_wins implies republican_wins")
    print("Violation: %s > %s (should be <=)" % (prices["trump_wins"], prices["republican_wins"]))
    print("\nCoherent prices:")
    print("  trump_wins: %.4f" % result.coherent_prices["trump_wins"])
    print("  republican_wins: %.4f" % result.coherent_prices["republican_wins"])
    print("KL divergence: %.6f" % result.kl_divergence)
    print("Converged: %s in %d iterations" % (result.converged, result.iterations))
    
    # Example 2: Mutually exclusive outcomes
    print("\n2. Mutual Exclusivity Demo")
    print("-" * 40)
    
    graph2 = RelationshipGraph(
        clusters=[
            MarketCluster(
                cluster_id="winner",
                market_ids=["candidate_A_wins", "candidate_B_wins"],
                relationships=[
                    MarketRelationship(
                        type="mutually_exclusive",
                        from_market="candidate_A_wins",
                        to_market="candidate_B_wins",
                        confidence=1.0,
                    )
                ],
            )
        ]
    )
    
    # Arbitrage: Both candidates priced to win with combined prob > 1
    prices2 = {"candidate_A_wins": 0.55, "candidate_B_wins": 0.55}
    
    result2 = find_arbitrage(prices2, graph2)
    
    print("Market prices: A_wins=%s, B_wins=%s" % (prices2["candidate_A_wins"], prices2["candidate_B_wins"]))
    print("Sum: %s > 1.0 (arbitrage!)" % (prices2["candidate_A_wins"] + prices2["candidate_B_wins"]))
    print("\nCoherent prices:")
    print("  candidate_A_wins: %.4f" % result2.coherent_prices["candidate_A_wins"])
    print("  candidate_B_wins: %.4f" % result2.coherent_prices["candidate_B_wins"])
    print("  Sum: %.4f" % (result2.coherent_prices["candidate_A_wins"] + result2.coherent_prices["candidate_B_wins"]))
    print("KL divergence: %.6f" % result2.kl_divergence)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demo
    run_demo()
    
    # Run tests
    print("\n\nRunning unit tests...")
    pytest.main([__file__, "-v"])
