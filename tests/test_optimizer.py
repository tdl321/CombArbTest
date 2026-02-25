"""Tests for marginal polytope arbitrage optimizer.

Tests the condition space model, LMO, Frank-Wolfe optimization,
and arbitrage detection.
"""

import numpy as np
import pytest

from src.optimizer import (
    ConditionSpace,
    MarginalConstraintBuilder,
    MarginalPolytopeLMO,
    MarketCluster,
    MarketRelationship,
    RelationshipGraph,
    RelationshipType,
    build_constraints_from_graph,
    build_theta_from_prices,
    categorical_kl,
    detect_arbitrage_simple,
    find_marginal_arbitrage,
    marginal_frank_wolfe,
)


class TestConditionSpace:
    """Tests for ConditionSpace data structure."""

    def test_from_market_data_default_outcomes(self):
        """Two markets with default YES/NO outcomes should have 4 conditions."""
        space = ConditionSpace.from_market_data(["A", "B"])

        assert space.n_conditions() == 4
        assert space.n_markets() == 2
        assert space.market_ids == ["A", "B"]

    def test_from_market_data_custom_outcomes(self):
        """Custom outcomes should work."""
        space = ConditionSpace.from_market_data(
            ["winner"],
            market_outcomes={"winner": ["Alice", "Bob", "Charlie"]},
        )

        assert space.n_conditions() == 3
        assert space.n_markets() == 1

    def test_get_condition_indices(self):
        """Should return correct indices for each market."""
        space = ConditionSpace.from_market_data(["A", "B"])

        a_indices = space.get_condition_indices("A")
        b_indices = space.get_condition_indices("B")

        assert len(a_indices) == 2
        assert len(b_indices) == 2
        assert set(a_indices).isdisjoint(set(b_indices))

    def test_get_yes_no_index(self):
        """Should return YES as index 0, NO as index 1."""
        space = ConditionSpace.from_market_data(["A"])

        yes_idx = space.get_yes_index("A")
        no_idx = space.get_no_index("A")

        assert yes_idx == 0
        assert no_idx == 1


class TestMarginalConstraintBuilder:
    """Tests for constraint building."""

    def test_exactly_one_constraints(self):
        """Each market should have exactly-one constraint."""
        space = ConditionSpace.from_market_data(["A", "B"])
        builder = MarginalConstraintBuilder(space)
        constraints = builder.build()

        # Should have 2 equality constraints (one per market)
        assert constraints.A_eq.shape[0] == 2
        assert np.allclose(constraints.b_eq, [1.0, 1.0])

    def test_implies_constraint(self):
        """B->A implication should add inequality constraint."""
        space = ConditionSpace.from_market_data(["A", "B"])
        builder = MarginalConstraintBuilder(space)
        builder.add_implies("B", "A")
        constraints = builder.build()

        # Should have 1 inequality constraint
        assert constraints.A_ub.shape[0] == 1

        # Constraint: z_B_yes - z_A_yes <= 0
        b_yes = space.get_yes_index("B")
        a_yes = space.get_yes_index("A")
        row = constraints.A_ub[0]
        assert row[b_yes] == 1.0
        assert row[a_yes] == -1.0

    def test_mutually_exclusive_constraint(self):
        """Mutex constraint should limit sum to 1."""
        space = ConditionSpace.from_market_data(["A", "B"])
        builder = MarginalConstraintBuilder(space)
        builder.add_mutually_exclusive("A", "B")
        constraints = builder.build()

        # Should have 1 inequality constraint
        assert constraints.A_ub.shape[0] == 1
        assert constraints.b_ub[0] == 1.0


class TestMarginalPolytopeLMO:
    """Tests for the Linear Minimization Oracle."""

    def test_two_independent_markets_4_vertices(self):
        """Two independent binary markets should have 4 vertices."""
        space = ConditionSpace.from_market_data(["A", "B"])
        builder = MarginalConstraintBuilder(space)
        constraints = builder.build()
        lmo = MarginalPolytopeLMO(constraints)

        vertices = lmo.enumerate_vertices(max_vertices=10)

        # Should find exactly 4 vertices: (0,1,0,1), (0,1,1,0), (1,0,0,1), (1,0,1,0)
        assert len(vertices) == 4

        # Each vertex should be binary
        for v in vertices:
            assert np.all((v == 0) | (v == 1))

        # Each vertex should satisfy exactly-one constraint
        for v in vertices:
            for market_id in space.market_ids:
                indices = space.get_condition_indices(market_id)
                assert sum(v[i] for i in indices) == 1

    def test_implication_reduces_vertices(self):
        """B->A implication should reduce vertices from 4 to 3."""
        space = ConditionSpace.from_market_data(["A", "B"])
        builder = MarginalConstraintBuilder(space)
        builder.add_implies("B", "A")
        constraints = builder.build()
        lmo = MarginalPolytopeLMO(constraints)

        vertices = lmo.enumerate_vertices(max_vertices=10)

        # Should have only 3 valid vertices:
        # (A=YES, B=YES), (A=YES, B=NO), (A=NO, B=NO)
        # NOT (A=NO, B=YES) because B->A
        assert len(vertices) == 3

        # Verify no vertex has A=NO and B=YES
        a_yes = space.get_yes_index("A")
        b_yes = space.get_yes_index("B")

        for v in vertices:
            # If B=YES, then A must be YES
            if v[b_yes] == 1:
                assert v[a_yes] == 1, "Implication violated: B=YES but A=NO"

    def test_lmo_solve_minimizes_gradient(self):
        """LMO should return vertex minimizing gradient."""
        space = ConditionSpace.from_market_data(["A"])
        builder = MarginalConstraintBuilder(space)
        constraints = builder.build()
        lmo = MarginalPolytopeLMO(constraints)

        # Gradient favoring YES (negative at YES index)
        gradient = np.array([-1.0, 1.0])
        z, obj = lmo.solve(gradient)

        # Should return (1, 0) - YES outcome
        assert z[0] == 1.0 and z[1] == 0.0


class TestCategoricalKL:
    """Tests for KL divergence functions."""

    def test_kl_same_distribution_is_zero(self):
        """KL(p || p) should be 0."""
        space = ConditionSpace.from_market_data(["A"])
        theta = np.array([0.6, 0.4])
        mu = np.array([0.6, 0.4])

        kl = categorical_kl(theta, mu, space)
        assert kl < 1e-10

    def test_kl_different_distributions(self):
        """KL(p || q) should be positive for p != q."""
        space = ConditionSpace.from_market_data(["A"])
        theta = np.array([0.8, 0.2])
        mu = np.array([0.5, 0.5])

        kl = categorical_kl(theta, mu, space)
        assert kl > 0

    def test_build_theta_from_prices(self):
        """Should correctly convert market prices to theta vector."""
        space = ConditionSpace.from_market_data(["A", "B"])
        market_prices = {"A": [0.3, 0.7], "B": [0.6, 0.4]}

        theta = build_theta_from_prices(market_prices, space)

        # Verify values
        a_indices = space.get_condition_indices("A")
        b_indices = space.get_condition_indices("B")

        assert np.isclose(theta[a_indices[0]], 0.3)
        assert np.isclose(theta[a_indices[1]], 0.7)
        assert np.isclose(theta[b_indices[0]], 0.6)
        assert np.isclose(theta[b_indices[1]], 0.4)


class TestMarginalFrankWolfe:
    """Tests for the Frank-Wolfe optimization."""

    def test_coherent_prices_remain_unchanged(self):
        """Prices already satisfying constraints should have KL ~= 0."""
        # Independent markets - any prices are coherent
        result = detect_arbitrage_simple(
            market_prices={"A": [0.5, 0.5], "B": [0.5, 0.5]},
        )

        assert result.kl_divergence < 0.01
        assert result.converged

    def test_implication_violation_detected(self):
        """Violating B->A (B_yes > A_yes) should have KL > 0."""
        # B implies A, but B_yes=0.4 > A_yes=0.3 is impossible
        result = detect_arbitrage_simple(
            market_prices={"A": [0.3, 0.7], "B": [0.4, 0.6]},
            implications=[("B", "A")],
        )

        assert result.kl_divergence > 0.01, f"Expected KL > 0.01, got {result.kl_divergence}"
        assert result.has_arbitrage()

        # Coherent prices should satisfy implication: A_yes >= B_yes
        a_yes = result.coherent_market_prices["A"][0]
        b_yes = result.coherent_market_prices["B"][0]
        assert a_yes >= b_yes - 0.01, f"Implication violated: A_yes={a_yes} < B_yes={b_yes}"

    def test_implication_satisfied_no_arbitrage(self):
        """Prices satisfying B->A should have KL ~= 0."""
        # B implies A, and A_yes=0.6 > B_yes=0.4 - no violation
        result = detect_arbitrage_simple(
            market_prices={"A": [0.6, 0.4], "B": [0.4, 0.6]},
            implications=[("B", "A")],
        )

        # Should have very low KL (prices are mostly coherent)
        assert result.kl_divergence < 0.1

    def test_mutex_violation_detected(self):
        """Mutex markets with high YES probabilities should show arbitrage."""
        # Both markets have YES probability = 0.6, but they're mutex
        # This is impossible - at most one can be YES
        result = detect_arbitrage_simple(
            market_prices={"A": [0.6, 0.4], "B": [0.6, 0.4]},
            mutex_pairs=[("A", "B")],
        )

        assert result.kl_divergence > 0.01


class TestFindMarginalArbitrage:
    """Tests for the high-level API."""

    def test_with_relationship_graph(self):
        """Should work with RelationshipGraph input."""
        graph = RelationshipGraph(
            clusters=[
                MarketCluster(
                    cluster_id="test",
                    market_ids=["A", "B"],
                    relationships=[
                        MarketRelationship(
                            type=RelationshipType.IMPLIES,
                            from_market="B",
                            to_market="A",
                        )
                    ],
                )
            ]
        )

        # Violating prices
        result = find_marginal_arbitrage(
            market_prices={"A": [0.3, 0.7], "B": [0.4, 0.6]},
            relationships=graph,
        )

        assert result.has_arbitrage()

    def test_pennsylvania_example(self):
        """Test the Pennsylvania election example from the plan.

        Market A: "Trump wins PA" (YES/NO)
        Market B: "Rep wins PA by 5+" (YES/NO)
        Constraint: B=YES implies A=YES

        If prices are coherent (B_yes < A_yes), KL should be low.
        """
        graph = RelationshipGraph(
            clusters=[
                MarketCluster(
                    cluster_id="pa_election",
                    market_ids=["trump_wins_pa", "rep_wins_5plus"],
                    relationships=[
                        MarketRelationship(
                            type=RelationshipType.IMPLIES,
                            from_market="rep_wins_5plus",
                            to_market="trump_wins_pa",
                        )
                    ],
                )
            ]
        )

        # Coherent prices: Trump=48%, Rep5+=32% (32% < 48%, satisfies implication)
        result = find_marginal_arbitrage(
            market_prices={
                "trump_wins_pa": [0.48, 0.52],
                "rep_wins_5plus": [0.32, 0.68],
            },
            relationships=graph,
        )

        # Should have low KL (prices are coherent)
        assert result.kl_divergence < 0.1, f"Expected coherent prices, got KL={result.kl_divergence}"

        # Incoherent prices: Rep5+=60% but Trump=40% (violates implication)
        result_incoherent = find_marginal_arbitrage(
            market_prices={
                "trump_wins_pa": [0.40, 0.60],
                "rep_wins_5plus": [0.60, 0.40],
            },
            relationships=graph,
        )

        # Should have high KL (arbitrage opportunity)
        assert result_incoherent.kl_divergence > 0.01
        assert result_incoherent.has_arbitrage()


class TestVertexEnumeration:
    """Tests for vertex enumeration correctness."""

    def test_three_way_implication_chain(self):
        """C->B->A should have correct vertex count."""
        # If C=YES then B=YES then A=YES
        # Valid: (A,B,C) = (Y,Y,Y), (Y,Y,N), (Y,N,N), (N,N,N)
        # Invalid: Any with C=Y but B=N, or B=Y but A=N

        space = ConditionSpace.from_market_data(["A", "B", "C"])
        builder = MarginalConstraintBuilder(space)
        builder.add_implies("C", "B")
        builder.add_implies("B", "A")
        constraints = builder.build()
        lmo = MarginalPolytopeLMO(constraints)

        vertices = lmo.enumerate_vertices(max_vertices=20)

        # Should have exactly 4 valid vertices
        assert len(vertices) == 4

        # Verify constraints on each vertex
        a_yes = space.get_yes_index("A")
        b_yes = space.get_yes_index("B")
        c_yes = space.get_yes_index("C")

        for v in vertices:
            # C->B: if C=YES then B=YES
            if v[c_yes] == 1:
                assert v[b_yes] == 1
            # B->A: if B=YES then A=YES
            if v[b_yes] == 1:
                assert v[a_yes] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
