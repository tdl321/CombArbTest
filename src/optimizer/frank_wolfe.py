"""Frank-Wolfe algorithm for marginal polytope optimization.

Implements the Barrier Frank-Wolfe variant for finding coherent prices
within the marginal polytope that are closest to market prices (in KL divergence).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .divergence import (
    build_theta_from_prices,
    categorical_kl,
    categorical_kl_gradient,
    line_search_exact,
    mu_to_market_prices,
)
from .lmo import (
    MarginalConstraintBuilder,
    MarginalConstraintMatrix,
    MarginalPolytopeLMO,
    build_constraints_from_graph,
)
from .schema import (
    ConditionSpace,
    MarginalArbitrageResult,
    OptimizationConfig,
    RelationshipGraph,
)


def marginal_frank_wolfe(
    market_prices: dict[str, list[float]],
    constraints: MarginalConstraintMatrix,
    config: OptimizationConfig | None = None,
) -> MarginalArbitrageResult:
    """Frank-Wolfe optimization over the marginal polytope.

    Finds the coherent price vector mu in the marginal polytope that minimizes
    KL(theta || mu), where theta represents market prices.

    Args:
        market_prices: Dict of market_id -> [p_yes, p_no, ...]
        constraints: Constraint matrices from MarginalConstraintBuilder
        config: Optimization configuration

    Returns:
        MarginalArbitrageResult with coherent prices and arbitrage metrics
    """
    if config is None:
        config = OptimizationConfig()

    space = constraints.condition_space

    # 1. Build theta vector (market prices in condition space)
    theta = build_theta_from_prices(market_prices, space)

    # 2. Initialize LMO
    lmo = MarginalPolytopeLMO(constraints)

    # 3. Initialize mu from centroid of vertices (guarantees interior point)
    mu = lmo.compute_centroid(n_samples=20)

    # Ensure mu is strictly interior (Barrier FW)
    epsilon = config.epsilon_init
    mu_interior = _contract_toward_centroid(mu, epsilon)

    # 4. Frank-Wolfe loop
    converged = False
    iterations = 0
    gap = float("inf")
    active_vertices: list[NDArray] = []

    for t in range(config.max_iterations):
        iterations = t + 1

        # Gradient of KL(theta || mu) w.r.t. mu
        gradient = categorical_kl_gradient(theta, mu_interior, space)

        # LMO: find vertex minimizing gradient^T z
        z_new, _ = lmo.solve(gradient)
        active_vertices.append(z_new)

        # Duality gap: gradient^T (mu - z_new)
        gap = gradient @ (mu_interior - z_new)

        if gap < config.tolerance:
            converged = True
            break

        # Direction: vertex - current
        direction = z_new - mu_interior

        # Line search for optimal step size
        gamma = line_search_exact(
            theta, mu_interior, direction, space, max_gamma=1.0
        )

        # Update
        mu_interior = mu_interior + gamma * direction

        # Decay epsilon (allow closer to boundary as we converge)
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        # Re-contract toward interior (Barrier FW)
        mu_interior = _contract_toward_centroid(mu_interior, epsilon, mu)

    # 5. Compute final KL divergence
    kl_divergence = categorical_kl(theta, mu_interior, space)

    # 6. Build result
    coherent_market_prices = mu_to_market_prices(mu_interior, space)

    # Build condition prices
    condition_prices = {}
    coherent_condition_prices = {}
    for i, cond in enumerate(space.conditions):
        condition_prices[cond.condition_id] = theta[i]
        coherent_condition_prices[cond.condition_id] = mu_interior[i]

    return MarginalArbitrageResult(
        condition_prices=condition_prices,
        coherent_condition_prices=coherent_condition_prices,
        market_prices=market_prices,
        coherent_market_prices=coherent_market_prices,
        kl_divergence=kl_divergence,
        duality_gap=gap,
        converged=converged,
        iterations=iterations,
        active_vertices=[v.astype(int).tolist() for v in active_vertices[-10:]],
    )


def _contract_toward_centroid(
    mu: NDArray,
    epsilon: float,
    centroid: NDArray | None = None,
) -> NDArray:
    """Contract mu toward an interior point.

    Implements Barrier FW: M_eps = (1 - eps)M + eps * centroid

    Args:
        mu: Current point
        epsilon: Contraction factor
        centroid: Interior point to contract toward (defaults to 0.5)

    Returns:
        Contracted point
    """
    if centroid is None:
        centroid = np.ones_like(mu) * 0.5

    return (1 - epsilon) * mu + epsilon * centroid


def find_marginal_arbitrage(
    market_prices: dict[str, list[float]],
    relationships: RelationshipGraph,
    market_outcomes: dict[str, list[str]] | None = None,
    config: OptimizationConfig | None = None,
) -> MarginalArbitrageResult:
    """High-level API for combinatorial arbitrage detection.

    This is the main entry point for detecting arbitrage opportunities
    across related prediction markets.

    Args:
        market_prices: Dict of market_id -> [p_yes, p_no, ...]
        relationships: Graph of logical relationships between markets
        market_outcomes: Optional outcome names (defaults to YES/NO)
        config: Optimization configuration

    Returns:
        MarginalArbitrageResult with arbitrage detection results

    Example:
        >>> # Market B implies Market A (if B=YES, then A=YES)
        >>> graph = RelationshipGraph(clusters=[
        ...     MarketCluster(
        ...         cluster_id='test',
        ...         market_ids=['A', 'B'],
        ...         relationships=[
        ...             MarketRelationship(type='implies', from_market='B', to_market='A')
        ...         ]
        ...     )
        ... ])
        >>> # Prices violate implication: B_yes=0.4 but A_yes=0.3
        >>> result = find_marginal_arbitrage(
        ...     market_prices={'A': [0.3, 0.7], 'B': [0.4, 0.6]},
        ...     relationships=graph
        ... )
        >>> print(f"KL divergence: {result.kl_divergence:.4f}")
        >>> print(f"Has arbitrage: {result.has_arbitrage()}")
    """
    # Extract market IDs from prices
    market_ids = list(market_prices.keys())

    # Build constraints from relationship graph
    constraints = build_constraints_from_graph(
        market_ids=market_ids,
        relationships=relationships,
        market_outcomes=market_outcomes,
    )

    # Run Frank-Wolfe optimization
    return marginal_frank_wolfe(market_prices, constraints, config)


def detect_arbitrage_simple(
    market_prices: dict[str, list[float]],
    implications: list[tuple[str, str]] | None = None,
    mutex_pairs: list[tuple[str, str]] | None = None,
    config: OptimizationConfig | None = None,
) -> MarginalArbitrageResult:
    """Simplified API for common constraint patterns.

    Args:
        market_prices: Dict of market_id -> [p_yes, p_no]
        implications: List of (from_market, to_market) for B->A constraints
        mutex_pairs: List of (market_a, market_b) for mutex constraints
        config: Optimization configuration

    Returns:
        MarginalArbitrageResult

    Example:
        >>> # B implies A: if B=YES then A must be YES
        >>> result = detect_arbitrage_simple(
        ...     market_prices={'A': [0.3, 0.7], 'B': [0.4, 0.6]},
        ...     implications=[('B', 'A')]  # B->A
        ... )
    """
    market_ids = list(market_prices.keys())

    # Build condition space
    space = ConditionSpace.from_market_data(market_ids)

    # Build constraints
    builder = MarginalConstraintBuilder(space)

    if implications:
        for from_m, to_m in implications:
            builder.add_implies(from_m, to_m)

    if mutex_pairs:
        for m_a, m_b in mutex_pairs:
            builder.add_mutually_exclusive(m_a, m_b)

    constraints = builder.build()

    return marginal_frank_wolfe(market_prices, constraints, config)


# =============================================================================
# Backward Compatibility Functions (for existing backtest/simulator code)
# =============================================================================

from .schema import ArbitrageResult, ConstraintViolation
from .divergence import kl_divergence, kl_gradient, line_search_kl
import logging

logger = logging.getLogger(__name__)


def _convert_to_list_prices(
    market_prices: dict[str, float],
) -> dict[str, list[float]]:
    """Convert dict[str, float] to dict[str, list[float]]."""
    return {mid: [p, 1.0 - p] for mid, p in market_prices.items()}


def _convert_to_single_prices(
    market_prices: dict[str, list[float]],
) -> dict[str, float]:
    """Convert dict[str, list[float]] to dict[str, float] (YES prob only)."""
    return {mid: prices[0] for mid, prices in market_prices.items()}


def _marginal_to_legacy_result(
    result: MarginalArbitrageResult,
    market_prices_input: dict[str, float],
) -> ArbitrageResult:
    """Convert MarginalArbitrageResult to legacy ArbitrageResult."""
    coherent_single = _convert_to_single_prices(result.coherent_market_prices)

    # Check for constraint violations
    violations = []
    for mid in market_prices_input:
        if mid in coherent_single:
            diff = abs(coherent_single[mid] - market_prices_input[mid])
            if diff > 0.01:
                violations.append(
                    ConstraintViolation(
                        constraint_type="price_adjustment",
                        from_market=mid,
                        to_market=None,
                        violation_amount=diff,
                        description=f"Price adjusted by {diff:.4f}",
                    )
                )

    return ArbitrageResult(
        market_prices=market_prices_input,
        coherent_prices=coherent_single,
        kl_divergence=result.kl_divergence,
        constraints_violated=violations,
        converged=result.converged,
        iterations=result.iterations,
        final_gap=result.duality_gap,
    )


def find_arbitrage(
    market_prices: dict[str, float],
    relationships: RelationshipGraph,
    config: OptimizationConfig | None = None,
    use_barrier: bool = True,
) -> ArbitrageResult:
    """High-level API: Find arbitrage-free prices (backward-compatible).

    Args:
        market_prices: Dict of market_id -> YES probability
        relationships: RelationshipGraph with constraints
        config: Optimization configuration
        use_barrier: Whether to use barrier method (always True now)

    Returns:
        ArbitrageResult with coherent prices
    """
    logger.info(
        "[FW] find_arbitrage called with %d markets, use_barrier=%s",
        len(market_prices),
        use_barrier,
    )

    # Convert to new format
    list_prices = _convert_to_list_prices(market_prices)

    # Run new optimization
    marginal_result = find_marginal_arbitrage(
        market_prices=list_prices,
        relationships=relationships,
        config=config,
    )

    # Convert result back
    return _marginal_to_legacy_result(marginal_result, market_prices)


def find_arbitrage_simple(
    market_prices: dict[str, float],
    constraints,  # ConstraintMatrix (ignored, rebuilt from scratch)
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Simplified API using pre-built constraints (backward-compatible).

    Note: The constraints parameter is ignored; constraints are rebuilt
    from the market structure.
    """
    logger.debug("[FW] find_arbitrage_simple called")

    # Convert and run without constraints (just market structure)
    list_prices = _convert_to_list_prices(market_prices)
    market_ids = list(market_prices.keys())

    space = ConditionSpace.from_market_data(market_ids)
    builder = MarginalConstraintBuilder(space)
    new_constraints = builder.build()

    marginal_result = marginal_frank_wolfe(list_prices, new_constraints, config)
    return _marginal_to_legacy_result(marginal_result, market_prices)


def barrier_frank_wolfe(
    market_prices: dict[str, float],
    constraints,  # ConstraintMatrix
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Barrier Frank-Wolfe algorithm (backward-compatible wrapper)."""
    return find_arbitrage_simple(market_prices, constraints, config)


def frank_wolfe(
    market_prices: dict[str, float],
    constraints,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Standard Frank-Wolfe (maps to barrier variant)."""
    return barrier_frank_wolfe(market_prices, constraints, config)


def init_fw(
    market_ids: list[str],
    constraints,
    n_samples: int = 10,
) -> dict[str, float]:
    """Initialize Frank-Wolfe by finding interior point.

    Returns starting point as dict[str, float].
    """
    # Just return uniform 0.5 for each market
    return {mid: 0.5 for mid in market_ids}


def projected_gradient_descent(
    market_prices: dict[str, float],
    constraints,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Projected gradient descent (maps to Frank-Wolfe)."""
    return frank_wolfe(market_prices, constraints, config)
