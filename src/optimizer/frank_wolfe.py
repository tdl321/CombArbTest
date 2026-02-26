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


# =============================================================================
# Adaptive Step Size Functions
# =============================================================================


def compute_adaptive_step(
    gap: float,
    direction: NDArray,
    L_est: float,
    L_min: float = 1e-6,
) -> float:
    """Compute adaptive Frank-Wolfe step size.

    Uses the short-step rule with estimated local smoothness:
    γ = min(1, gap / (L * ||d||²))

    Args:
        gap: Duality gap (gradient^T (x - z))
        direction: Update direction (z - x)
        L_est: Estimated local Lipschitz constant
        L_min: Minimum smoothness to prevent division by zero

    Returns:
        Step size gamma in [0, 1]
    """
    d_norm_sq = np.dot(direction, direction)
    if d_norm_sq < 1e-12:
        return 0.0

    L_safe = max(L_est, L_min)
    gamma = gap / (L_safe * d_norm_sq)
    return min(1.0, max(0.0, gamma))


def estimate_smoothness(
    grad_new: NDArray,
    grad_old: NDArray,
    x_new: NDArray,
    x_old: NDArray,
) -> float:
    """Estimate local Lipschitz constant from gradient change.

    L_k = ||∇f(x_{k+1}) - ∇f(x_k)|| / ||x_{k+1} - x_k||

    Args:
        grad_new: Gradient at new point
        grad_old: Gradient at old point
        x_new: New point
        x_old: Old point

    Returns:
        Estimated local smoothness constant (inf if denominator is zero)
    """
    grad_diff = np.linalg.norm(grad_new - grad_old)
    x_diff = np.linalg.norm(x_new - x_old)
    if x_diff < 1e-12:
        return float("inf")
    return grad_diff / x_diff


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

    # Adaptive step size state
    L_est = config.initial_smoothness
    alpha = config.smoothness_alpha
    grad_old: NDArray | None = None
    mu_old: NDArray | None = None

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

        # Compute step size based on mode
        if config.step_mode == "adaptive":
            # Update smoothness estimate from previous iteration
            if grad_old is not None and mu_old is not None:
                L_k = estimate_smoothness(gradient, grad_old, mu_interior, mu_old)
                if L_k < float("inf"):
                    L_est = alpha * L_k + (1 - alpha) * L_est

            gamma = compute_adaptive_step(
                gap, direction, L_est, config.min_smoothness
            )

            # Store for next iteration
            grad_old = gradient.copy()
            mu_old = mu_interior.copy()

        elif config.step_mode == "fixed":
            gamma = config.fixed_step_size
        else:
            # Default: line search
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
    condition_space: ConditionSpace | None = None,
) -> NDArray:
    """Contract mu toward an interior point.

    Implements Barrier FW: M_eps = (1 - eps)M + eps * centroid

    Args:
        mu: Current point
        epsilon: Contraction factor
        centroid: Interior point to contract toward
        condition_space: Used to build proper per-market uniform centroid

    Returns:
        Contracted point
    """
    if centroid is None:
        if condition_space is not None:
            # Build proper per-market uniform centroid: 1/N for N outcomes
            centroid = np.zeros_like(mu)
            for market_id in condition_space.market_ids:
                indices = condition_space.get_condition_indices(market_id)
                n_outcomes = len(indices)
                for idx in indices:
                    centroid[idx] = 1.0 / n_outcomes
        else:
            # Fallback: 0.5 works for binary markets only
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
    relationships: RelationshipGraph | None = None,
) -> ArbitrageResult:
    """Convert MarginalArbitrageResult to legacy ArbitrageResult.
    
    Uses the relationship graph to generate properly typed constraint violations
    so the ArbitrageExtractor can map them to executable trades.
    """
    coherent_single = _convert_to_single_prices(result.coherent_market_prices)

    violations = []

    if relationships:
        # Generate violations from the relationship graph with correct types
        for rel in relationships.get_all_relationships():
            rel_type = rel.type.lower() if isinstance(rel.type, str) else rel.type.value

            if rel_type in ("implies", "prerequisite"):
                # Implication: from_market=YES implies to_market=YES
                # Violation: P(from) > P(to)
                if rel.from_market in market_prices_input and rel.to_market and rel.to_market in market_prices_input:
                    p_from = market_prices_input[rel.from_market]
                    p_to = market_prices_input[rel.to_market]
                    if p_from > p_to + 0.001:
                        violations.append(
                            ConstraintViolation(
                                constraint_type="implies",
                                from_market=rel.from_market,
                                to_market=rel.to_market,
                                violation_amount=p_from - p_to,
                                description=f"implies({rel.from_market}->{rel.to_market}): P(from)={p_from:.4f} > P(to)={p_to:.4f}",
                            )
                        )

            elif rel_type in ("mutually_exclusive", "incompatible", "mutex"):
                # Mutex: P(A) + P(B) <= 1
                # Violation: P(A) + P(B) > 1
                if rel.from_market in market_prices_input and rel.to_market and rel.to_market in market_prices_input:
                    p_a = market_prices_input[rel.from_market]
                    p_b = market_prices_input[rel.to_market]
                    if p_a + p_b > 1.0 + 0.001:
                        violations.append(
                            ConstraintViolation(
                                constraint_type="mutex",
                                from_market=rel.from_market,
                                to_market=rel.to_market,
                                violation_amount=p_a + p_b - 1.0,
                                description=f"mutex: P({rel.from_market})+P({rel.to_market})={p_a + p_b:.4f} > 1.0",
                            )
                        )

            elif rel_type == "equivalent":
                # Equivalent: P(A) = P(B)
                # Violation: |P(A) - P(B)| > threshold
                if rel.from_market in market_prices_input and rel.to_market and rel.to_market in market_prices_input:
                    p_a = market_prices_input[rel.from_market]
                    p_b = market_prices_input[rel.to_market]
                    diff = abs(p_a - p_b)
                    if diff > 0.001:
                        # Treat as two-directional implication violation
                        if p_a > p_b:
                            violations.append(
                                ConstraintViolation(
                                    constraint_type="implies",
                                    from_market=rel.from_market,
                                    to_market=rel.to_market,
                                    violation_amount=diff,
                                    description=f"equivalent({rel.from_market}={rel.to_market}): P(A)={p_a:.4f} > P(B)={p_b:.4f}",
                                )
                            )
                        else:
                            violations.append(
                                ConstraintViolation(
                                    constraint_type="implies",
                                    from_market=rel.to_market,
                                    to_market=rel.from_market,
                                    violation_amount=diff,
                                    description=f"equivalent({rel.to_market}={rel.from_market}): P(B)={p_b:.4f} > P(A)={p_a:.4f}",
                                )
                            )

            elif rel_type == "opposite":
                # Opposite: P(A) + P(B) = 1
                # Can violate in either direction
                if rel.from_market in market_prices_input and rel.to_market and rel.to_market in market_prices_input:
                    p_a = market_prices_input[rel.from_market]
                    p_b = market_prices_input[rel.to_market]
                    total = p_a + p_b
                    if abs(total - 1.0) > 0.001:
                        violations.append(
                            ConstraintViolation(
                                constraint_type="binary",
                                from_market=rel.from_market,
                                to_market=rel.to_market,
                                violation_amount=abs(total - 1.0),
                                description=f"opposite: P({rel.from_market})+P({rel.to_market})={total:.4f} != 1.0",
                            )
                        )

            elif rel_type == "exhaustive":
                # Exhaustive constraint handled at partition level, not pairwise
                pass

    else:
        # Fallback: generate generic price_adjustment violations (old behavior)
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

    # Convert result back, passing relationships for proper violation typing
    return _marginal_to_legacy_result(marginal_result, market_prices, relationships)


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
