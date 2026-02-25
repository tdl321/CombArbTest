"""Frank-Wolfe Solver for Arbitrage-Free Price Optimization.

OPT-01: Frank-Wolfe Algorithm
OPT-03: Barrier Frank-Wolfe Variant
OPT-05: InitFW (Interior Point)
"""

import logging
import time
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing import Callable

from .schema import (
    ArbitrageResult,
    ConstraintViolation,
    OptimizationConfig,
    RelationshipGraph,
)
from .divergence import (
    kl_divergence,
    kl_gradient,
    line_search_kl,
    compute_duality_gap,
)
from .lmo import (
    ConstraintMatrix,
    ConstraintBuilder,
    LinearMinimizationOracle,
    build_constraints_from_graph,
)

logger = logging.getLogger(__name__)


def init_fw(
    lmo: LinearMinimizationOracle,
    market_prices: NDArray[np.float64],
    num_vertices: int = 20,
) -> NDArray[np.float64]:
    """Find valid interior starting point (OPT-05)."""
    logger.debug("[FW] Finding interior starting point with %d vertices", num_vertices)

    extreme_points = lmo.get_extreme_points(num_vertices)

    if len(extreme_points) == 0:
        logger.debug("[FW] No extreme points found, using fallback")
        center = np.full(lmo.n, 0.5)
        violations = lmo.find_violated_constraints(market_prices)
        if not violations:
            return market_prices.copy()
        return center

    centroid = np.mean(extreme_points, axis=0)

    violations = lmo.find_violated_constraints(centroid)
    if violations:
        logger.debug("[FW] Centroid violates constraints, adjusting")
        center = np.full(lmo.n, 0.5)
        for alpha in [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
            test = alpha * centroid + (1 - alpha) * center
            if not lmo.find_violated_constraints(test):
                return test
        return center

    logger.debug("[FW] Interior point found")
    return centroid


def project_onto_polytope(
    x: NDArray[np.float64],
    lmo: LinearMinimizationOracle,
    max_iters: int = 100,
) -> NDArray[np.float64]:
    """Project point onto polytope using Dykstra algorithm."""
    n = len(x)

    x = np.clip(x, 0.01, 0.99)

    if not lmo.find_violated_constraints(x):
        return x

    logger.debug("[FW] Projecting point onto polytope")

    def objective(y):
        return 0.5 * np.sum((y - x) ** 2)

    def grad(y):
        return y - x

    constraints = []
    A = lmo.constraints.A
    b = lmo.constraints.b

    for i in range(A.shape[0]):
        constraints.append({
            "type": "ineq",
            "fun": lambda y, i=i: b[i] - np.dot(A[i], y),
            "jac": lambda y, i=i: -A[i],
        })

    bounds = [(0.01, 0.99) for _ in range(n)]

    result = minimize(
        objective,
        x,
        method="SLSQP",
        jac=grad,
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": max_iters}
    )

    return result.x


def _parse_violations(violations: list) -> list[ConstraintViolation]:
    """Parse violation tuples into ConstraintViolation objects."""
    result = []
    for name, amount in violations:
        parts = name.split("(")
        constraint_type = parts[0]
        from_market = ""
        to_market = None
        if len(parts) > 1:
            inner = parts[1].rstrip(")")
            if "->" in inner:
                from_market = inner.split("->")[0]
                to_market = inner.split("->")[1]
            elif "," in inner:
                from_market = inner.split(",")[0]
                to_market = inner.split(",")[1]
            else:
                from_market = inner
        result.append(ConstraintViolation(
            constraint_type=constraint_type,
            from_market=from_market,
            to_market=to_market,
            violation_amount=amount,
            description=name,
        ))
    return result


def frank_wolfe(
    market_prices: dict[str, float],
    constraints: ConstraintMatrix,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Run Frank-Wolfe to find arbitrage-free prices (OPT-01)."""
    if config is None:
        config = OptimizationConfig()

    logger.info("[FW] Starting Frank-Wolfe: %d markets, %d constraints",
                len(constraints.market_ids), constraints.A.shape[0])
    start_time = time.time()

    market_ids = constraints.market_ids
    n = len(market_ids)
    p = np.array([market_prices.get(mid, 0.5) for mid in market_ids])
    p = np.clip(p, 1e-6, 1 - 1e-6)

    lmo = LinearMinimizationOracle(constraints)
    q = project_onto_polytope(p.copy(), lmo)

    converged = False
    final_gap = float("inf")

    for t in range(config.max_iterations):
        g = kl_gradient(p, q)
        s, _ = lmo.solve(g)
        gap = compute_duality_gap(p, q, s)
        final_gap = gap

        if config.verbose and t % 100 == 0:
            kl = kl_divergence(p, q)
            logger.debug("[FW] Iter %d: KL=%.6f, gap=%.6f", t, kl, gap)

        if gap < config.tolerance:
            converged = True
            logger.debug("[FW] Converged at iteration %d with gap=%.2e", t, gap)
            break

        direction = s - q
        if config.line_search:
            gamma = line_search_kl(p, q, direction, max_step=1.0)
        else:
            gamma = 2.0 / (t + 2)

        q = q + gamma * direction
        q = np.clip(q, 1e-10, 1 - 1e-10)

    final_kl = kl_divergence(p, q)
    elapsed = time.time() - start_time

    violations = lmo.find_violated_constraints(p)
    constraint_violations = _parse_violations(violations)

    coherent_prices = {mid: float(q[i]) for i, mid in enumerate(market_ids)}
    original_prices = {mid: float(p[i]) for i, mid in enumerate(market_ids)}

    logger.info("[FW] Complete: KL=%.6f, gap=%.2e, iters=%d, violations=%d, time=%.3fs",
                final_kl, final_gap, t + 1, len(violations), elapsed)

    return ArbitrageResult(
        market_prices=original_prices,
        coherent_prices=coherent_prices,
        kl_divergence=final_kl,
        constraints_violated=constraint_violations,
        converged=converged,
        iterations=t + 1,
        final_gap=final_gap,
    )


def barrier_frank_wolfe(
    market_prices: dict[str, float],
    constraints: ConstraintMatrix,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Run Barrier Frank-Wolfe for numerical stability (OPT-03)."""
    if config is None:
        config = OptimizationConfig()

    logger.info("[FW] Starting Barrier Frank-Wolfe: %d markets, barrier=%.3f",
                len(constraints.market_ids), config.initial_barrier)
    start_time = time.time()

    market_ids = constraints.market_ids
    n = len(market_ids)
    p = np.array([market_prices.get(mid, 0.5) for mid in market_ids])
    p = np.clip(p, 1e-6, 1 - 1e-6)

    lmo = LinearMinimizationOracle(constraints)
    center = init_fw(lmo, p)
    q = project_onto_polytope(p.copy(), lmo)

    epsilon = config.initial_barrier
    converged = False
    final_gap = float("inf")
    total_iters = 0

    while epsilon >= config.min_barrier:
        logger.debug("[FW] Barrier phase: epsilon=%.4f", epsilon)

        for t in range(config.max_iterations // 10):
            total_iters += 1

            g = kl_gradient(p, q)
            s, _ = lmo.solve(g, epsilon=epsilon, center=center)
            gap = compute_duality_gap(p, q, s)
            final_gap = gap

            if gap < config.tolerance:
                converged = True
                break

            direction = s - q
            if config.line_search:
                gamma = line_search_kl(p, q, direction, max_step=1.0)
            else:
                gamma = 2.0 / (t + 2)

            q = q + gamma * direction
            q = np.clip(q, 1e-10, 1 - 1e-10)

        if converged:
            break

        epsilon *= config.barrier_decay

        if config.verbose:
            kl = kl_divergence(p, q)
            logger.debug("[FW] Barrier eps=%.4f: KL=%.6f, gap=%.6f", epsilon, kl, final_gap)

    logger.debug("[FW] Final polishing phase")
    for t in range(min(100, config.max_iterations)):
        total_iters += 1

        g = kl_gradient(p, q)
        s, _ = lmo.solve(g, epsilon=0)
        gap = compute_duality_gap(p, q, s)
        final_gap = gap

        if gap < config.tolerance:
            converged = True
            break

        direction = s - q
        if config.line_search:
            gamma = line_search_kl(p, q, direction, max_step=1.0)
        else:
            gamma = 2.0 / (t + 2)

        q = q + gamma * direction
        q = np.clip(q, 1e-10, 1 - 1e-10)

    final_kl = kl_divergence(p, q)
    elapsed = time.time() - start_time

    violations = lmo.find_violated_constraints(p)
    constraint_violations = _parse_violations(violations)

    coherent_prices = {mid: float(q[i]) for i, mid in enumerate(market_ids)}
    original_prices = {mid: float(p[i]) for i, mid in enumerate(market_ids)}

    logger.info("[FW] Barrier complete: KL=%.6f, gap=%.2e, iters=%d, violations=%d, time=%.3fs",
                final_kl, final_gap, total_iters, len(violations), elapsed)

    return ArbitrageResult(
        market_prices=original_prices,
        coherent_prices=coherent_prices,
        kl_divergence=final_kl,
        constraints_violated=constraint_violations,
        converged=converged,
        iterations=total_iters,
        final_gap=final_gap,
    )


def projected_gradient_descent(
    market_prices: dict[str, float],
    constraints: ConstraintMatrix,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Alternative solver using projected gradient descent."""
    if config is None:
        config = OptimizationConfig()

    logger.info("[FW] Starting PGD: %d markets", len(constraints.market_ids))
    start_time = time.time()

    market_ids = constraints.market_ids
    n = len(market_ids)
    p = np.array([market_prices.get(mid, 0.5) for mid in market_ids])
    p = np.clip(p, 1e-6, 1 - 1e-6)

    lmo = LinearMinimizationOracle(constraints)

    def objective(q):
        return kl_divergence(p, q)

    def gradient(q):
        return kl_gradient(p, q)

    scipy_constraints = []
    A = constraints.A
    b = constraints.b

    for i in range(A.shape[0]):
        scipy_constraints.append({
            "type": "ineq",
            "fun": lambda q, i=i: b[i] - np.dot(A[i], q),
            "jac": lambda q, i=i: -A[i],
        })

    bounds = [(0.01, 0.99) for _ in range(n)]
    q0 = project_onto_polytope(p.copy(), lmo)

    result = minimize(
        objective,
        q0,
        method="SLSQP",
        jac=gradient,
        constraints=scipy_constraints,
        bounds=bounds,
        options={"maxiter": config.max_iterations, "ftol": config.tolerance}
    )

    q = result.x
    final_kl = kl_divergence(p, q)
    elapsed = time.time() - start_time

    violations = lmo.find_violated_constraints(p)
    constraint_violations = _parse_violations(violations)

    coherent_prices = {mid: float(q[i]) for i, mid in enumerate(market_ids)}
    original_prices = {mid: float(p[i]) for i, mid in enumerate(market_ids)}

    logger.info("[FW] PGD complete: KL=%.6f, iters=%d, success=%s, time=%.3fs",
                final_kl, result.nit, result.success, elapsed)

    return ArbitrageResult(
        market_prices=original_prices,
        coherent_prices=coherent_prices,
        kl_divergence=final_kl,
        constraints_violated=constraint_violations,
        converged=result.success,
        iterations=result.nit,
        final_gap=0.0,
    )


def find_arbitrage(
    market_prices: dict[str, float],
    relationships: RelationshipGraph,
    config: OptimizationConfig | None = None,
    use_barrier: bool = True,
) -> ArbitrageResult:
    """High-level API: Find arbitrage-free prices from market relationships."""
    logger.info("[OPT] Finding arbitrage: %d prices, barrier=%s", len(market_prices), use_barrier)

    constraints = build_constraints_from_graph(relationships)

    if use_barrier:
        return barrier_frank_wolfe(market_prices, constraints, config)
    else:
        return frank_wolfe(market_prices, constraints, config)


def find_arbitrage_simple(
    market_prices: dict[str, float],
    constraints: ConstraintMatrix,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Simplified API using pre-built constraints."""
    logger.debug("[OPT] find_arbitrage_simple called")
    return barrier_frank_wolfe(market_prices, constraints, config)
