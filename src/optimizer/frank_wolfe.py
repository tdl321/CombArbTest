"""Frank-Wolfe Solver for Arbitrage-Free Price Optimization.

OPT-01: Frank-Wolfe Algorithm
OPT-03: Barrier Frank-Wolfe Variant  
OPT-05: InitFW (Interior Point)

The Frank-Wolfe algorithm finds arbitrage-free prices by minimizing
KL divergence subject to the marginal polytope constraints.
"""

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


def init_fw(
    lmo: LinearMinimizationOracle,
    market_prices: NDArray[np.float64],
    num_vertices: int = 20,
) -> NDArray[np.float64]:
    """Find valid interior starting point (OPT-05).
    
    Finds the centroid of extreme points of the polytope, which is
    guaranteed to be in the relative interior. This ensures the
    Frank-Wolfe algorithm starts from a valid point.
    
    Args:
        lmo: Linear Minimization Oracle
        market_prices: Initial market prices (used as fallback)
        num_vertices: Number of vertices to sample
        
    Returns:
        Interior starting point
    """
    # Get extreme points
    extreme_points = lmo.get_extreme_points(num_vertices)
    
    if len(extreme_points) == 0:
        # Fallback: project market prices onto polytope
        # Start from center of [0, 1]^n and solve
        center = np.full(lmo.n, 0.5)
        violations = lmo.find_violated_constraints(market_prices)
        if not violations:
            return market_prices.copy()
        return center
    
    # Centroid of vertices (guaranteed in relative interior)
    centroid = np.mean(extreme_points, axis=0)
    
    # Ensure centroid satisfies constraints
    violations = lmo.find_violated_constraints(centroid)
    if violations:
        # If somehow violated, take convex combination with center
        center = np.full(lmo.n, 0.5)
        for alpha in [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
            test = alpha * centroid + (1 - alpha) * center
            if not lmo.find_violated_constraints(test):
                return test
        return center
    
    return centroid


def project_onto_polytope(
    x: NDArray[np.float64],
    lmo: LinearMinimizationOracle,
    max_iters: int = 100,
) -> NDArray[np.float64]:
    """Project point onto polytope using Dykstra's algorithm.
    
    This is useful for initializing from market prices.
    """
    n = len(x)
    
    # Simple box projection first
    x = np.clip(x, 0.01, 0.99)
    
    # If satisfies all constraints, return
    if not lmo.find_violated_constraints(x):
        return x
    
    # Use scipy to project
    from scipy.optimize import minimize
    
    def objective(y):
        return 0.5 * np.sum((y - x) ** 2)
    
    def grad(y):
        return y - x
    
    # Build linear constraints for scipy
    constraints = []
    A = lmo.constraints.A
    b = lmo.constraints.b
    
    for i in range(A.shape[0]):
        constraints.append({
            'type': 'ineq',
            'fun': lambda y, i=i: b[i] - np.dot(A[i], y),
            'jac': lambda y, i=i: -A[i],
        })
    
    bounds = [(0.01, 0.99) for _ in range(n)]
    
    result = minimize(
        objective,
        x,
        method='SLSQP',
        jac=grad,
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': max_iters}
    )
    
    return result.x


def frank_wolfe(
    market_prices: dict[str, float],
    constraints: ConstraintMatrix,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Run Frank-Wolfe to find arbitrage-free prices (OPT-01).
    
    Minimizes KL(market_prices || coherent_prices) subject to the
    marginal polytope constraints defined by market relationships.
    
    Frank-Wolfe iteration:
    1. g = gradient of KL at current point
    2. s = LMO(g) = argmin_{s in polytope} <g, s>
    3. gamma = step size (line search or 2/(t+2))
    4. x_{t+1} = (1-gamma) * x_t + gamma * s
    
    Args:
        market_prices: Dict of market_id -> price
        constraints: Constraint matrix from LMO
        config: Optimization configuration
        
    Returns:
        ArbitrageResult with coherent prices
    """
    if config is None:
        config = OptimizationConfig()
    
    # Map prices to vector
    market_ids = constraints.market_ids
    n = len(market_ids)
    p = np.array([market_prices.get(mid, 0.5) for mid in market_ids])
    
    # Clip to valid probability range
    p = np.clip(p, 1e-6, 1 - 1e-6)
    
    # Initialize LMO
    lmo = LinearMinimizationOracle(constraints)
    
    # Try to start from projected market prices for faster convergence
    q = project_onto_polytope(p.copy(), lmo)
    
    # Track convergence
    converged = False
    final_gap = float("inf")
    
    for t in range(config.max_iterations):
        # Step 1: Compute gradient
        g = kl_gradient(p, q)
        
        # Step 2: LMO - find minimizer over polytope
        s, _ = lmo.solve(g)
        
        # Compute duality gap
        gap = compute_duality_gap(p, q, s)
        final_gap = gap
        
        if config.verbose and t % 100 == 0:
            kl = kl_divergence(p, q)
            print("Iter %d: KL=%.6f, gap=%.6f" % (t, kl, gap))
        
        # Check convergence
        if gap < config.tolerance:
            converged = True
            break
        
        # Step 3: Step size
        direction = s - q
        if config.line_search:
            gamma = line_search_kl(p, q, direction, max_step=1.0)
        else:
            gamma = 2.0 / (t + 2)
        
        # Step 4: Update
        q = q + gamma * direction
        
        # Ensure we stay in [0, 1]
        q = np.clip(q, 1e-10, 1 - 1e-10)
    
    # Compute final metrics
    final_kl = kl_divergence(p, q)
    
    # Find violated constraints in original prices
    violations = lmo.find_violated_constraints(p)
    constraint_violations = [
        ConstraintViolation(
            constraint_type=name.split("(")[0],
            from_market=name.split("(")[1].rstrip(")").split("->")[0].split(",")[0] if "(" in name else "",
            to_market=name.split("->")[1].rstrip(")") if "->" in name else (
                name.split(",")[1].rstrip(")") if "," in name else None
            ),
            violation_amount=amount,
            description=name,
        )
        for name, amount in violations
    ]
    
    # Map back to dict
    coherent_prices = {mid: float(q[i]) for i, mid in enumerate(market_ids)}
    original_prices = {mid: float(p[i]) for i, mid in enumerate(market_ids)}
    
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
    """Run Barrier Frank-Wolfe for numerical stability (OPT-03).
    
    Contracts the polytope toward the interior to avoid numerical
    issues near boundaries:
    
    M_epsilon = (1 - epsilon) * M + epsilon * mu_0
    
    Where M is the original polytope and mu_0 is the centroid.
    epsilon starts large (~0.1) and shrinks as we converge.
    
    Args:
        market_prices: Dict of market_id -> price
        constraints: Constraint matrix
        config: Optimization configuration
        
    Returns:
        ArbitrageResult with coherent prices
    """
    if config is None:
        config = OptimizationConfig()
    
    # Map prices to vector
    market_ids = constraints.market_ids
    n = len(market_ids)
    p = np.array([market_prices.get(mid, 0.5) for mid in market_ids])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    
    # Initialize LMO
    lmo = LinearMinimizationOracle(constraints)
    
    # Find centroid for barrier contraction
    center = init_fw(lmo, p)
    
    # Start from projected prices
    q = project_onto_polytope(p.copy(), lmo)
    
    # Start with barrier
    epsilon = config.initial_barrier
    
    converged = False
    final_gap = float("inf")
    total_iters = 0
    
    while epsilon >= config.min_barrier:
        # Run FW with contracted polytope
        for t in range(config.max_iterations // 10):
            total_iters += 1
            
            # Gradient
            g = kl_gradient(p, q)
            
            # LMO with contracted polytope
            s, _ = lmo.solve(g, epsilon=epsilon, center=center)
            
            # Duality gap
            gap = compute_duality_gap(p, q, s)
            final_gap = gap
            
            if gap < config.tolerance:
                converged = True
                break
            
            # Step size
            direction = s - q
            if config.line_search:
                gamma = line_search_kl(p, q, direction, max_step=1.0)
            else:
                gamma = 2.0 / (t + 2)
            
            # Update
            q = q + gamma * direction
            q = np.clip(q, 1e-10, 1 - 1e-10)
        
        if converged:
            break
        
        # Reduce barrier
        epsilon *= config.barrier_decay
        
        if config.verbose:
            kl = kl_divergence(p, q)
            print("Barrier eps=%.4f: KL=%.6f, gap=%.6f" % (epsilon, kl, final_gap))
    
    # Final polishing without barrier
    for t in range(min(100, config.max_iterations)):
        total_iters += 1
        
        g = kl_gradient(p, q)
        s, _ = lmo.solve(g, epsilon=0)  # No barrier
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
    
    # Compute final metrics
    final_kl = kl_divergence(p, q)
    
    # Find violated constraints in original prices
    violations = lmo.find_violated_constraints(p)
    constraint_violations = [
        ConstraintViolation(
            constraint_type=name.split("(")[0],
            from_market=name.split("(")[1].rstrip(")").split("->")[0].split(",")[0] if "(" in name else "",
            to_market=name.split("->")[1].rstrip(")") if "->" in name else (
                name.split(",")[1].rstrip(")") if "," in name else None
            ),
            violation_amount=amount,
            description=name,
        )
        for name, amount in violations
    ]
    
    coherent_prices = {mid: float(q[i]) for i, mid in enumerate(market_ids)}
    original_prices = {mid: float(p[i]) for i, mid in enumerate(market_ids)}
    
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
    """Alternative solver using projected gradient descent.
    
    Can be faster than Frank-Wolfe for some problems.
    Uses scipy's SLSQP optimizer with KL objective.
    
    Args:
        market_prices: Dict of market_id -> price
        constraints: Constraint matrix
        config: Optimization configuration
        
    Returns:
        ArbitrageResult with coherent prices
    """
    if config is None:
        config = OptimizationConfig()
    
    market_ids = constraints.market_ids
    n = len(market_ids)
    p = np.array([market_prices.get(mid, 0.5) for mid in market_ids])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    
    lmo = LinearMinimizationOracle(constraints)
    
    # Objective: minimize KL(p || q)
    def objective(q):
        return kl_divergence(p, q)
    
    def gradient(q):
        return kl_gradient(p, q)
    
    # Build constraints for scipy
    scipy_constraints = []
    A = constraints.A
    b = constraints.b
    
    for i in range(A.shape[0]):
        scipy_constraints.append({
            'type': 'ineq',
            'fun': lambda q, i=i: b[i] - np.dot(A[i], q),
            'jac': lambda q, i=i: -A[i],
        })
    
    bounds = [(0.01, 0.99) for _ in range(n)]
    
    # Start from projected market prices
    q0 = project_onto_polytope(p.copy(), lmo)
    
    result = minimize(
        objective,
        q0,
        method='SLSQP',
        jac=gradient,
        constraints=scipy_constraints,
        bounds=bounds,
        options={'maxiter': config.max_iterations, 'ftol': config.tolerance}
    )
    
    q = result.x
    final_kl = kl_divergence(p, q)
    
    # Find violated constraints in original prices
    violations = lmo.find_violated_constraints(p)
    constraint_violations = [
        ConstraintViolation(
            constraint_type=name.split("(")[0],
            from_market=name.split("(")[1].rstrip(")").split("->")[0].split(",")[0] if "(" in name else "",
            to_market=name.split("->")[1].rstrip(")") if "->" in name else (
                name.split(",")[1].rstrip(")") if "," in name else None
            ),
            violation_amount=amount,
            description=name,
        )
        for name, amount in violations
    ]
    
    coherent_prices = {mid: float(q[i]) for i, mid in enumerate(market_ids)}
    original_prices = {mid: float(p[i]) for i, mid in enumerate(market_ids)}
    
    return ArbitrageResult(
        market_prices=original_prices,
        coherent_prices=coherent_prices,
        kl_divergence=final_kl,
        constraints_violated=constraint_violations,
        converged=result.success,
        iterations=result.nit,
        final_gap=0.0,  # PGD doesn't track this
    )


def find_arbitrage(
    market_prices: dict[str, float],
    relationships: RelationshipGraph,
    config: OptimizationConfig | None = None,
    use_barrier: bool = True,
) -> ArbitrageResult:
    """High-level API: Find arbitrage-free prices from market relationships.
    
    This is the main entry point for the optimization engine.
    Takes the relationship graph from the LLM and market prices,
    returns coherent (arbitrage-free) prices.
    
    Args:
        market_prices: Current market prices {market_id: price}
        relationships: RelationshipGraph from LLM agent
        config: Optimization configuration
        use_barrier: Use barrier variant for numerical stability
        
    Returns:
        ArbitrageResult with coherent prices and diagnostics
    """
    # Build constraints from relationship graph
    constraints = build_constraints_from_graph(relationships)
    
    # Run optimization
    if use_barrier:
        return barrier_frank_wolfe(market_prices, constraints, config)
    else:
        return frank_wolfe(market_prices, constraints, config)


def find_arbitrage_simple(
    market_prices: dict[str, float],
    constraints: ConstraintMatrix,
    config: OptimizationConfig | None = None,
) -> ArbitrageResult:
    """Simplified API using pre-built constraints.
    
    Useful when you want to build constraints manually without
    using the full RelationshipGraph.
    
    Args:
        market_prices: Current market prices
        constraints: Pre-built constraint matrix
        config: Optimization configuration
        
    Returns:
        ArbitrageResult
    """
    return barrier_frank_wolfe(market_prices, constraints, config)
