"""Optimizer module for combinatorial arbitrage detection.

This module implements Frank-Wolfe optimization over the marginal polytope
to detect arbitrage opportunities in prediction markets with logical constraints.

Main Entry Points:
    - find_arbitrage: High-level API (backward-compatible, uses dict[str, float])
    - find_marginal_arbitrage: New API with full condition space
    - detect_arbitrage_simple: Simplified API for common patterns

Key Classes:
    - ConditionSpace: Models the outcome space across markets
    - MarginalConstraintBuilder: Builds IP constraints for the marginal polytope
    - MarginalPolytopeLMO: Linear Minimization Oracle using combinatorial vertex enumeration
    - ArbitrageResult: Result container (backward-compatible)
    - MarginalArbitrageResult: Full result with condition-level detail
"""

# Schema - new types
from .schema import (
    Condition,
    ConditionSpace,
    MarketCluster,
    MarketRelationship,
    MarginalArbitrageResult,
    OptimizationConfig,
    RelationshipGraph,
    RelationshipType,
    # Backward-compatible types
    ArbitrageResult,
    ConstraintViolation,
)

# Divergence - new functions
from .divergence import (
    build_theta_from_prices,
    categorical_kl,
    categorical_kl_gradient,
    line_search_categorical_kl,
    line_search_exact,
    mu_to_market_prices,
    # Backward-compatible functions (Bernoulli KL)
    kl_divergence,
    kl_gradient,
    kl_hessian_diag,
    line_search_kl,
    compute_duality_gap,
)

# LMO - new types
from .lmo import (
    MarginalConstraintBuilder,
    MarginalConstraintMatrix,
    MarginalPolytopeLMO,
    build_constraints_from_graph,
    enumerate_vertices_combinatorial,
    # Backward-compatible aliases
    ConstraintMatrix,
    ConstraintBuilder,
    LinearMinimizationOracle,
    SolverMode,
)

# Frank-Wolfe - new and backward-compatible functions
from .frank_wolfe import (
    detect_arbitrage_simple,
    find_marginal_arbitrage,
    marginal_frank_wolfe,
    # Backward-compatible API
    find_arbitrage,
    find_arbitrage_simple,
    frank_wolfe,
    barrier_frank_wolfe,
    init_fw,
    projected_gradient_descent,
)


__all__ = [
    # Schema - new
    "Condition",
    "ConditionSpace",
    "MarketCluster",
    "MarketRelationship",
    "MarginalArbitrageResult",
    "OptimizationConfig",
    "RelationshipGraph",
    "RelationshipType",
    # Schema - backward-compatible
    "ArbitrageResult",
    "ConstraintViolation",
    # Divergence - new
    "build_theta_from_prices",
    "categorical_kl",
    "categorical_kl_gradient",
    "line_search_categorical_kl",
    "line_search_exact",
    "mu_to_market_prices",
    # Divergence - backward-compatible
    "kl_divergence",
    "kl_gradient",
    "kl_hessian_diag",
    "line_search_kl",
    "compute_duality_gap",
    # LMO - new
    "MarginalConstraintBuilder",
    "MarginalConstraintMatrix",
    "MarginalPolytopeLMO",
    "build_constraints_from_graph",
    "enumerate_vertices_combinatorial",
    # LMO - backward-compatible
    "ConstraintMatrix",
    "ConstraintBuilder",
    "LinearMinimizationOracle",
    "SolverMode",
    # Frank-Wolfe - new
    "detect_arbitrage_simple",
    "find_marginal_arbitrage",
    "marginal_frank_wolfe",
    # Frank-Wolfe - backward-compatible
    "find_arbitrage",
    "find_arbitrage_simple",
    "frank_wolfe",
    "barrier_frank_wolfe",
    "init_fw",
    "projected_gradient_descent",
]
