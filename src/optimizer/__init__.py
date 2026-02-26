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
    - MarginalPolytopeLMO: Linear Minimization Oracle using MILP
    - ArbitrageResult: Result container (backward-compatible)
    - MarginalArbitrageResult: Full result with condition-level detail
"""

# =============================================================================
# Rust Solver Integration
# =============================================================================
# Try to import the Rust solver for 4500x+ speedup.
# Falls back transparently to the Python implementation if unavailable.
import logging as _logging

_rust_logger = _logging.getLogger(__name__)

try:
    from solver import (
        find_marginal_arbitrage as _rust_find_marginal_arbitrage,
        detect_arbitrage_simple as _rust_detect_arbitrage_simple,
    )
    RUST_SOLVER_AVAILABLE = True
    _rust_logger.info("[Optimizer] Rust solver loaded — using native backend")
except ImportError:
    RUST_SOLVER_AVAILABLE = False
    _rust_logger.debug("[Optimizer] Rust solver not available — using Python backend")

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
    # Backward-compatible aliases
    ConstraintMatrix,
    ConstraintBuilder,
    LinearMinimizationOracle,
    SolverMode,
)

# Frank-Wolfe - new and backward-compatible functions
from .frank_wolfe import (
    # New API (Python implementations — may be wrapped by Rust below)
    detect_arbitrage_simple as _py_detect_arbitrage_simple,
    find_marginal_arbitrage as _py_find_marginal_arbitrage,
    marginal_frank_wolfe,
    # Backward-compatible API
    find_arbitrage,
    find_arbitrage_simple,
    frank_wolfe,
    barrier_frank_wolfe,
    init_fw,
    projected_gradient_descent,
)

def find_marginal_arbitrage(
    market_prices,
    relationships,
    market_outcomes=None,
    config=None,
    force_python=False,
):
    """Find marginal arbitrage. Uses Rust solver when available for 4500x+ speedup.

    Args:
        market_prices: dict[str, list[float]] — market prices
        relationships: RelationshipGraph — logical constraints
        market_outcomes: Optional dict[str, list[str]] — outcome names
        config: Optional OptimizationConfig
        force_python: If True, always use Python implementation

    Returns:
        MarginalArbitrageResult
    """
    if RUST_SOLVER_AVAILABLE and not force_python:
        try:
            result_dict = _rust_find_marginal_arbitrage(
                market_prices=market_prices,
                relationships=relationships,
                market_outcomes=market_outcomes,
                config=config,
            )
            # Convert dict result to MarginalArbitrageResult
            return MarginalArbitrageResult(
                condition_prices=result_dict["condition_prices"],
                coherent_condition_prices=result_dict["coherent_condition_prices"],
                market_prices=result_dict["market_prices"],
                coherent_market_prices=result_dict["coherent_market_prices"],
                kl_divergence=result_dict["kl_divergence"],
                duality_gap=result_dict["duality_gap"],
                converged=result_dict["converged"],
                iterations=result_dict["iterations"],
                active_vertices=result_dict.get("active_vertices", []),
            )
        except Exception as e:
            _rust_logger.warning("[Optimizer] Rust solver failed, falling back to Python: %s", e)

    return _py_find_marginal_arbitrage(
        market_prices=market_prices,
        relationships=relationships,
        market_outcomes=market_outcomes,
        config=config,
    )


def detect_arbitrage_simple(
    market_prices,
    implications=None,
    mutex_pairs=None,
    config=None,
    force_python=False,
):
    """Detect arbitrage with simple constraints. Uses Rust solver when available.

    Args:
        market_prices: dict[str, list[float]]
        implications: Optional list of (from, to) tuples
        mutex_pairs: Optional list of (a, b) tuples
        config: Optional OptimizationConfig
        force_python: If True, always use Python implementation

    Returns:
        MarginalArbitrageResult
    """
    if RUST_SOLVER_AVAILABLE and not force_python:
        try:
            result_dict = _rust_detect_arbitrage_simple(
                market_prices=market_prices,
                implications=implications,
                mutex_pairs=mutex_pairs,
                config=config,
            )
            return MarginalArbitrageResult(
                condition_prices=result_dict["condition_prices"],
                coherent_condition_prices=result_dict["coherent_condition_prices"],
                market_prices=result_dict["market_prices"],
                coherent_market_prices=result_dict["coherent_market_prices"],
                kl_divergence=result_dict["kl_divergence"],
                duality_gap=result_dict["duality_gap"],
                converged=result_dict["converged"],
                iterations=result_dict["iterations"],
                active_vertices=result_dict.get("active_vertices", []),
            )
        except Exception as e:
            _rust_logger.warning("[Optimizer] Rust solver failed, falling back to Python: %s", e)

    return _py_detect_arbitrage_simple(
        market_prices=market_prices,
        implications=implications,
        mutex_pairs=mutex_pairs,
        config=config,
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
