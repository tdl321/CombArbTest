"""Optimization Engine for Combinatorial Arbitrage.

This module implements the Frank-Wolfe algorithm to find arbitrage-free
prices given market relationships identified by the LLM.

Components:
- OPT-01: Frank-Wolfe solver (frank_wolfe.py)
- OPT-02: Linear Minimization Oracle with HiGHS (lmo.py)
- OPT-03: Barrier Frank-Wolfe variant (frank_wolfe.py)
- OPT-04: KL Divergence calculations (divergence.py)
- OPT-05: InitFW interior point finder (frank_wolfe.py)

Usage:
    from src.optimizer import find_arbitrage, ArbitrageResult
    from src.optimizer.schema import RelationshipGraph, MarketCluster, MarketRelationship
    
    # Create relationships (normally from LLM)
    graph = RelationshipGraph(clusters=[
        MarketCluster(
            cluster_id="test",
            market_ids=["A", "B"],
            relationships=[
                MarketRelationship(type="implies", from_market="A", to_market="B", confidence=1.0)
            ]
        )
    ])
    
    # Market prices with arbitrage opportunity
    prices = {"A": 0.6, "B": 0.5}  # Violates A implies B
    
    # Find arbitrage-free prices
    result = find_arbitrage(prices, graph)
    
    print(f"Arbitrage detected: {result.has_arbitrage}")
    print(f"Coherent prices: {result.coherent_prices}")
"""

from .schema import (
    MarketRelationship,
    MarketCluster,
    RelationshipGraph,
    ConstraintViolation,
    ArbitrageResult,
    OptimizationConfig,
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

from .frank_wolfe import (
    init_fw,
    frank_wolfe,
    barrier_frank_wolfe,
    projected_gradient_descent,
    find_arbitrage,
    find_arbitrage_simple,
)

__all__ = [
    # Schema
    "MarketRelationship",
    "MarketCluster",
    "RelationshipGraph",
    "ConstraintViolation",
    "ArbitrageResult",
    "OptimizationConfig",
    # Divergence
    "kl_divergence",
    "kl_gradient",
    "line_search_kl",
    "compute_duality_gap",
    # LMO
    "ConstraintMatrix",
    "ConstraintBuilder",
    "LinearMinimizationOracle",
    "build_constraints_from_graph",
    # Frank-Wolfe
    "init_fw",
    "frank_wolfe",
    "barrier_frank_wolfe",
    "projected_gradient_descent",
    "find_arbitrage",
    "find_arbitrage_simple",
]
