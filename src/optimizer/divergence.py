"""KL Divergence calculations for the Optimization Engine.

OPT-04: KL Divergence
- For combinatorial markets, each market i has a Bernoulli distribution
- KL for market i: p_i * log(p_i / q_i) + (1-p_i) * log((1-p_i) / (1-q_i))
- Total KL is sum over all markets
"""

import numpy as np
from numpy.typing import NDArray


# Small constant to avoid numerical issues
EPS = 1e-12


def bernoulli_kl(p: float, q: float) -> float:
    """KL divergence between two Bernoulli distributions.
    
    KL(Bernoulli(p) || Bernoulli(q)) = 
        p * log(p/q) + (1-p) * log((1-p)/(1-q))
    
    Args:
        p: First probability (market price)
        q: Second probability (coherent price)
        
    Returns:
        KL divergence (non-negative)
    """
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute total KL divergence across all markets.
    
    Treats each market as an independent Bernoulli distribution.
    KL_total = sum_i KL(Bernoulli(p_i) || Bernoulli(q_i))
    
    This is the proper objective for finding arbitrage-free prices
    that are closest to market prices in the information-theoretic sense.
    
    Args:
        p: Market prices (observed probabilities)
        q: Coherent prices (arbitrage-free probabilities)
        
    Returns:
        Total KL divergence (non-negative)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p.shape={p.shape}, q.shape={q.shape}")
    
    # Clip to valid probability range
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    
    # Sum of Bernoulli KL divergences
    kl_per_market = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    return float(np.sum(kl_per_market))


def kl_gradient(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute gradient of total KL divergence with respect to q.
    
    d/dq_i KL = d/dq_i [p_i * log(p_i/q_i) + (1-p_i) * log((1-p_i)/(1-q_i))]
              = -p_i/q_i + (1-p_i)/(1-q_i)
    
    Args:
        p: Market prices
        q: Current coherent prices
        
    Returns:
        Gradient vector of shape (n,)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Clip to avoid division by zero
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    
    # Gradient: -p/q + (1-p)/(1-q)
    return -p / q + (1 - p) / (1 - q)


def kl_hessian_diag(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute diagonal of Hessian of KL divergence.
    
    d^2/dq_i^2 KL = p_i/q_i^2 + (1-p_i)/(1-q_i)^2
    
    Always positive, so the objective is strictly convex.
    
    Args:
        p: Market prices
        q: Current coherent prices
        
    Returns:
        Diagonal of Hessian matrix
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    
    return p / (q ** 2) + (1 - p) / ((1 - q) ** 2)


def line_search_kl(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    direction: NDArray[np.float64],
    max_step: float = 1.0,
) -> float:
    """Find optimal step size for KL divergence along direction.
    
    Solve: min_{gamma in [0, max_step]} KL(p || q + gamma * direction)
    
    Args:
        p: Market prices
        q: Current coherent prices
        direction: Search direction (s - q for Frank-Wolfe)
        max_step: Maximum step size
        
    Returns:
        Optimal step size gamma in [0, max_step]
    """
    from scipy.optimize import minimize_scalar
    
    def objective(gamma: float) -> float:
        q_new = q + gamma * direction
        # Ensure q_new stays in valid range
        if np.any(q_new < EPS) or np.any(q_new > 1 - EPS):
            return float("inf")
        return kl_divergence(p, q_new)
    
    # Use bounded optimization
    result = minimize_scalar(
        objective,
        bounds=(0, max_step),
        method="bounded",
        options={"xatol": 1e-8}
    )
    
    return float(result.x)


def compute_duality_gap(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    s: NDArray[np.float64],
) -> float:
    """Compute Frank-Wolfe duality gap.
    
    Gap = <grad_f(q), q - s>
    
    This provides an upper bound on how far we are from optimal.
    When gap < tolerance, we can stop.
    
    Args:
        p: Market prices
        q: Current coherent prices
        s: LMO solution (vertex of polytope)
        
    Returns:
        Duality gap (non-negative)
    """
    grad = kl_gradient(p, q)
    gap = float(np.dot(grad, q - s))
    return max(0.0, gap)
