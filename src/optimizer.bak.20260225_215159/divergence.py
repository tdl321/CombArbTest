"""KL Divergence calculations for the Optimization Engine.

OPT-04: KL Divergence
- For combinatorial markets, each market i has a Bernoulli distribution
- KL for market i: p_i * log(p_i / q_i) + (1-p_i) * log((1-p_i) / (1-q_i))
- Total KL is sum over all markets
"""

import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Small constant to avoid numerical issues
EPS = 1e-12


def bernoulli_kl(p: float, q: float) -> float:
    """KL divergence between two Bernoulli distributions."""
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)

    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute total KL divergence across all markets."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.shape != q.shape:
        logger.error("[KL] Shape mismatch: p.shape=%s, q.shape=%s", p.shape, q.shape)
        raise ValueError("Shape mismatch: p.shape=%s, q.shape=%s" % (p.shape, q.shape))

    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)

    kl_per_market = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    total_kl = float(np.sum(kl_per_market))
    logger.debug("[KL] Computed KL divergence: %.6f over %d markets", total_kl, len(p))

    return total_kl


def kl_gradient(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute gradient of total KL divergence with respect to q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)

    return -p / q + (1 - p) / (1 - q)


def kl_hessian_diag(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute diagonal of Hessian of KL divergence."""
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
    """Find optimal step size for KL divergence along direction."""
    from scipy.optimize import minimize_scalar

    def objective(gamma: float) -> float:
        q_new = q + gamma * direction
        if np.any(q_new < EPS) or np.any(q_new > 1 - EPS):
            return float("inf")
        return kl_divergence(p, q_new)

    result = minimize_scalar(
        objective,
        bounds=(0, max_step),
        method="bounded",
        options={"xatol": 1e-8}
    )

    logger.debug("[KL] Line search: optimal gamma=%.6f", result.x)
    return float(result.x)


def compute_duality_gap(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    s: NDArray[np.float64],
) -> float:
    """Compute Frank-Wolfe duality gap."""
    grad = kl_gradient(p, q)
    gap = float(np.dot(grad, q - s))
    return max(0.0, gap)
