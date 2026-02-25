"""KL divergence functions for categorical distributions.

For marginal polytope optimization, we compute KL divergence over the
condition space, treating each market as an independent categorical distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .schema import ConditionSpace


def categorical_kl(
    theta: NDArray,
    mu: NDArray,
    condition_space: ConditionSpace,
    eps: float = 1e-10,
) -> float:
    """Compute KL divergence for categorical distributions.

    For each market, computes KL(theta_m || mu_m) where theta_m and mu_m
    are the probability distributions over outcomes for market m.

    KL(p || q) = sum_i p_i * log(p_i / q_i)

    Args:
        theta: Market prices (reference distribution) in condition space
        mu: Current estimate (approximation) in condition space
        condition_space: The condition space structure
        eps: Small constant for numerical stability

    Returns:
        Total KL divergence summed over all markets
    """
    total_kl = 0.0

    for market_id in condition_space.market_ids:
        indices = condition_space.get_condition_indices(market_id)

        # Extract probabilities for this market
        theta_m = theta[indices]
        mu_m = mu[indices]

        # Normalize to ensure valid distributions
        theta_m = np.clip(theta_m, eps, 1.0 - eps)
        mu_m = np.clip(mu_m, eps, 1.0 - eps)

        theta_m = theta_m / theta_m.sum()
        mu_m = mu_m / mu_m.sum()

        # KL divergence for this market
        kl_m = np.sum(theta_m * np.log(theta_m / mu_m))
        total_kl += kl_m

    return total_kl


def categorical_kl_gradient(
    theta: NDArray,
    mu: NDArray,
    condition_space: ConditionSpace,
    eps: float = 1e-10,
) -> NDArray:
    """Gradient of KL divergence w.r.t. mu.

    For KL(theta || mu), the gradient w.r.t. mu_i is:
    d/d(mu_i) KL = -theta_i / mu_i

    This is the direction that, when followed, decreases the KL divergence.

    Args:
        theta: Market prices (reference distribution)
        mu: Current estimate
        condition_space: The condition space structure
        eps: Small constant for numerical stability

    Returns:
        Gradient vector of same shape as mu
    """
    gradient = np.zeros_like(mu)

    for market_id in condition_space.market_ids:
        indices = condition_space.get_condition_indices(market_id)

        theta_m = theta[indices]
        mu_m = mu[indices]

        # Clip for numerical stability
        mu_m_safe = np.clip(mu_m, eps, 1.0 - eps)

        # Gradient: -theta_i / mu_i
        gradient[indices] = -theta_m / mu_m_safe

    return gradient


def line_search_categorical_kl(
    theta: NDArray,
    mu: NDArray,
    direction: NDArray,
    condition_space: ConditionSpace,
    max_gamma: float = 1.0,
    n_steps: int = 20,
) -> float:
    """Find optimal step size for Frank-Wolfe update.

    Performs line search to minimize KL(theta || mu + gamma * direction).

    Args:
        theta: Market prices (reference distribution)
        mu: Current estimate
        direction: Update direction (typically vertex - mu)
        condition_space: The condition space structure
        max_gamma: Maximum step size
        n_steps: Number of steps for grid search

    Returns:
        Optimal step size gamma in [0, max_gamma]
    """
    best_gamma = 0.0
    best_kl = categorical_kl(theta, mu, condition_space)

    for i in range(1, n_steps + 1):
        gamma = max_gamma * i / n_steps
        mu_new = mu + gamma * direction

        # Ensure we stay in valid probability range
        if np.any(mu_new < 0) or np.any(mu_new > 1):
            break

        kl = categorical_kl(theta, mu_new, condition_space)
        if kl < best_kl:
            best_kl = kl
            best_gamma = gamma

    return best_gamma


def line_search_exact(
    theta: NDArray,
    mu: NDArray,
    direction: NDArray,
    condition_space: ConditionSpace,
    max_gamma: float = 1.0,
    tol: float = 1e-8,
) -> float:
    """Exact line search using golden section search.

    More precise than grid search but slower.

    Args:
        theta: Market prices (reference distribution)
        mu: Current estimate
        direction: Update direction
        condition_space: The condition space structure
        max_gamma: Maximum step size
        tol: Tolerance for convergence

    Returns:
        Optimal step size gamma
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    def objective(gamma: float) -> float:
        mu_new = mu + gamma * direction
        # Penalize out-of-bounds
        if np.any(mu_new < 0) or np.any(mu_new > 1):
            return float("inf")
        return categorical_kl(theta, mu_new, condition_space)

    a, b = 0.0, max_gamma
    c = b - (b - a) / phi
    d = a + (b - a) / phi

    fc, fd = objective(c), objective(d)

    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - (b - a) / phi
            fc = objective(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (b - a) / phi
            fd = objective(d)

    return (a + b) / 2


def build_theta_from_prices(
    market_prices: dict[str, list[float]],
    condition_space: ConditionSpace,
) -> NDArray:
    """Convert market prices to theta vector in condition space.

    Args:
        market_prices: Dict of market_id -> [p_yes, p_no, ...]
        condition_space: The condition space structure

    Returns:
        Theta vector with prices for each condition
    """
    theta = np.zeros(condition_space.n_conditions())

    for market_id in condition_space.market_ids:
        indices = condition_space.get_condition_indices(market_id)
        prices = market_prices.get(market_id, [0.5, 0.5])

        # Ensure prices sum to 1
        prices = np.array(prices)
        prices = prices / prices.sum()

        for i, idx in enumerate(indices):
            if i < len(prices):
                theta[idx] = prices[i]
            else:
                theta[idx] = 1.0 / len(indices)

    return theta


def mu_to_market_prices(
    mu: NDArray,
    condition_space: ConditionSpace,
) -> dict[str, list[float]]:
    """Convert mu vector back to market prices format.

    Args:
        mu: Probability vector in condition space
        condition_space: The condition space structure

    Returns:
        Dict of market_id -> [p_yes, p_no, ...]
    """
    market_prices = {}

    for market_id in condition_space.market_ids:
        indices = condition_space.get_condition_indices(market_id)
        prices = [mu[idx] for idx in indices]

        # Normalize to sum to 1
        total = sum(prices)
        if total > 0:
            prices = [p / total for p in prices]
        else:
            prices = [1.0 / len(prices)] * len(prices)

        market_prices[market_id] = prices

    return market_prices


# =============================================================================
# Backward Compatibility Functions (Bernoulli KL for single-probability markets)
# =============================================================================

EPS = 1e-12


def bernoulli_kl(p: float, q: float) -> float:
    """KL divergence between two Bernoulli distributions."""
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_divergence(p: NDArray, q: NDArray) -> float:
    """Compute total KL divergence across all markets (Bernoulli).

    Each element is treated as P(YES) for that market.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)

    kl_per_market = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return float(np.sum(kl_per_market))


def kl_gradient(p: NDArray, q: NDArray) -> NDArray:
    """Gradient of Bernoulli KL divergence w.r.t. q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)

    return -p / q + (1 - p) / (1 - q)


def kl_hessian_diag(p: NDArray, q: NDArray) -> NDArray:
    """Diagonal of Hessian of Bernoulli KL divergence."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)

    return p / (q**2) + (1 - p) / ((1 - q) ** 2)


def line_search_kl(
    p: NDArray,
    q: NDArray,
    direction: NDArray,
    max_step: float = 1.0,
) -> float:
    """Find optimal step size for Bernoulli KL along direction."""
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
        options={"xatol": 1e-8},
    )
    return float(result.x)


def compute_duality_gap(p: NDArray, q: NDArray, s: NDArray) -> float:
    """Compute Frank-Wolfe duality gap.

    gap = gradient(q)^T (q - s)
    where s is the LMO solution (vertex).
    """
    grad = kl_gradient(p, q)
    return float(grad @ (q - s))
