//! KL divergence functions for categorical distributions.
//!
//! All functions operate on flat f64 slices indexed by ConditionSpace.
//! Hot-path functions are `#[inline(always)]` and allocation-free.

use crate::types::ConditionSpace;

const EPS: f64 = 1e-10;

/// Compute KL(theta || mu) summed over all markets.
///
/// For each market m with outcomes i:
///   KL_m = sum_i theta_m[i] * ln(theta_m[i] / mu_m[i])
///
/// NOTE: No normalization — the FW algorithm guarantees mu stays
/// on the simplex via convex combinations of vertices.
#[inline]
pub fn categorical_kl(theta: &[f64], mu: &[f64], space: &ConditionSpace) -> f64 {
    let mut total_kl = 0.0;

    for m in 0..space.n_markets() {
        let range = space.condition_range(m);

        // Clip and compute KL for this market
        for i in range {
            let t = theta[i].clamp(EPS, 1.0 - EPS);
            let m_val = mu[i].clamp(EPS, 1.0 - EPS);
            total_kl += t * (t / m_val).ln();
        }
    }

    total_kl
}

/// Gradient of KL(theta || mu) w.r.t. mu, written in-place.
///
/// For KL(theta || mu): d/d(mu_i) = -theta_i / mu_i
#[inline]
pub fn categorical_kl_gradient(
    theta: &[f64],
    mu: &[f64],
    space: &ConditionSpace,
    gradient: &mut [f64],
) {
    for m in 0..space.n_markets() {
        let range = space.condition_range(m);
        for i in range {
            let mu_safe = mu[i].clamp(EPS, 1.0 - EPS);
            gradient[i] = -theta[i] / mu_safe;
        }
    }
}

/// Golden section line search for optimal step size.
///
/// Minimizes f(gamma) = KL(theta || mu + gamma * direction) over gamma in [0, max_gamma].
#[inline]
pub fn line_search_exact(
    theta: &[f64],
    mu: &[f64],
    direction: &[f64],
    space: &ConditionSpace,
    max_gamma: f64,
    tol: f64,
    scratch: &mut [f64],
) -> f64 {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let n = mu.len();

    let mut objective = |gamma: f64| -> f64 {
        for i in 0..n {
            scratch[i] = mu[i] + gamma * direction[i];
            if scratch[i] < 0.0 || scratch[i] > 1.0 {
                return f64::INFINITY;
            }
        }
        categorical_kl(theta, scratch, space)
    };

    let mut a = 0.0_f64;
    let mut b = max_gamma;
    let mut c = b - (b - a) / phi;
    let mut d = a + (b - a) / phi;
    let mut fc = objective(c);
    let mut fd = objective(d);

    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) / phi;
            fc = objective(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) / phi;
            fd = objective(d);
        }
    }

    (a + b) / 2.0
}

/// Build theta vector from market prices dict.
///
/// Normalizes each market's prices to sum to 1.
pub fn build_theta_from_prices(
    market_prices: &std::collections::HashMap<String, Vec<f64>>,
    space: &ConditionSpace,
    theta: &mut [f64],
) {
    for (m_idx, market_id) in space.market_ids.iter().enumerate() {
        let range = space.condition_range(m_idx);
        let default_prices = vec![0.5, 0.5];
        let prices = market_prices.get(market_id).unwrap_or(&default_prices);

        // Normalize prices to sum to 1
        let sum: f64 = prices.iter().sum();
        let sum = if sum > 0.0 { sum } else { 1.0 };

        for (offset, i) in range.enumerate() {
            if offset < prices.len() {
                theta[i] = prices[offset] / sum;
            } else {
                let n_outcomes = space.market_offsets[m_idx + 1] - space.market_offsets[m_idx];
                theta[i] = 1.0 / n_outcomes as f64;
            }
        }
    }
}

/// Convert mu vector back to market prices format.
///
/// Normalizes each market's mu values to sum to 1.
pub fn mu_to_market_prices(
    mu: &[f64],
    space: &ConditionSpace,
) -> std::collections::HashMap<String, Vec<f64>> {
    let mut market_prices = std::collections::HashMap::new();

    for (m_idx, market_id) in space.market_ids.iter().enumerate() {
        let range = space.condition_range(m_idx);
        let mut prices: Vec<f64> = range.clone().map(|i| mu[i]).collect();

        let total: f64 = prices.iter().sum();
        if total > 0.0 {
            for p in prices.iter_mut() {
                *p /= total;
            }
        } else {
            let n = prices.len() as f64;
            for p in prices.iter_mut() {
                *p = 1.0 / n;
            }
        }

        market_prices.insert(market_id.clone(), prices);
    }

    market_prices
}
