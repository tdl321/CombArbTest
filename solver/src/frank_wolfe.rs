//! Barrier Frank-Wolfe algorithm for marginal polytope optimization.
//!
//! Finds the coherent price vector mu in the marginal polytope that minimizes
//! KL(theta || mu), where theta represents market prices.
//!
//! All buffers are pre-allocated; the inner loop is allocation-free.

use std::collections::HashMap;

use crate::divergence::{
    build_theta_from_prices, categorical_kl, categorical_kl_gradient, line_search_exact,
    mu_to_market_prices,
};
use crate::lmo::MarginalPolytopeLMO;
use crate::types::{ArbitrageResult, ConstraintMatrix, OptimizationConfig, StepMode};

/// Compute adaptive Frank-Wolfe step size using the short-step rule.
///
/// gamma = min(1, gap / (L * ||d||^2))
#[inline(always)]
fn compute_adaptive_step(gap: f64, direction: &[f64], l_est: f64, l_min: f64) -> f64 {
    let d_norm_sq: f64 = direction.iter().map(|d| d * d).sum();
    if d_norm_sq < 1e-12 {
        return 0.0;
    }
    let l_safe = l_est.max(l_min);
    (gap / (l_safe * d_norm_sq)).clamp(0.0, 1.0)
}

/// Estimate local Lipschitz constant from gradient change.
///
/// L_k = ||grad_new - grad_old|| / ||x_new - x_old||
#[inline(always)]
fn estimate_smoothness(grad_new: &[f64], grad_old: &[f64], x_new: &[f64], x_old: &[f64]) -> f64 {
    let mut grad_diff_sq = 0.0;
    let mut x_diff_sq = 0.0;
    for i in 0..grad_new.len() {
        let gd = grad_new[i] - grad_old[i];
        let xd = x_new[i] - x_old[i];
        grad_diff_sq += gd * gd;
        x_diff_sq += xd * xd;
    }
    if x_diff_sq < 1e-24 {
        f64::INFINITY
    } else {
        grad_diff_sq.sqrt() / x_diff_sq.sqrt()
    }
}

/// Contract mu toward an interior point (Barrier FW).
///
/// M_eps = (1 - eps) * M + eps * centroid
#[inline(always)]
fn contract_toward_centroid(mu: &mut [f64], epsilon: f64, centroid: &[f64]) {
    for i in 0..mu.len() {
        mu[i] = (1.0 - epsilon) * mu[i] + epsilon * centroid[i];
    }
}

/// Main Frank-Wolfe optimization over the marginal polytope.
///
/// All buffers are pre-allocated before the loop. The inner loop does
/// zero heap allocations (aside from the LMO's solution vector).
pub fn marginal_frank_wolfe(
    market_prices: &HashMap<String, Vec<f64>>,
    constraints: &ConstraintMatrix,
    config: &OptimizationConfig,
) -> ArbitrageResult {
    let space = &constraints.space;
    let n = space.n_conditions();

    // Pre-allocate ALL buffers (zero allocations in the loop)
    let mut theta = vec![0.0; n];
    let mut mu = vec![0.0; n];
    let mut gradient = vec![0.0; n];
    let mut direction = vec![0.0; n];
    let mut scratch = vec![0.0; n];
    let mut grad_old = vec![0.0; n];
    let mut mu_old = vec![0.0; n];

    // 1. Build theta from market prices
    build_theta_from_prices(market_prices, space, &mut theta);

    // 2. Initialize LMO (persistent model — built once)
    let lmo = MarginalPolytopeLMO::new(constraints);

    // 3. Compute centroid (initial interior point)
    let centroid = lmo.compute_centroid();
    mu.copy_from_slice(&centroid);

    // 4. Contract toward interior (Barrier FW)
    let mut epsilon = config.epsilon_init;
    contract_toward_centroid(&mut mu, epsilon, &centroid);

    // 5. Frank-Wolfe loop
    let mut converged = false;
    let mut iterations = 0;
    let mut gap = f64::INFINITY;
    let mut active_vertices: Vec<Vec<f64>> = Vec::new();

    // Adaptive step state
    let mut l_est = config.initial_smoothness;
    let alpha = config.smoothness_alpha;
    let mut has_prev = false;

    for t in 0..config.max_iterations {
        iterations = t + 1;

        // Gradient of KL(theta || mu) w.r.t. mu
        categorical_kl_gradient(&theta, &mu, space, &mut gradient);

        // LMO: find vertex minimizing gradient^T z
        let (z_new, _obj) = lmo.solve(&gradient);
        active_vertices.push(z_new.clone());

        // Duality gap: gradient^T (mu - z_new)
        gap = 0.0;
        for i in 0..n {
            gap += gradient[i] * (mu[i] - z_new[i]);
            direction[i] = z_new[i] - mu[i];
        }

        if gap < config.tolerance {
            converged = true;
            break;
        }

        // Compute step size based on mode
        let gamma = match config.step_mode {
            StepMode::Adaptive => {
                if has_prev {
                    let l_k = estimate_smoothness(&gradient, &grad_old, &mu, &mu_old);
                    if l_k.is_finite() {
                        l_est = alpha * l_k + (1.0 - alpha) * l_est;
                    }
                }

                // Save for next iteration
                grad_old.copy_from_slice(&gradient);
                mu_old.copy_from_slice(&mu);
                has_prev = true;

                compute_adaptive_step(gap, &direction, l_est, config.min_smoothness)
            }
            StepMode::LineSearch => {
                line_search_exact(&theta, &mu, &direction, space, 1.0, 1e-8, &mut scratch)
            }
            StepMode::Fixed(step) => step,
        };

        // Update: mu = mu + gamma * direction
        for i in 0..n {
            mu[i] += gamma * direction[i];
        }

        // Decay epsilon
        epsilon = (epsilon * config.epsilon_decay).max(config.epsilon_min);

        // Re-contract toward interior (Barrier FW)
        contract_toward_centroid(&mut mu, epsilon, &centroid);
    }

    // 6. Final KL divergence
    let kl_divergence = categorical_kl(&theta, &mu, space);

    // 7. Build result
    let coherent_market_prices = mu_to_market_prices(&mu, space);

    // Build condition price maps
    let mut condition_prices = HashMap::new();
    let mut coherent_condition_prices = HashMap::new();
    for (i, cond) in space.conditions.iter().enumerate() {
        condition_prices.insert(cond.condition_id.clone(), theta[i]);
        coherent_condition_prices.insert(cond.condition_id.clone(), mu[i]);
    }

    // Active vertices (last 10, as integer lists)
    let active_verts_output: Vec<Vec<i32>> = active_vertices
        .iter()
        .rev()
        .take(10)
        .rev()
        .map(|v| v.iter().map(|x| x.round() as i32).collect())
        .collect();

    ArbitrageResult {
        condition_prices,
        coherent_condition_prices,
        market_prices: market_prices.clone(),
        coherent_market_prices,
        kl_divergence,
        duality_gap: gap,
        converged,
        iterations,
        active_vertices: active_verts_output,
    }
}
