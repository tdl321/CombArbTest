//! Core data types for the solver.
//!
//! Mirrors the Python `schema.py` but with efficient Rust representations.
//! Uses contiguous offsets instead of HashMap lookups for hot paths.

use std::collections::HashMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

/// A single condition (outcome) in the condition space.
#[derive(Debug, Clone)]
pub struct Condition {
    pub condition_id: String,
    pub market_id: String,
    pub outcome_index: usize,
    pub outcome_name: String,
}

/// The full condition space across all markets.
///
/// Key optimization: `market_offsets` gives O(1) range lookups,
/// avoiding HashMap access on the hot path.
#[derive(Debug, Clone)]
pub struct ConditionSpace {
    pub conditions: Vec<Condition>,
    /// For market i, conditions are at indices [market_offsets[i]..market_offsets[i+1])
    pub market_offsets: Vec<usize>,
    /// Market ID -> market index (only used at boundary, not in hot loops)
    pub market_index: HashMap<String, usize>,
    pub market_ids: Vec<String>,
}

impl ConditionSpace {
    /// Build from market data, defaulting to YES/NO outcomes.
    pub fn from_market_data(
        market_ids: &[String],
        market_outcomes: Option<&HashMap<String, Vec<String>>>,
    ) -> Self {
        let mut conditions = Vec::new();
        let mut market_offsets = Vec::with_capacity(market_ids.len() + 1);
        let mut market_index = HashMap::new();
        let mut offset = 0;

        for (m_idx, market_id) in market_ids.iter().enumerate() {
            market_offsets.push(offset);
            market_index.insert(market_id.clone(), m_idx);

            let outcomes = market_outcomes
                .and_then(|mo| mo.get(market_id))
                .cloned()
                .unwrap_or_else(|| vec!["YES".to_string(), "NO".to_string()]);

            for (o_idx, outcome_name) in outcomes.iter().enumerate() {
                conditions.push(Condition {
                    condition_id: format!("{}::{}", market_id, outcome_name),
                    market_id: market_id.clone(),
                    outcome_index: o_idx,
                    outcome_name: outcome_name.clone(),
                });
                offset += 1;
            }
        }
        market_offsets.push(offset);

        Self {
            conditions,
            market_offsets,
            market_index,
            market_ids: market_ids.to_vec(),
        }
    }

    /// Get condition index range for a market (O(1)).
    #[inline(always)]
    pub fn condition_range(&self, market_idx: usize) -> Range<usize> {
        self.market_offsets[market_idx]..self.market_offsets[market_idx + 1]
    }

    #[inline(always)]
    pub fn n_conditions(&self) -> usize {
        self.conditions.len()
    }

    #[inline(always)]
    pub fn n_markets(&self) -> usize {
        self.market_ids.len()
    }

    /// YES index for market (first outcome).
    #[inline(always)]
    pub fn yes_index(&self, market_idx: usize) -> usize {
        self.market_offsets[market_idx]
    }

    /// NO index for market (second outcome).
    #[inline(always)]
    pub fn no_index(&self, market_idx: usize) -> usize {
        self.market_offsets[market_idx] + 1
    }

    /// Look up market index by string ID.
    #[inline]
    pub fn market_idx(&self, market_id: &str) -> Option<usize> {
        self.market_index.get(market_id).copied()
    }
}

/// Sparse constraint matrix in CSR format.
///
/// Stores equality constraints (A_eq @ z = b_eq) and
/// inequality constraints (A_ub @ z <= b_ub).
#[derive(Debug, Clone)]
pub struct ConstraintMatrix {
    // Equality constraints
    pub eq_row_offsets: Vec<usize>,
    pub eq_col_indices: Vec<usize>,
    pub eq_values: Vec<f64>,
    pub b_eq: Vec<f64>,

    // Inequality constraints
    pub ub_row_offsets: Vec<usize>,
    pub ub_col_indices: Vec<usize>,
    pub ub_values: Vec<f64>,
    pub b_ub: Vec<f64>,

    pub n_conditions: usize,
    pub space: ConditionSpace,
}

/// Step mode for the Frank-Wolfe algorithm.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StepMode {
    Adaptive,
    LineSearch,
    Fixed(f64),
}

/// Optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub epsilon_init: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
    pub step_mode: StepMode,
    pub smoothness_alpha: f64,
    pub initial_smoothness: f64,
    pub min_smoothness: f64,
    pub fixed_step_size: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            epsilon_init: 0.1,
            epsilon_min: 0.001,
            epsilon_decay: 0.9,
            step_mode: StepMode::LineSearch,
            smoothness_alpha: 0.1,
            initial_smoothness: 1.0,
            min_smoothness: 1e-6,
            fixed_step_size: 0.5,
        }
    }
}

/// Result of the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageResult {
    /// Original market prices (condition_id -> price)
    pub condition_prices: HashMap<String, f64>,
    /// Coherent projected prices (condition_id -> price)
    pub coherent_condition_prices: HashMap<String, f64>,
    /// Original market prices (market_id -> [p_yes, p_no, ...])
    pub market_prices: HashMap<String, Vec<f64>>,
    /// Coherent market prices
    pub coherent_market_prices: HashMap<String, Vec<f64>>,
    /// KL divergence between market and coherent prices
    pub kl_divergence: f64,
    /// Duality gap at termination
    pub duality_gap: f64,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of iterations run
    pub iterations: usize,
    /// Active vertices found (last 10, as integer lists)
    pub active_vertices: Vec<Vec<i32>>,
}

/// Relationship type between markets.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipType {
    Implies,
    MutuallyExclusive,
    Equivalent,
    Opposite,
    Prerequisite,
    Incompatible,
    Exhaustive,
    And,
    Or,
}

/// A relationship between two markets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRelationship {
    #[serde(rename = "type")]
    pub rel_type: String,
    pub from_market: String,
    pub to_market: Option<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

fn default_confidence() -> f64 {
    1.0
}

/// A cluster of related markets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketCluster {
    pub cluster_id: String,
    pub market_ids: Vec<String>,
    pub relationships: Vec<MarketRelationship>,
}

/// The full relationship graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipGraph {
    pub clusters: Vec<MarketCluster>,
}

impl RelationshipGraph {
    /// Get all relationships involving the given markets.
    pub fn get_relationships(&self, market_ids: &[String]) -> Vec<&MarketRelationship> {
        let market_set: std::collections::HashSet<&str> =
            market_ids.iter().map(|s| s.as_str()).collect();
        let mut rels = Vec::new();
        for cluster in &self.clusters {
            for rel in &cluster.relationships {
                if market_set.contains(rel.from_market.as_str())
                    || rel
                        .to_market
                        .as_ref()
                        .map_or(false, |t| market_set.contains(t.as_str()))
                {
                    rels.push(rel);
                }
            }
        }
        rels
    }
}
