//! Linear Minimization Oracle for the Marginal Polytope.
//!
//! For small binary problems (up to 20 markets = 2^20 combinations max),
//! we enumerate ALL feasible vertices combinatorially, checking each
//! candidate against the constraint matrix. This avoids HiGHS entirely
//! for the production case of 2-10 markets.
//!
//! After enumeration, each LMO solve is a linear scan over the vertex
//! table: O(V * n) where V is the number of vertices and n is the
//! dimension. For typical problems (V < 20, n < 20), this takes
//! a few hundred nanoseconds.

use crate::types::{ConditionSpace, ConstraintMatrix, MarketRelationship};

/// Builder for marginal polytope constraints.
pub struct MarginalConstraintBuilder {
    space: ConditionSpace,
    n: usize,
    eq_rows: Vec<(Vec<usize>, Vec<f64>)>,
    eq_rhs: Vec<f64>,
    ub_rows: Vec<(Vec<usize>, Vec<f64>)>,
    ub_rhs: Vec<f64>,
}

impl MarginalConstraintBuilder {
    pub fn new(space: ConditionSpace) -> Self {
        let n = space.n_conditions();
        let mut builder = Self {
            space,
            n,
            eq_rows: Vec::new(),
            eq_rhs: Vec::new(),
            ub_rows: Vec::new(),
            ub_rhs: Vec::new(),
        };
        builder.build_exactly_one_constraints();
        builder
    }

    fn build_exactly_one_constraints(&mut self) {
        for m in 0..self.space.n_markets() {
            let range = self.space.condition_range(m);
            let cols: Vec<usize> = range.collect();
            let vals: Vec<f64> = vec![1.0; cols.len()];
            self.eq_rows.push((cols, vals));
            self.eq_rhs.push(1.0);
        }
    }

    pub fn add_implies(&mut self, from_market: &str, to_market: &str) {
        let from_m = self.space.market_idx(from_market).expect("from_market not found");
        let to_m = self.space.market_idx(to_market).expect("to_market not found");
        let from_yes = self.space.yes_index(from_m);
        let to_yes = self.space.yes_index(to_m);
        self.ub_rows.push((vec![from_yes, to_yes], vec![1.0, -1.0]));
        self.ub_rhs.push(0.0);
    }

    pub fn add_mutually_exclusive(&mut self, market_a: &str, market_b: &str) {
        let a_m = self.space.market_idx(market_a).expect("market_a not found");
        let b_m = self.space.market_idx(market_b).expect("market_b not found");
        let a_yes = self.space.yes_index(a_m);
        let b_yes = self.space.yes_index(b_m);
        self.ub_rows.push((vec![a_yes, b_yes], vec![1.0, 1.0]));
        self.ub_rhs.push(1.0);
    }

    pub fn add_equivalent(&mut self, market_a: &str, market_b: &str) {
        self.add_implies(market_a, market_b);
        self.add_implies(market_b, market_a);
    }

    pub fn add_opposite(&mut self, market_a: &str, market_b: &str) {
        let a_m = self.space.market_idx(market_a).expect("market_a not found");
        let b_m = self.space.market_idx(market_b).expect("market_b not found");
        let a_yes = self.space.yes_index(a_m);
        let b_yes = self.space.yes_index(b_m);
        self.ub_rows.push((vec![a_yes, b_yes], vec![1.0, 1.0]));
        self.ub_rhs.push(1.0);
        self.ub_rows.push((vec![a_yes, b_yes], vec![-1.0, -1.0]));
        self.ub_rhs.push(-1.0);
    }

    pub fn add_or(&mut self, market_a: &str, market_b: &str) {
        let a_m = self.space.market_idx(market_a).expect("market_a not found");
        let b_m = self.space.market_idx(market_b).expect("market_b not found");
        let a_yes = self.space.yes_index(a_m);
        let b_yes = self.space.yes_index(b_m);
        self.ub_rows.push((vec![a_yes, b_yes], vec![-1.0, -1.0]));
        self.ub_rhs.push(-1.0);
    }

    pub fn add_relationship(&mut self, rel: &MarketRelationship) {
        let rel_type = rel.rel_type.to_lowercase();
        match rel_type.as_str() {
            "implies" => {
                if let Some(ref to) = rel.to_market {
                    self.add_implies(&rel.from_market, to);
                }
            }
            "mutually_exclusive" | "mutex" | "incompatible" => {
                if let Some(ref to) = rel.to_market {
                    self.add_mutually_exclusive(&rel.from_market, to);
                }
            }
            "equivalent" | "and" => {
                if let Some(ref to) = rel.to_market {
                    self.add_equivalent(&rel.from_market, to);
                }
            }
            "opposite" => {
                if let Some(ref to) = rel.to_market {
                    self.add_opposite(&rel.from_market, to);
                }
            }
            "prerequisite" => {
                if let Some(ref to) = rel.to_market {
                    self.add_implies(&rel.from_market, to);
                }
            }
            "or" => {
                if let Some(ref to) = rel.to_market {
                    self.add_or(&rel.from_market, to);
                }
            }
            "exhaustive" => {}
            _ => {}
        }
    }

    pub fn add_relationships(&mut self, rels: &[&MarketRelationship]) {
        for rel in rels {
            self.add_relationship(rel);
        }
    }

    pub fn build(self) -> ConstraintMatrix {
        let n = self.n;

        let mut eq_row_offsets = Vec::with_capacity(self.eq_rows.len() + 1);
        let mut eq_col_indices = Vec::new();
        let mut eq_values = Vec::new();
        eq_row_offsets.push(0);
        for (cols, vals) in &self.eq_rows {
            for (&c, &v) in cols.iter().zip(vals.iter()) {
                eq_col_indices.push(c);
                eq_values.push(v);
            }
            eq_row_offsets.push(eq_col_indices.len());
        }

        let mut ub_row_offsets = Vec::with_capacity(self.ub_rows.len() + 1);
        let mut ub_col_indices = Vec::new();
        let mut ub_values = Vec::new();
        ub_row_offsets.push(0);
        for (cols, vals) in &self.ub_rows {
            for (&c, &v) in cols.iter().zip(vals.iter()) {
                ub_col_indices.push(c);
                ub_values.push(v);
            }
            ub_row_offsets.push(ub_col_indices.len());
        }

        ConstraintMatrix {
            eq_row_offsets,
            eq_col_indices,
            eq_values,
            b_eq: self.eq_rhs,
            ub_row_offsets,
            ub_col_indices,
            ub_values,
            b_ub: self.ub_rhs,
            n_conditions: n,
            space: self.space,
        }
    }
}

// =============================================================================
// Combinatorial Vertex Enumeration
// =============================================================================

/// Enumerate all feasible vertices by combinatorial enumeration.
///
/// For N binary markets, there are 2^N possible joint outcomes.
/// For each candidate, we check the inequality constraints.
/// The equality constraints (exactly one per market) are built in
/// by construction: we enumerate over market outcomes, not conditions.
///
/// For 10 markets, this is 1024 candidates — trivial to enumerate.
/// For 20 markets, this is ~1M candidates — still fast in Rust.
fn enumerate_vertices_combinatorial(constraints: &ConstraintMatrix) -> Vec<Vec<f64>> {
    let space = &constraints.space;
    let n_markets = space.n_markets();
    let n_conditions = space.n_conditions();

    // Collect the number of outcomes per market
    let outcomes_per_market: Vec<usize> = (0..n_markets)
        .map(|m| space.condition_range(m).len())
        .collect();

    // Total combinations
    let total_combos: usize = outcomes_per_market.iter().product();

    let mut vertices = Vec::new();
    let mut z = vec![0.0; n_conditions];

    // Enumerate all combinations
    for combo in 0..total_combos {
        // Build the binary vector z for this combination
        for i in 0..n_conditions {
            z[i] = 0.0;
        }

        let mut remainder = combo;
        for m in 0..n_markets {
            let n_outcomes = outcomes_per_market[m];
            let choice = remainder % n_outcomes;
            remainder /= n_outcomes;
            let base = space.market_offsets[m];
            z[base + choice] = 1.0;
        }

        // Check inequality constraints: A_ub @ z <= b_ub
        let n_ub = constraints.b_ub.len();
        let mut feasible = true;
        for r in 0..n_ub {
            let start = constraints.ub_row_offsets[r];
            let end = constraints.ub_row_offsets[r + 1];
            let mut row_val = 0.0;
            for j in start..end {
                row_val += constraints.ub_values[j] * z[constraints.ub_col_indices[j]];
            }
            if row_val > constraints.b_ub[r] + 1e-10 {
                feasible = false;
                break;
            }
        }

        if feasible {
            vertices.push(z.clone());
        }
    }

    vertices
}

// =============================================================================
// The LMO: vertex table + linear scan
// =============================================================================

/// Linear Minimization Oracle using a pre-computed vertex table.
///
/// All feasible vertices are enumerated combinatorially during construction.
/// Each `solve(gradient)` call is a linear scan: O(V * n).
pub struct MarginalPolytopeLMO {
    /// All feasible vertices stored flat: vertices_flat[v * n + i]
    vertices_flat: Vec<f64>,
    n_vertices: usize,
    n: usize,
}

impl MarginalPolytopeLMO {
    /// Build the LMO by enumerating all feasible vertices combinatorially.
    pub fn new(constraints: &ConstraintMatrix) -> Self {
        let n = constraints.n_conditions;
        let vertices = enumerate_vertices_combinatorial(constraints);
        let n_vertices = vertices.len();

        // Flatten into contiguous memory for cache-friendly linear scans
        let mut vertices_flat = Vec::with_capacity(n_vertices * n);
        for v in &vertices {
            vertices_flat.extend_from_slice(v);
        }

        Self {
            vertices_flat,
            n_vertices,
            n,
        }
    }

    /// Solve the LMO: find the vertex minimizing gradient^T z.
    ///
    /// Linear scan over pre-computed vertex table.
    /// Cost: O(V * n) ~ a few hundred nanoseconds for typical problems.
    #[inline]
    pub fn solve(&self, gradient: &[f64]) -> (Vec<f64>, f64) {
        let n = self.n;
        let mut best_obj = f64::INFINITY;
        let mut best_idx = 0;

        for v in 0..self.n_vertices {
            let base = v * n;
            let mut obj = 0.0;
            for i in 0..n {
                obj += gradient[i] * self.vertices_flat[base + i];
            }
            if obj < best_obj {
                best_obj = obj;
                best_idx = v;
            }
        }

        if self.n_vertices == 0 {
            return (vec![0.5; n], gradient.iter().sum::<f64>() * 0.5);
        }

        let base = best_idx * n;
        let z = self.vertices_flat[base..base + n].to_vec();
        (z, best_obj)
    }

    /// Compute the centroid of all vertices.
    pub fn compute_centroid(&self) -> Vec<f64> {
        let n = self.n;
        if self.n_vertices == 0 {
            return vec![0.5; n];
        }

        let mut centroid = vec![0.0; n];
        for v in 0..self.n_vertices {
            let base = v * n;
            for i in 0..n {
                centroid[i] += self.vertices_flat[base + i];
            }
        }
        let nv = self.n_vertices as f64;
        for c in centroid.iter_mut() {
            *c /= nv;
        }
        centroid
    }

    /// Number of vertices found.
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }
}

/// Build constraints from a relationship graph.
pub fn build_constraints_from_graph(
    market_ids: &[String],
    relationships: &crate::types::RelationshipGraph,
    market_outcomes: Option<&std::collections::HashMap<String, Vec<String>>>,
) -> ConstraintMatrix {
    let space = ConditionSpace::from_market_data(market_ids, market_outcomes);
    let mut builder = MarginalConstraintBuilder::new(space);
    let rels = relationships.get_relationships(market_ids);
    builder.add_relationships(&rels);
    builder.build()
}
