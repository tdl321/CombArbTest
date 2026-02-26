//! combarbbot Rust solver — drop-in replacement for the Python optimizer.
//!
//! Exposes `find_marginal_arbitrage` as a PyO3 function that accepts
//! the same arguments as the Python version and returns identical results.
//!
//! All computation runs with the GIL released via `py.allow_threads()`.

mod divergence;
mod frank_wolfe;
mod lmo;
mod types;

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use types::{
    MarketCluster, MarketRelationship, OptimizationConfig, RelationshipGraph, StepMode,
};

/// Convert Python market_prices dict[str, list[float]] to Rust HashMap.
fn extract_market_prices(py_dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, Vec<f64>>> {
    let mut prices = HashMap::new();
    for (key, value) in py_dict.iter() {
        let market_id: String = key.extract()?;
        let price_list: Vec<f64> = value.extract()?;
        prices.insert(market_id, price_list);
    }
    Ok(prices)
}

/// Convert Python relationship graph to Rust RelationshipGraph.
///
/// Accepts the graph as a JSON string (from `graph.model_dump_json()`)
/// or as a Python dict structure.
fn extract_relationship_graph(obj: &Bound<'_, PyAny>) -> PyResult<RelationshipGraph> {
    // Try as JSON string first
    if let Ok(json_str) = obj.extract::<String>() {
        let graph: RelationshipGraph = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse JSON: {}", e)))?;
        return Ok(graph);
    }

    // Try extracting as a Python object with clusters attribute
    let clusters_obj = obj.getattr("clusters")?;
    let clusters_list: &Bound<'_, PyList> = clusters_obj.downcast()?;

    let mut clusters = Vec::new();
    for cluster_obj in clusters_list.iter() {
        let cluster_id: String = cluster_obj.getattr("cluster_id")?.extract()?;
        let market_ids: Vec<String> = cluster_obj.getattr("market_ids")?.extract()?;

        let rels_obj = cluster_obj.getattr("relationships")?;
        let rels_list: &Bound<'_, PyList> = rels_obj.downcast()?;

        let mut relationships = Vec::new();
        for rel_obj in rels_list.iter() {
            // The type field might be a string or an enum
            let rel_type: String = {
                let type_obj = rel_obj.getattr("type")?;
                // Try getting .value for enums
                if let Ok(val) = type_obj.getattr("value") {
                    val.extract()?
                } else {
                    type_obj.extract()?
                }
            };

            let from_market: String = rel_obj.getattr("from_market")?.extract()?;
            let to_market: Option<String> = rel_obj.getattr("to_market")?.extract()?;
            let confidence: f64 = rel_obj
                .getattr("confidence")
                .and_then(|c| c.extract())
                .unwrap_or(1.0);

            relationships.push(MarketRelationship {
                rel_type,
                from_market,
                to_market,
                confidence,
            });
        }

        clusters.push(MarketCluster {
            cluster_id,
            market_ids,
            relationships,
        });
    }

    Ok(RelationshipGraph { clusters })
}

/// Extract OptimizationConfig from Python object or use defaults.
fn extract_config(obj: Option<&Bound<'_, PyAny>>) -> PyResult<OptimizationConfig> {
    let mut config = OptimizationConfig::default();

    if let Some(py_config) = obj {
        if let Ok(v) = py_config.getattr("max_iterations") {
            config.max_iterations = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("tolerance") {
            config.tolerance = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("epsilon_init") {
            config.epsilon_init = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("epsilon_min") {
            config.epsilon_min = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("epsilon_decay") {
            config.epsilon_decay = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("step_mode") {
            let mode_str: String = v.extract()?;
            config.step_mode = match mode_str.as_str() {
                "adaptive" => StepMode::Adaptive,
                "line_search" => StepMode::LineSearch,
                "fixed" => {
                    let step = py_config
                        .getattr("fixed_step_size")
                        .and_then(|s| s.extract())
                        .unwrap_or(0.5);
                    StepMode::Fixed(step)
                }
                _ => StepMode::LineSearch,
            };
        }
        if let Ok(v) = py_config.getattr("smoothness_alpha") {
            config.smoothness_alpha = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("initial_smoothness") {
            config.initial_smoothness = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("min_smoothness") {
            config.min_smoothness = v.extract()?;
        }
        if let Ok(v) = py_config.getattr("fixed_step_size") {
            config.fixed_step_size = v.extract()?;
        }
    }

    Ok(config)
}

/// Find marginal arbitrage — drop-in replacement for the Python version.
///
/// This is the main entry point. It:
/// 1. Converts Python types to Rust
/// 2. Releases the GIL
/// 3. Runs the Frank-Wolfe solver in pure Rust
/// 4. Converts results back to Python
///
/// Args:
///     market_prices: dict[str, list[float]] — market prices
///     relationships: RelationshipGraph object or JSON string
///     market_outcomes: Optional dict[str, list[str]] — outcome names
///     config: Optional OptimizationConfig object
///
/// Returns:
///     dict with solver results (same structure as MarginalArbitrageResult)
#[pyfunction]
#[pyo3(signature = (market_prices, relationships, market_outcomes=None, config=None))]
fn find_marginal_arbitrage(
    py: Python<'_>,
    market_prices: &Bound<'_, PyDict>,
    relationships: &Bound<'_, PyAny>,
    market_outcomes: Option<&Bound<'_, PyDict>>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    // Extract all Python data before releasing GIL
    let prices = extract_market_prices(market_prices)?;
    let graph = extract_relationship_graph(relationships)?;
    let opt_config = extract_config(config)?;

    let outcomes: Option<HashMap<String, Vec<String>>> = match market_outcomes {
        Some(d) => {
            let mut map = HashMap::new();
            for (key, value) in d.iter() {
                let k: String = key.extract()?;
                let v: Vec<String> = value.extract()?;
                map.insert(k, v);
            }
            Some(map)
        }
        None => None,
    };

    // Release GIL and run solver in pure Rust
    let result = py.allow_threads(|| {
        let market_ids: Vec<String> = prices.keys().cloned().collect();
        let constraints = lmo::build_constraints_from_graph(
            &market_ids,
            &graph,
            outcomes.as_ref(),
        );
        frank_wolfe::marginal_frank_wolfe(&prices, &constraints, &opt_config)
    });

    // Convert result back to Python dict
    let dict = PyDict::new(py);
    
    // condition_prices
    let cp = PyDict::new(py);
    for (k, v) in &result.condition_prices {
        cp.set_item(k, v)?;
    }
    dict.set_item("condition_prices", cp)?;

    // coherent_condition_prices
    let ccp = PyDict::new(py);
    for (k, v) in &result.coherent_condition_prices {
        ccp.set_item(k, v)?;
    }
    dict.set_item("coherent_condition_prices", ccp)?;

    // market_prices
    let mp = PyDict::new(py);
    for (k, v) in &result.market_prices {
        mp.set_item(k, v.clone())?;
    }
    dict.set_item("market_prices", mp)?;

    // coherent_market_prices
    let cmp = PyDict::new(py);
    for (k, v) in &result.coherent_market_prices {
        cmp.set_item(k, v.clone())?;
    }
    dict.set_item("coherent_market_prices", cmp)?;

    // Scalar fields
    dict.set_item("kl_divergence", result.kl_divergence)?;
    dict.set_item("duality_gap", result.duality_gap)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("active_vertices", result.active_vertices)?;

    Ok(dict.into())
}

/// Simplified API for common constraint patterns (implies + mutex only).
///
/// Args:
///     market_prices: dict[str, list[float]]
///     implications: Optional list of (from_market, to_market) tuples
///     mutex_pairs: Optional list of (market_a, market_b) tuples
///     config: Optional OptimizationConfig object
///
/// Returns:
///     dict with solver results
#[pyfunction]
#[pyo3(signature = (market_prices, implications=None, mutex_pairs=None, config=None))]
fn detect_arbitrage_simple(
    py: Python<'_>,
    market_prices: &Bound<'_, PyDict>,
    implications: Option<Vec<(String, String)>>,
    mutex_pairs: Option<Vec<(String, String)>>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let prices = extract_market_prices(market_prices)?;
    let opt_config = extract_config(config)?;

    let result = py.allow_threads(|| {
        let market_ids: Vec<String> = prices.keys().cloned().collect();
        let space = types::ConditionSpace::from_market_data(&market_ids, None);
        let mut builder = lmo::MarginalConstraintBuilder::new(space);

        if let Some(ref impls) = implications {
            for (from_m, to_m) in impls {
                builder.add_implies(from_m, to_m);
            }
        }

        if let Some(ref mutexes) = mutex_pairs {
            for (m_a, m_b) in mutexes {
                builder.add_mutually_exclusive(m_a, m_b);
            }
        }

        let constraints = builder.build();
        frank_wolfe::marginal_frank_wolfe(&prices, &constraints, &opt_config)
    });

    // Convert to Python dict (same as above)
    let dict = PyDict::new(py);
    
    let cp = PyDict::new(py);
    for (k, v) in &result.condition_prices {
        cp.set_item(k, v)?;
    }
    dict.set_item("condition_prices", cp)?;

    let ccp = PyDict::new(py);
    for (k, v) in &result.coherent_condition_prices {
        ccp.set_item(k, v)?;
    }
    dict.set_item("coherent_condition_prices", ccp)?;

    let mp = PyDict::new(py);
    for (k, v) in &result.market_prices {
        mp.set_item(k, v.clone())?;
    }
    dict.set_item("market_prices", mp)?;

    let cmp = PyDict::new(py);
    for (k, v) in &result.coherent_market_prices {
        cmp.set_item(k, v.clone())?;
    }
    dict.set_item("coherent_market_prices", cmp)?;

    dict.set_item("kl_divergence", result.kl_divergence)?;
    dict.set_item("duality_gap", result.duality_gap)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("active_vertices", result.active_vertices)?;

    Ok(dict.into())
}

/// Python module definition.
#[pymodule]
fn solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_marginal_arbitrage, m)?)?;
    m.add_function(wrap_pyfunction!(detect_arbitrage_simple, m)?)?;
    Ok(())
}
