# Stack Research

## Recommended Stack

### Data Processing
- **DuckDB** (1.1+) — Best for 50GB+ parquet files. Automatic disk spilling, SQL interface, memory-efficient (~1.3GB for 140GB data vs Polars' 17GB). More reliable for production workloads.
- **Polars** (1.0+) — Use for DataFrame operations after DuckDB loads data. Fast for transformations, but memory-hungry for large files directly.
- **PyArrow** (17.0+) — Underlying parquet engine, required by both.

### Optimization
- **CVXPY** (1.5+) — High-level convex optimization modeling. Supports multiple solvers.
- **HiGHS** (via `highspy` 1.7+) — Fast open-source LP/MIP solver for Linear Minimization Oracle in Frank-Wolfe.
- **SciPy** (1.14+) — Fallback optimization routines, `scipy.optimize.linprog`.
- **NumPy** (2.0+) — Matrix operations for gradients.

Note: No mature Python Frank-Wolfe library exists. Implement manually using CVXPY/HiGHS for the LMO subproblem.

### LLM Integration
- **LiteLLM** (1.50+) — Lightweight, unified API for 100+ LLM providers. Easy model switching. Best for this use case (simple completions, not complex chains).
- **Instructor** (1.4+) — Structured output extraction with Pydantic validation. Pairs well with LiteLLM.

Avoid LangChain — over-engineered for simple semantic filtering tasks.

### Statistics / Time Series
- **statsmodels** (0.14+) — Granger causality tests, VAR models.
- **scipy.stats** — Statistical tests, distributions.

### Utilities
- **Pydantic** (2.9+) — Data validation, config management.
- **Rich** (13.0+) — CLI output formatting.
- **python-dotenv** (1.0+) — Environment variable management.

## Anti-Recommendations

| Library | Why Avoid |
|---------|-----------|
| **Pandas** | Too slow for 50GB+ data, memory inefficient |
| **LangChain** | Over-engineered for simple LLM calls, high complexity |
| **Polars (direct parquet)** | Memory issues at 50GB+ scale, use DuckDB instead |
| **PuLP** | Slower than HiGHS, less maintained |
| **GEKKO** | Overkill for linear subproblems |

## Confidence Notes

- DuckDB vs Polars: High confidence based on [recent benchmarks](https://www.codecentric.de/en/knowledge-hub/blog/duckdb-vs-polars-performance-and-memory-with-massive-parquet-data)
- Frank-Wolfe: No production Python library — must implement. Reference [FrankWolfe.jl](https://github.com/ZIB-IOL/FrankWolfe.jl) for algorithm structure.
- LiteLLM: Good for prototyping, may need to evaluate for production scale per [known issues](https://www.pomerium.com/blog/litellm-alternatives)

## Version Summary

```
duckdb>=1.1.0
polars>=1.0.0
pyarrow>=17.0.0
cvxpy>=1.5.0
highspy>=1.7.0
numpy>=2.0.0
scipy>=1.14.0
litellm>=1.50.0
instructor>=1.4.0
statsmodels>=0.14.0
pydantic>=2.9.0
```
