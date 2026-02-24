# Research Summary

## Key Findings

### Stack
**DuckDB + Polars** for data (DuckDB handles 50GB+ reliably), **CVXPY + HiGHS** for optimization, **LiteLLM** for flexible LLM backend. No off-the-shelf Frank-Wolfe library — implement from scratch using HiGHS for the linear minimization oracle.

### Table Stakes
1. Load and query 50GB parquet dataset efficiently
2. Build price time series from trade data
3. Detect logically related markets
4. Frank-Wolfe solver for coherent prices
5. KL divergence profit calculation
6. Walk-forward backtesting with PnL reporting

### Watch Out For
- **Lookahead bias** — strict point-in-time data access
- **Transaction costs** — 1-2% fees eat small opportunities
- **Liquidity** — can't assume mid-price execution
- **Numerical stability** — boundary issues in Frank-Wolfe (use Barrier variant)

## Architecture Overview

```
Data Loading → Statistical Screening → LLM Filtering → Optimization → Backtesting
     │                  │                   │              │              │
   DuckDB          Granger            LiteLLM        Frank-Wolfe     Walk-forward
   Polars          causality          Instructor       HiGHS         PnL calc
```

## Recommended Build Order

| Phase | Focus | Key Deliverable |
|-------|-------|-----------------|
| 1 | Data Layer | Load markets/trades, build price series |
| 2 | Frank-Wolfe Solver | Optimization engine with test cases |
| 3 | Market Clustering | Semantic grouping + LLM integration |
| 4 | Backtester | Historical simulation, PnL reporting |

## Critical Success Factors

1. **Data integrity** — No lookahead, proper timestamps
2. **Solver correctness** — Frank-Wolfe must converge to true optimum
3. **Realistic assumptions** — Fees, slippage, liquidity
4. **Validation** — Out-of-sample testing, sanity checks

## Files

- [STACK.md](./STACK.md) — Technology recommendations
- [FEATURES.md](./FEATURES.md) — Feature prioritization
- [ARCHITECTURE.md](./ARCHITECTURE.md) — System design
- [PITFALLS.md](./PITFALLS.md) — Common mistakes to avoid
