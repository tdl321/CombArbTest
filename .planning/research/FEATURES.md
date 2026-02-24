# Features Research

## Table Stakes (Must Have)

### Data Layer
- **Parquet data loading** — Load 50GB+ historical market/trade data efficiently — Complexity: Medium
- **Market metadata access** — Query market questions, outcomes, prices, dates — Complexity: Low
- **Time series construction** — Build price time series from trade data — Complexity: Medium

### Analysis Layer
- **Market relationship detection** — Identify semantically related markets (championship → bracket) — Complexity: High
- **Dependency extraction** — Classify relationships: AND, OR, prerequisite, mutually exclusive — Complexity: High
- **Probability coherence check** — Detect when market prices violate logical constraints — Complexity: Medium

### Optimization Layer
- **Arbitrage-free price calculation** — Frank-Wolfe solver to find coherent price vector — Complexity: High
- **Profit metric computation** — KL divergence between market and coherent prices — Complexity: Medium
- **Feasibility validation** — Ensure solutions within marginal polytope — Complexity: Medium

### Backtesting Layer
- **Historical simulation** — Walk through time, detect opportunities as they appear — Complexity: Medium
- **PnL calculation** — Theoretical profit accounting for detected opportunities — Complexity: Low
- **Results reporting** — Summary statistics, opportunity distribution — Complexity: Low

## Differentiators (Competitive Advantage)

- **LLM semantic filtering** — Filter spurious statistical correlations using reasoning — Complexity: Medium
- **Granger causality pre-screening** — Statistical lead-lag detection before LLM — Complexity: Medium
- **Multi-leg arbitrage paths** — Handle 3+ market arbitrage cycles — Complexity: High
- **Confidence intervals** — Uncertainty quantification on profit estimates — Complexity: Medium
- **Opportunity visualization** — Charts showing price divergence over time — Complexity: Low

## Anti-Features (Don't Build)

| Feature | Why Avoid |
|---------|-----------|
| **Live execution** | Risk of capital loss, regulatory concerns, out of scope |
| **Order book simulation** | Massive complexity, dataset may not have full depth |
| **Real-time streaming** | Using historical data only |
| **Portfolio optimization** | Separate concern from opportunity detection |
| **Risk management / position sizing** | Execution-phase concern, not backtest |
| **Multi-exchange arbitrage** | Focus on Polymarket only for v1 |

## Feature Dependencies

```
Data Loading
    └── Time Series Construction
            └── Granger Causality (statistical screening)
                    └── LLM Semantic Filtering
                            └── Dependency Extraction
                                    └── Coherence Check
                                            └── Frank-Wolfe Solver
                                                    └── PnL Calculation
                                                            └── Reporting
```

## Priority Matrix

| Feature | Value | Complexity | Priority |
|---------|-------|------------|----------|
| Data loading | High | Medium | P0 |
| Time series | High | Medium | P0 |
| Frank-Wolfe solver | High | High | P0 |
| Market clustering | High | Medium | P1 |
| LLM filtering | Medium | Medium | P1 |
| Granger causality | Medium | Medium | P2 |
| Visualization | Low | Low | P2 |
