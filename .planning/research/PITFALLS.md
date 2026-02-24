# Pitfalls Research

## Critical Mistakes

### 1. Lookahead Bias
- **What goes wrong**: Using future information to make past decisions. E.g., knowing a market resolved to 1 when detecting the "arbitrage" opportunity.
- **Warning signs**: Unrealistically high Sharpe ratios, no losing periods
- **Prevention**: Strict point-in-time data access. Only use `_fetched_at` <= current_backtest_time.
- **Phase to address**: Phase 1 (Data Layer)

### 2. Survivorship Bias
- **What goes wrong**: Only analyzing markets that successfully resolved, ignoring cancelled/voided markets.
- **Warning signs**: Missing markets in dataset, gaps in time series
- **Prevention**: Include all markets regardless of outcome. Track `active`, `closed` status.
- **Phase to address**: Phase 1 (Data Layer)

### 3. Ignoring Transaction Costs
- **What goes wrong**: Reporting gross profit without fees. Polymarket charges ~1-2% maker/taker fees.
- **Warning signs**: Many small "profitable" opportunities that wouldn't survive fees
- **Prevention**: Model fees explicitly. Filter opportunities where `profit < fees + slippage`.
- **Phase to address**: Phase 4 (Backtesting)

### 4. Liquidity Assumption
- **What goes wrong**: Assuming you can trade at mid-price. Real execution has spread and market impact.
- **Warning signs**: Large theoretical profits on low-volume markets
- **Prevention**: Use volume/liquidity filters. Discount profits by estimated slippage.
- **Phase to address**: Phase 4 (Backtesting)

## Backtesting Pitfalls

| Pitfall | Description | Prevention |
|---------|-------------|------------|
| **Data snooping** | Testing many strategies, reporting best one | Pre-register hypotheses, out-of-sample validation |
| **Overfitting** | Tuning parameters to historical data | Walk-forward validation, parameter stability tests |
| **Selection bias** | Cherry-picking time periods | Full dataset analysis, multiple regime testing |
| **Market impact** | Large trades moving prices | Size-aware execution simulation |
| **Correlation breakdown** | Historical relationships not holding | Regime detection, recent-data weighting |

## Numerical Issues

### Frank-Wolfe Specific

| Issue | Description | Mitigation |
|-------|-------------|------------|
| **Boundary convergence** | Algorithm slows near polytope edges | Barrier FW variant with ε contraction |
| **Numerical instability** | Log(0) in KL divergence | Add small ε (1e-10) to probabilities |
| **Non-convergence** | Poor initialization | InitFW with interior point (centroid of vertices) |
| **Oscillation** | Step size too large | Adaptive step size, line search |
| **Infeasibility** | Constraints have no solution | IP solver to verify feasibility first |

### Linear Programming (LMO)

| Issue | Description | Mitigation |
|-------|-------------|------------|
| **Degeneracy** | Multiple optimal vertices | Solver handles internally (HiGHS) |
| **Ill-conditioning** | Near-parallel constraints | Constraint normalization, regularization |
| **Timeout** | Large constraint sets | Constraint pruning, warm starts |

## Domain-Specific Gotchas

### Prediction Market Quirks

| Gotcha | Description |
|--------|-------------|
| **Resolution timing** | Markets resolve at different times. A "prerequisite" market might not resolve before the dependent one. |
| **Partial resolution** | Some markets have multiple outcomes that resolve independently. |
| **Market amendments** | Rules can change mid-market (date extensions, clarifications). |
| **Correlated but not causal** | Two markets may move together without logical dependency. |
| **Arbitrage window** | Mispricings may exist for seconds, not long enough to trade. |

### Polymarket Specifics

| Gotcha | Description |
|--------|-------------|
| **USDC decimals** | Amounts in 6 decimals (1000000 = $1) |
| **Two contract types** | CTF Exchange vs NegRisk — different settlement mechanics |
| **Token IDs** | `clob_token_ids` map outcomes to tradeable tokens |
| **Outcome prices** | JSON string, not array — must parse |

## Risk Checklist

Before trusting backtest results:

- [ ] Verified no lookahead bias (point-in-time data only)
- [ ] Included transaction costs in profit calculation
- [ ] Applied realistic liquidity constraints
- [ ] Tested on out-of-sample period
- [ ] Checked for data quality issues (missing data, outliers)
- [ ] Verified logical constraints are correct
- [ ] Confirmed optimizer converges properly
- [ ] Sanity-checked PnL distribution (no implausible outliers)
