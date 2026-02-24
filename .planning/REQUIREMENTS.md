# Requirements: Combinatorial Arbitrage Backtester

**Defined:** 2025-02-24
**Core Value:** Identify historically profitable arbitrage opportunities by detecting when market-implied probabilities violate logical constraints across related events.

## v1 Requirements

### Data Layer

- [x] **DATA-01**: Load Polymarket market metadata from parquet files (id, question, outcomes, prices)
- [x] **DATA-02**: Load trade history from parquet files (block_number, amounts, timestamps)
- [x] **DATA-03**: Build price time series from trade data with configurable resolution
- [x] **DATA-04**: Enforce point-in-time data access (no lookahead bias)
- [x] **DATA-05**: Query markets by date range, volume, and status filters

### LLM Analysis

- [x] **LLM-01**: Connect to Deepseek or Kimi 2.5 API via LiteLLM
- [x] **LLM-02**: Identify semantically related markets from questions/slugs
- [x] **LLM-03**: Extract logical dependencies (AND, OR, prerequisite, mutually exclusive)
- [x] **LLM-04**: Output structured relationship graph (JSON with market IDs and relationship types)

### Optimization

- [x] **OPT-01**: Implement Frank-Wolfe solver for marginal polytope constraints
- [x] **OPT-02**: Build Linear Minimization Oracle using HiGHS
- [x] **OPT-03**: Implement Barrier Frank-Wolfe variant for numerical stability
- [x] **OPT-04**: Calculate KL divergence between market prices and coherent prices
- [x] **OPT-05**: InitFW: Find valid interior starting point via centroid calculation

### Backtesting

- [x] **BT-01**: Walk-forward simulation through historical data
- [x] **BT-02**: Detect arbitrage opportunities when KL divergence exceeds threshold
- [x] **BT-03**: Calculate theoretical PnL accounting for transaction costs (~1-2%)
- [x] **BT-04**: Generate summary report (total opportunities, PnL distribution, win rate)

## v2 Requirements

### Enhanced Analysis
- **LLM-05**: Multi-leg arbitrage path discovery (3+ markets)
- **LLM-06**: Confidence scoring for relationship classifications

### Enhanced Backtesting
- **BT-05**: Liquidity-adjusted profit estimates
- **BT-06**: Visualization of opportunity timelines
- **BT-07**: Export trade signals to CSV

### Infrastructure
- **INFRA-01**: Caching layer for LLM responses
- **INFRA-02**: Incremental data loading (resume from checkpoint)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live trading execution | Backtest-only system, no capital risk |
| Order book simulation | Dataset lacks full depth, excessive complexity |
| Granger causality | LLM handles semantic detection directly |
| Real-time streaming | Using historical dataset |
| Multi-exchange arbitrage | Focus on Polymarket only |
| Portfolio optimization | Separate concern from opportunity detection |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| LLM-01 | Phase 2 | Complete |
| LLM-02 | Phase 2 | Complete |
| LLM-03 | Phase 2 | Complete |
| LLM-04 | Phase 2 | Complete |
| OPT-01 | Phase 3 | Complete |
| OPT-02 | Phase 3 | Complete |
| OPT-03 | Phase 3 | Complete |
| OPT-04 | Phase 3 | Complete |
| OPT-05 | Phase 3 | Complete |
| BT-01 | Phase 4 | Complete |
| BT-02 | Phase 4 | Complete |
| BT-03 | Phase 4 | Complete |
| BT-04 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2025-02-24*
*Last updated: 2025-02-24 after initial definition*
