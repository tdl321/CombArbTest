# Roadmap: Combinatorial Arbitrage Backtester

**Version:** 1.2
**Phases:** 6
**Requirements:** 26
**Status:** Complete

## Overview

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Data Layer | Load and query Polymarket dataset | DATA-01 to DATA-06 | ● Complete |
| 2 | LLM Analysis | Semantic clustering and dependency extraction | LLM-01 to LLM-04 | ● Complete |
| 3 | Optimization Engine | Frank-Wolfe solver for arbitrage-free prices | OPT-01 to OPT-05 | ● Complete |
| 4 | Backtester | Historical simulation and PnL reporting | BT-01 to BT-07 | ● Complete |
| 5 | Comprehensive Logging | Structured logging across all components | LOG-01 to LOG-04 | ● Complete |
| 6 | Visualization | Simplex geometry and Bregman projection plots | VIZ-01 to VIZ-04 | ● Complete |

---

## Phase 1: Data Layer ✅

**Goal:** Load and query 51GB Polymarket parquet dataset efficiently with point-in-time access.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06

**Implemented:**
- `src/data/loader.py` - DuckDB-based MarketLoader, TradeLoader, BlockLoader
- `src/data/models.py` - Pydantic models for Market, Trade, BlockTimestamp
- `src/data/price_series.py` - PriceSeriesBuilder with configurable resolution
- `src/data/tick_stream.py` - TickStream, CrossMarketIterator for trade-by-trade iteration

---

## Phase 2: LLM Analysis ✅

**Goal:** Use Kimi 2.5 to identify related markets and extract logical dependencies that create mathematical constraints for arbitrage detection.

**Requirements:** LLM-01, LLM-02, LLM-03, LLM-04

### Core Logical Constraints

The LLM must identify relationships that fall into these formal constraint categories:

#### 1. Mutual Exclusivity
**Definition:** Only one outcome within a set can be true. If market A resolves YES, market B MUST resolve NO.
**Mathematical constraint:** `P(A) + P(B) ≤ 1` (pairwise), or `Σ P(outcomes) ≤ 1` (sets)
**Examples:**
- Trump wins election and Harris wins election - only one candidate can win
- Lakers win NBA Finals and Celtics win NBA Finals - only one team can win
- Fed raises rates and Fed cuts rates at the same meeting

#### 2. Exhaustiveness
**Definition:** At least one outcome in a set MUST occur. The outcomes collectively cover all possibilities.
**Mathematical constraint:** `Σ P(outcomes) ≥ 1` (combined with mutual exclusivity gives `Σ P = 1`)
**Examples:**
- Binary markets: YES or NO must resolve true
- Republican wins OR Democrat wins OR Third party wins - someone must win
- Fed raises, Fed holds, Fed cuts - one must happen

#### 3. Logical Implication
**Definition:** The truth of one outcome is a prerequisite for another. If A is true, B must also be true.
**Mathematical constraint:** `P(A) ≤ P(B)`, equivalently: `A → B` means `¬B → ¬A`
**Examples:**
- Trump wins Pennsylvania IMPLIES Trump wins at least one swing state
- Team wins Finals IMPLIES Team made playoffs
- Bill becomes law IMPLIES Bill passed Congress

#### 4. Incompatibility
**Definition:** Certain outcomes cannot coexist even if not direct competitors. Structural impossibility.
**Mathematical constraint:** `P(A ∧ B) = 0`, so `P(A) + P(B) ≤ 1`
**Examples:**
- Lakers vs Celtics Finals and Lakers vs Warriors Finals - can't have two different matchups
- Trump wins popular vote and Harris wins by landslide - contradictory scenarios
- Two teams from same conference both making Finals

### Relationship Types

| Type | Constraint | Mathematical Form |
|------|------------|-------------------|
| `mutually_exclusive` | Only one can be true | `P(A) + P(B) ≤ 1` |
| `exhaustive` | At least one must be true | `Σ P ≥ 1` |
| `implies` | A requires B | `P(A) ≤ P(B)` |
| `prerequisite` | Temporal/causal dependency | `P(B) ≤ P(A)` |
| `incompatible` | Cannot coexist | `P(A ∧ B) = 0` |
| `and` | Correlated outcomes | Strong positive correlation |
| `or` | Disjunction | `P(A ∨ B) ≥ threshold` |

**Implemented:**
- `src/llm/client.py` - OpenRouter client for Kimi 2.5 with robust JSON extraction
- `src/llm/clustering.py` - MarketClusterer for semantic grouping with retry logic
- `src/llm/extractor.py` - RelationshipExtractor with formal constraint definitions
- `src/llm/schema.py` - MarketInfo, MarketCluster, MarketRelationship, RelationshipGraph

---

## Phase 3: Optimization Engine ✅

**Goal:** Implement Frank-Wolfe algorithm to find arbitrage-free prices given logical constraints.

**Requirements:** OPT-01, OPT-02, OPT-03, OPT-04, OPT-05

**Constraint Implementation:**
- Mutual exclusivity/incompatibility → Linear constraints: `p_i + p_j ≤ 1`
- Exhaustiveness → Linear constraints: `Σ p_i ≥ 1`
- Implication → Linear constraints: `p_from ≤ p_to`
- Prerequisite → Linear constraints: `p_to ≤ p_from`

**Implemented:**
- `src/optimizer/frank_wolfe.py` - frank_wolfe(), barrier_frank_wolfe(), projected_gradient_descent()
- `src/optimizer/lmo.py` - LinearMinimizationOracle with HiGHS, ConstraintBuilder
- `src/optimizer/divergence.py` - kl_divergence(), kl_gradient(), compute_duality_gap(), line_search_kl()
- `src/optimizer/schema.py` - ArbitrageResult, OptimizationConfig, ConstraintViolation

---

## Phase 4: Backtester ✅

**Goal:** Run historical simulation and report theoretical PnL.

**Requirements:** BT-01, BT-02, BT-03, BT-04, BT-05, BT-06, BT-07

**Implemented:**
- `src/backtest/runner.py` - run_backtest() entry point
- `src/backtest/simulator.py` - WalkForwardSimulator with optimized constraint checking
  - Simple 2-market constraints checked algebraically (fast path)
  - Partition constraints (exhaustive + exclusive) checked separately
  - Complex multi-market constraints use Frank-Wolfe solver
- `src/backtest/pnl.py` - PnLTracker, calculate_opportunity_pnl()
- `src/backtest/report.py` - generate_report(), format_report()
- `src/backtest/schema.py` - ArbitrageOpportunity, BacktestConfig, BacktestReport
- `src/backtest/constraint_checker.py` - Algebraic constraint checking
  - SimpleViolation: implies, prerequisite, mutually_exclusive, incompatible
  - PartitionViolation: sum ≠ 1 for exhaustive+exclusive sets
  - is_simple_constraint(), is_complex_constraint(), check_partition()
- `src/backtest/signal_report.py` - generate_signal_report() for pure arbitrage proofs

---

## Phase 5: Comprehensive Logging ✅

**Goal:** Add structured logging across all components for debugging, monitoring, and analysis.

**Requirements:** LOG-01, LOG-02, LOG-03, LOG-04

**Implemented:**
- `src/logging_config.py` - Centralized logging configuration
  - PrefixFormatter: Consistent timestamp + prefix formatting
  - PrefixLoggerAdapter: Automatic prefix injection per module
  - MODULE_PREFIXES: DATA, LLM, CLUSTER, EXTRACT, OPT, FW, LMO, KL, BACKTEST, SIM, PNL, REPORT
  - TimingContext: Performance timing with automatic logging
  - setup_logging(): Application-wide configuration
  - get_logger(): Per-module logger retrieval
  - timed(): Decorator for timing operations

---

## Phase 6: Visualization ✅

**Goal:** Visualize arbitrage signals using simplex geometry and Bregman projections.

**Requirements:** VIZ-01, VIZ-02, VIZ-03, VIZ-04

**Implemented:**
- `src/visualization/schema.py` - ArbitrageSignal dataclass
  - Pure signal representation (no PnL coupling)
  - market_prices, coherent_prices, edge_magnitude, kl_divergence
  - direction dict (positive=buy, negative=sell)
  - constraint_violation description
- `src/visualization/simplex.py` - SimplexProjector
  - N-dimensional probability → 2D via barycentric coordinates
  - Handles 2, 3, 4, N-market cases with appropriate geometry
  - is_feasible(), distance_to_simplex(), get_simplex_boundary()
- `src/visualization/bregman_plot.py` - Bregman projection visualization
  - BregmanAnalysis: cluster analysis with overround, KL divergence
  - plot_bregman_dual_panel(): Price correction bars + simplex radar
  - plot_single_cluster_summary(): Three-panel with trade table
  - plot_bregman_report(): Multi-page report generation
- `src/visualization/signal_plot.py` - Arbitrage signal plotting
  - plot_arbitrage_signal(): Single signal on simplex
  - plot_signal_batch(): Grid of multiple signals
  - Market vs coherent points with edge vectors

---

## Milestone: v1.0 ✅

**Target:** Working backtester that identifies historical arbitrage opportunities

**Deliverables:**
- [x] Python package with CLI entry points
- [x] Backtest runner with configurable parameters
- [x] Support for LLM-based relationship discovery
- [x] Frank-Wolfe optimization with HiGHS solver
- [x] PnL reporting with transaction costs

---

## Milestone: v1.1 ✅

**Target:** Enhanced constraint formalization and robustness

**Deliverables:**
- [x] Formal logical constraint definitions in LLM prompts
- [x] Extended relationship types (exhaustive, incompatible)
- [x] Robust JSON extraction with retry logic
- [x] Mathematical constraint documentation

---

## Milestone: v1.2 ✅

**Target:** Observability, debugging, and visualization

**Deliverables:**
- [x] Structured logging with consistent format
- [x] Component-level log prefixes
- [x] Performance timing logs
- [x] Simplex geometry visualization
- [x] Bregman projection dual-panel plots
- [x] Optimized algebraic constraint checking
- [x] Partition violation detection
- [x] Pure signal reports (arbitrage proof)

---

## Future Considerations

Potential enhancements for v1.3+:
- Real-time signal detection (live market monitoring)
- Position sizing and Kelly criterion integration
- Multi-exchange arbitrage (cross-platform)
- Web dashboard for signal visualization
- Automated trade execution

---
*Roadmap created: 2025-02-24*
*v1.0 Completed: 2025-02-25*
*v1.1 Completed: 2025-02-25*
*v1.2 Completed: 2025-02-25*
