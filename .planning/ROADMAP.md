# Roadmap: Combinatorial Arbitrage Backtester

**Version:** 1.0
**Phases:** 4
**Requirements:** 18
**Status:** ✅ COMPLETE

## Overview

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Data Layer | Load and query Polymarket dataset | DATA-01 to DATA-06 | ● Complete |
| 2 | LLM Analysis | Semantic clustering and dependency extraction | LLM-01 to LLM-04 | ● Complete |
| 3 | Optimization Engine | Frank-Wolfe solver for arbitrage-free prices | OPT-01 to OPT-05 | ● Complete |
| 4 | Backtester | Historical simulation and PnL reporting | BT-01 to BT-04 | ● Complete |

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

**Goal:** Use Kimi 2.5 to identify related markets and extract logical dependencies.

**Requirements:** LLM-01, LLM-02, LLM-03, LLM-04

**Implemented:**
- `src/llm/client.py` - OpenRouter client for Kimi 2.5
- `src/llm/clustering.py` - MarketClusterer for semantic grouping
- `src/llm/extractor.py` - RelationshipExtractor for dependency types
- `src/llm/schema.py` - MarketInfo, MarketCluster, MarketRelationship, RelationshipGraph

**Relationship Types:** IMPLIES, MUTUALLY_EXCLUSIVE, AND, OR, PREREQUISITE

---

## Phase 3: Optimization Engine ✅

**Goal:** Implement Frank-Wolfe algorithm to find arbitrage-free prices given logical constraints.

**Requirements:** OPT-01, OPT-02, OPT-03, OPT-04, OPT-05

**Implemented:**
- `src/optimizer/frank_wolfe.py` - frank_wolfe(), barrier_frank_wolfe(), projected_gradient_descent()
- `src/optimizer/lmo.py` - LinearMinimizationOracle with HiGHS, ConstraintBuilder
- `src/optimizer/divergence.py` - kl_divergence(), kl_gradient(), compute_duality_gap(), line_search_kl()
- `src/optimizer/schema.py` - ArbitrageResult, OptimizationConfig, ConstraintViolation

---

## Phase 4: Backtester ✅

**Goal:** Run historical simulation and report theoretical PnL.

**Requirements:** BT-01, BT-02, BT-03, BT-04

**Implemented:**
- `src/backtest/runner.py` - run_backtest() entry point
- `src/backtest/simulator.py` - WalkForwardSimulator with tick-level iteration
- `src/backtest/pnl.py` - PnLTracker, calculate_opportunity_pnl()
- `src/backtest/report.py` - generate_report(), format_report()
- `src/backtest/schema.py` - ArbitrageOpportunity, BacktestConfig, BacktestReport

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
*Roadmap created: 2025-02-24*
*Completed: 2025-02-25*
