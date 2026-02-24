# Combinatorial Arbitrage Backtester

## What This Is

A backtesting system for combinatorial arbitrage strategies on Polymarket. It analyzes historical prediction market data to find mispricings across interdependent events (e.g., championship wins dependent on bracket progression), validates opportunities using LLM semantic filtering, and calculates theoretical profit using Frank-Wolfe convex optimization.

## Core Value

Identify historically profitable arbitrage opportunities by detecting when market-implied probabilities violate logical constraints across related events.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Load and query Polymarket historical data (51GB parquet dataset)
- [ ] Identify related markets using semantic similarity (LLM-based)
- [ ] Detect logical dependencies between markets (AND, OR, prerequisite relationships)
- [ ] Calculate arbitrage-free prices using Frank-Wolfe optimization
- [ ] Compute KL divergence between market prices and coherent prices
- [ ] Backtest strategies against historical price movements
- [ ] Report theoretical PnL and opportunity statistics

### Out of Scope

- Live trading execution — this is a backtest-only system
- Real-time data streaming — using historical dataset
- Rust execution engine — Python prototype only for now
- Order book simulation — using market prices, not CLOB depth

## Context

**Dataset**: Jon-Becker/prediction-market-analysis
- 51GB Polymarket data (markets + trades)
- 68,646 trade parquet files
- Markets from 2020 onwards (COVID, elections, crypto, sports)
- Schema: market metadata, trade history with block numbers

**Mathematical Framework** (from PRD):
- Frank-Wolfe algorithm for convex optimization over marginal polytope
- KL divergence as profit metric
- Linear Minimization Oracle using HiGHS solver
- Barrier FW variant for numerical stability

**LLM Integration**:
- Semantic re-ranking of statistically-discovered pairs
- Dependency extraction (market A requires market B)
- Flexible backend (OpenAI, Anthropic, local Ollama)

## Constraints

- **Infrastructure**: VPS at 76.13.197.7, Python 3.12, ~100GB storage
- **Data**: Historical only, no API keys for live Polymarket data
- **Compute**: No GPU, CPU-only optimization

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Python-first | Rapid prototyping, existing data science ecosystem | — Pending |
| Polars for data | Faster than pandas for large parquet files | — Pending |
| Flexible LLM backend | Test multiple models, avoid vendor lock-in | — Pending |
| Backtest before live | Validate strategy before risking capital | — Pending |

---
*Last updated: 2025-02-24 after initialization*
