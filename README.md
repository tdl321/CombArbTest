# CombArbTest

Combinatorial arbitrage detection for Polymarket prediction markets. Uses LLM-discovered logical constraints and Frank-Wolfe optimization over the marginal polytope to find mispricing across related markets.

## What It Does

Prediction markets on Polymarket have logical relationships that constrain their prices. For example, if "Driver X wins the championship" is true, then "Driver X's team wins the constructors' championship" must also be plausible. When market prices violate these constraints, there is arbitrage.

This system:

1. **Fetches live markets** from Polymarket via the Gamma (metadata) and CLOB (prices) APIs
2. **Discovers logical constraints** between markets using an LLM (Kimi K2 via OpenRouter) with domain-specific prompts for F1, NFL, NBA, and World Cup
3. **Projects market prices** onto the nearest coherent distribution inside the marginal polytope using a Barrier Frank-Wolfe solver
4. **Extracts executable trades** from constraint violations (implies, mutex, binary) with profit/fee calculations
5. **Visualizes mispricing** via Bregman projection plots (correction bars, simplex radar, trade recommendation tables)

## Architecture

```
src/
  core/           # Canonical types (MarketMeta, Opportunity, Constraint) and protocols
  data/
    polymarket/   # Gamma API client, CLOB API client, dataset builder, relationship inference
  llm/            # OpenRouter client, two-stage market clustering, constraint extraction
    prompts/      # Domain-specific constraint prompts (F1, NFL, NBA, World Cup)
  optimizer/      # Frank-Wolfe solver, KL divergence, Linear Minimization Oracle (HiGHS MILP)
  arbitrage/      # Trade extractor: constraint violations -> executable trades
  strategies/     # Strategy wrappers and registry (combinatorial_arb)
  backtest/       # Backtest engine (single-point instant arb + time-stepped held positions)
  grouping/       # Category-based and LLM-based market grouping adapters
  visualization/  # Bregman projection plots, simplex charts, signal plots
  pipeline.py     # Pipeline orchestrator coordinating strategies, groupers, data sources
  config.py       # Environment config, secret management
docs/             # Alpha research: BTC monotonicity, implied distribution, term structure
tests/            # Unit + integration tests for all modules
output/           # Generated visualization PNGs
```

## Key Components

### Solver (Frank-Wolfe over Marginal Polytope)

The optimizer finds the distribution closest to market prices (in KL divergence) that satisfies all logical constraints. The marginal polytope is the set of all valid joint outcome probability vectors.

- **Frank-Wolfe algorithm** with barrier variant to stay in the polytope interior
- **Linear Minimization Oracle (LMO)** using HiGHS MILP for large problems, combinatorial vertex enumeration for small ones (up to 20 markets)
- Step size modes: adaptive (local smoothness estimation), exact line search, fixed
- Convergence: duality gap tolerance, configurable iteration limits

### LLM Constraint Discovery

Two-stage pipeline:
1. **Semantic clustering** -- groups related markets by theme
2. **Constraint extraction** -- identifies logical relationships within clusters

Supports both partition constraints (mutually exclusive + exhaustive sets where prices must sum to 1) and complex constraints (implies, prerequisite, incompatible) that require the solver.

Domain-specific prompts encode structural knowledge (e.g., F1 team-driver pairings, championship relationships) to improve constraint accuracy.

### Data Layer (Polymarket)

- **Gamma API** (public, no auth): Event/market discovery, metadata, search
- **CLOB API** (public read): Live midpoint prices, orderbooks, price history
- **Dataset builder**: Declarative specs -> resolved datasets with inferred relationships
- **Relationship inference**: Auto-detects partitions from Polymarket's `negRisk` flag; falls back to LLM for non-negRisk events

### Trade Extraction

Converts constraint violations into executable trades with profit calculations:

| Constraint Type | Violation | Trade |
|----------------|-----------|-------|
| Binary (YES+NO=1) | Sum != 1 | Buy both (sum < 1) or sell both (sum > 1) |
| Implies (A -> B) | P(A) > P(B) | Sell A, buy B |
| Mutex (A XOR B) | P(A)+P(B) > 1 | Sell both |

Profit is the violation magnitude minus transaction fees.

## Running

### Prerequisites

- Python 3.11+
- OpenRouter API key (for LLM constraint discovery)

### Setup

```bash
pip install -e .
cp .env.example .env  # Add your OPENROUTER_API_KEY
chmod 600 .env
```

### F1 Cross-Event Arbitrage Test

The main demo fetches live F1 drivers and constructors championship markets, discovers cross-event constraints via LLM, runs the solver, and outputs trades:

```bash
python run_f1_solver_test.py
```

### Tests

```bash
pytest tests/ -v
```

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Optimization | numpy, scipy, cvxpy, highspy (HiGHS MILP) |
| Data | polars, duckdb, pyarrow |
| LLM | litellm, instructor, httpx (OpenRouter / Kimi K2) |
| Schema | pydantic |
| Visualization | matplotlib |
| Config | python-dotenv |
| Build | hatchling |

## Alpha Research

The `docs/` directory contains analysis of three alpha sources in Polymarket BTC "above $X" binary markets:

1. **Monotonicity Violations** (Alpha 1): Cross-strike same-date violations in BTC ladders. Verdict: low-yield, not worth standalone system.
2. **Implied Distribution Arbitrage** (Alpha 2): Negative densities in strike ladders imply risk-free butterfly spreads. Requires live monitoring.
3. **Term Structure Inversions** (Alpha 3): Same-strike cross-date violations with 45.8% inversion rate and 19.2c average magnitude. Strongest signal found.

## Project Status

- Solver: Fully implemented (Frank-Wolfe with barrier, adaptive step sizes, combinatorial vertex enumeration)
- LLM constraint extraction: Working with domain-specific prompts (F1, NFL, NBA, World Cup)
- Polymarket data layer: Live API clients for Gamma and CLOB
- Backtest engine: Single-point and time-stepped evaluators
- Visualization: Bregman projection plots operational
- Trade extraction: Binary, implies, and mutex violations -> executable trades
- Pipeline: Modular orchestrator with strategy registry
