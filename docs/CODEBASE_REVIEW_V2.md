# CombArbBot Codebase Review v2

**Generated**: 2026-02-26
**Total Files Analyzed**: 40+ Python files across 7 modules

---

## Executive Summary

CombArbBot is a **combinatorial arbitrage detection system** for prediction markets (primarily Polymarket). The system detects arbitrage opportunities when related markets have prices that violate logical constraints (e.g., mutually exclusive outcomes summing to >100%).

### Core Pipeline
```
Market Data → LLM Clustering → Constraint Extraction → Frank-Wolfe Optimization → Trade Extraction → Backtest/Execution
```

### Key Technologies
- **Optimization**: Frank-Wolfe algorithm with KL divergence minimization
- **Constraint Solving**: Gurobi MILP for Linear Minimization Oracle (LMO)
- **LLM Integration**: OpenRouter API (Kimi 2.5 model) for market clustering
- **Data Layer**: DuckDB + Polars for parquet file processing
- **Visualization**: Matplotlib for simplex and Bregman plots

---

## Module Architecture

```
src/
├── optimizer/          # Core Frank-Wolfe optimization (4 files)
├── backtest/           # Walk-forward simulation (9 files)
├── data/               # Data loading layer (7 files)
├── llm/                # LLM-powered clustering (6 files)
├── visualization/      # Plotting utilities (5 files)
├── arbitrage/          # Trade extraction (2 files)
├── config.py           # Secure configuration
└── logging_config.py   # Centralized logging
```

---

## 1. Optimizer Module (`src/optimizer/`)

### Purpose
Implements **Frank-Wolfe optimization over the marginal polytope** to find coherent prices that minimize KL divergence from market prices.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `schema.py` | ~300 | Data models: ConditionSpace, RelationshipGraph, OptimizationConfig, ArbitrageResult |
| `divergence.py` | ~342 | KL divergence computation and line search |
| `lmo.py` | ~406 | Linear Minimization Oracle using Gurobi MILP |
| `frank_wolfe.py` | ~500 | Frank-Wolfe algorithm with adaptive step sizes |
| `__init__.py` | ~126 | Public API exports |

### Key Algorithms

**Frank-Wolfe Loop**:
```
1. Initialize mu from centroid (interior point)
2. For t = 1 to max_iterations:
   a. Compute gradient: ∇KL = -theta_i / mu_i
   b. LMO: Find vertex z* = argmin gradient^T z
   c. Duality gap: g = gradient^T (mu - z*)
   d. If gap < tolerance: CONVERGED
   e. Step size: γ (adaptive/line_search/fixed)
   f. Update: mu = mu + γ(z* - mu)
3. Return coherent prices mu
```

**Step Size Modes**:
- `adaptive`: γ = gap / (L * ||d||²) with smoothness estimation (3x faster)
- `line_search`: Golden section search (precise but slow)
- `fixed`: Constant step size

**Constraint Types**:
- `IMPLIES`: P(A) ≤ P(B)
- `MUTUALLY_EXCLUSIVE`: P(A) + P(B) ≤ 1
- `PARTITION`: sum(P) = 1

### Dependencies
- `numpy`, `pydantic`, `gurobipy`, `scipy.optimize`

### External Dependents
- `backtest/simulator.py` imports `find_arbitrage`, `OptimizationConfig`
- `arbitrage/extractor.py` imports `ArbitrageResult`, `ConstraintViolation`

---

## 2. Backtest Module (`src/backtest/`)

### Purpose
Walk-forward simulation for detecting arbitrage opportunities in historical data.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `schema.py` | ~230 | ArbitrageOpportunity, BacktestConfig, BacktestReport |
| `runner.py` | ~180 | Entry points: run_backtest(), run_backtest_with_synthetic_relationships() |
| `simulator.py` | ~800 | WalkForwardSimulator - core iteration engine |
| `constraint_checker.py` | ~200 | Partition constraint violation detection |
| `pnl.py` | ~250 | PnL calculation (deprecated + new methods) |
| `report.py` | ~200 | Report generation and formatting |
| `report_generator.py` | ~400 | New report system with visualizations |
| `signal_report.py` | ~100 | Signal-only reports |
| `__init__.py` | ~120 | Public API exports |

### Data Flow

```
run_backtest()
    ↓
MarketLoader, TradeLoader, BlockLoader
    ↓
WalkForwardSimulator.run()
    ↓ (for each tick)
├── check_partition() [fast, algebraic]
└── find_arbitrage() [solver fallback]
    ↓
ArbitrageOpportunity objects
    ↓
generate_report() → BacktestReport
```

### Detection Logic
1. **Partition Check** (fast): If `is_partition=True`, check `|sum(prices) - 1| > threshold`
2. **Solver Fallback** (complex): Use Frank-Wolfe optimizer for general constraints

### Key Configuration (`BacktestConfig`)
- `kl_threshold`: Minimum violation to flag (default: 0.01)
- `transaction_cost`: Fee per leg (default: 0.015)
- `solver_mode`: "adaptive" | "line_search" | "fixed"
- `max_ticks`: Stop after N ticks (for testing)

---

## 3. Data Module (`src/data/`)

### Purpose
Data access layer for Polymarket data stored in parquet files.

### Files

| File | Purpose |
|------|---------|
| `models.py` | Pydantic models: Market, Trade, BlockTimestamp |
| `loader.py` | DuckDB-based loaders: MarketLoader, TradeLoader, BlockLoader |
| `loader_category.py` | Category-aware loading with DuckDB index |
| `category_index.py` | Query interface for market_categories table |
| `price_series.py` | OHLCV candlestick builder |
| `tick_stream.py` | Tick-level iteration for arbitrage detection |
| `__init__.py` | Public API exports |

### Data Sources
```
{data_dir}/
├── markets/    → Market metadata parquet files
├── trades/     → OrderFilled event parquet files
├── blocks/     → Block timestamp parquet files
```

### Key Classes

**`CrossMarketIterator`**: Synchronized iteration across multiple markets
```python
for snapshot in iterator.iter_snapshots():
    prices = snapshot.get_prices()  # {market_id: price}
```

**`CategoryIndex`**: Query markets by LLM-assigned category
```python
politics_markets = index.query_by_category("politics", "us-election")
```

### Category Taxonomy
| Category | Subcategories |
|----------|---------------|
| politics | us-election, us-congress, international, policy |
| sports | nba, nfl, mlb, nhl, soccer, mma |
| crypto | bitcoin, ethereum, altcoins, defi |
| finance | fed, markets, economics |
| weather | temperature, storms, climate |
| science | space, research, tech, ai |
| entertainment | awards, tv, movies |

---

## 4. LLM Module (`src/llm/`)

### Purpose
LLM-powered market analysis for semantic clustering and constraint extraction.

### Files

| File | Purpose |
|------|---------|
| `schema.py` | MarketCluster, MarketRelationship, RelationshipGraph |
| `client.py` | OpenRouter API client for Kimi 2.5 |
| `categorizer.py` | Rule-based market categorization (17 keyword rules) |
| `clustering.py` | Two-stage LLM clustering pipeline |
| `extractor.py` | Partition constraint extraction |
| `__init__.py` | Public API exports |

### Two-Stage Pipeline

**Stage 1: Semantic Clustering**
- LLM groups markets by shared event, subject, or competing outcomes
- Output: `MarketCluster` objects with theme and market IDs

**Stage 2: Partition Extraction** (3+ markets only)
- LLM identifies exhaustive + mutually exclusive relationships
- Output: `is_partition=True` flag and pairwise constraints

### LLM Configuration
- **Provider**: OpenRouter
- **Model**: `moonshotai/kimi-k2` (Kimi 2.5)
- **Temperature**: 0.2 (low for consistency)
- **JSON Mode**: Enabled for structured output

### Relationship Types
| Type | Constraint | Example |
|------|------------|---------|
| `MUTUALLY_EXCLUSIVE` | P(A) + P(B) ≤ 1 | Trump vs Harris |
| `EXHAUSTIVE` | sum(P) ≥ 1 | At least one wins |
| `IMPLIES` | P(A) ≤ P(B) | State win → national win |
| `PREREQUISITE` | P(B) ≤ P(A) | Nomination → general |

---

## 5. Visualization Module (`src/visualization/`)

### Purpose
Matplotlib-based plotting for arbitrage analysis visualization.

### Files

| File | Purpose |
|------|---------|
| `schema.py` | ArbitrageSignal dataclass |
| `simplex.py` | SimplexProjector for N-dim → 2D projection |
| `signal_plot.py` | Simplex polygon plots |
| `bregman_plot.py` | Bar charts, radar charts, trade tables |
| `__init__.py` | Public API exports |

### Plot Types

1. **Simplex Plot** (`signal_plot.py`)
   - Market prices (red X) vs coherent prices (green dot)
   - Edge vector (blue arrow) showing correction direction
   - Works for 2-4 market clusters

2. **Bregman Dual Panel** (`bregman_plot.py`)
   - Left: Horizontal bar chart of price corrections
   - Right: Polar radar chart comparing prices
   - Trade recommendations table

3. **Batch Grid** (`signal_plot.py`)
   - 3xN grid of mini simplex visualizations
   - For bulk analysis

---

## 6. Arbitrage Module (`src/arbitrage/`)

### Purpose
Converts constraint violations into executable arbitrage trades.

### Files

| File | Purpose |
|------|---------|
| `extractor.py` | ArbitrageExtractor, ArbitrageTrade |
| `__init__.py` | Public API exports |

### Key Insight
> **Profit = Violation Magnitude**, not price convergence

### Trade Extraction Logic

| Constraint | Condition | Strategy | Profit |
|------------|-----------|----------|--------|
| Binary (YES+NO=1) | sum < 1 | Buy both | `1 - sum` |
| Binary (YES+NO=1) | sum > 1 | Sell both | `sum - 1` |
| Implies (A→B) | P(A) > P(B) | Sell A, Buy B | `P(A) - P(B)` |
| Partition | sum < 1 | Buy all | `1 - sum` |
| Partition | sum > 1 | Sell all | `sum - 1` |

### ArbitrageTrade Dataclass
```python
@dataclass
class ArbitrageTrade:
    constraint_type: str
    positions: dict[str, Literal["BUY", "SELL"]]
    violation_amount: float
    locked_profit: float
    market_prices: dict[str, float]
    description: str

    def net_profit(self, fee_per_leg: float = 0.01) -> float:
        return self.locked_profit - (self.num_legs * fee_per_leg)
```

---

## 7. Configuration & Utilities

### `config.py` - Secure Configuration
- `SecretString` class prevents accidental logging of secrets
- Validates `.env` file permissions (must be 600)
- Manages `OPENROUTER_API_KEY`

### `logging_config.py` - Centralized Logging
- Module prefixes: `[FW]`, `[LMO]`, `[SIM]`, `[ARB]`, etc.
- `PrefixFormatter` for consistent output
- `timed()` context manager for performance measurement

---

## Cross-Module Dependencies

```
                    ┌─────────────────┐
                    │   config.py     │
                    │ logging_config  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │  data/  │    │  llm/   │    │optimizer/│
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
             │              ▼              │
             │    ┌─────────────────┐      │
             │    │RelationshipGraph│      │
             │    └────────┬────────┘      │
             │             │               │
             └─────────────┼───────────────┘
                           ▼
                    ┌─────────────┐
                    │  backtest/  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │arbitrage/│ │  report  │ │   viz/   │
        └──────────┘ └──────────┘ └──────────┘
```

---

## Performance Characteristics

### Optimizer Performance
| Mode | Time per Call | Speedup |
|------|---------------|---------|
| Line Search | 342ms | baseline |
| Adaptive | 112ms | 3.1x |

### Backtest Throughput
- With adaptive solver: ~44 ticks/second
- 10,000 ticks in ~227 seconds

### Bottlenecks
1. **Gurobi MILP** in LMO (dominates at small problem sizes)
2. **Data loading** from parquet files
3. **LLM API calls** for clustering (external latency)

---

## Configuration Reference

### OptimizationConfig
```python
OptimizationConfig(
    max_iterations=100,
    tolerance=1e-4,
    step_mode="adaptive",  # or "line_search", "fixed"
    smoothness_alpha=0.1,  # EMA decay for L estimation
    epsilon_init=0.1,      # Barrier contraction
    epsilon_min=0.001,
    epsilon_decay=0.9,
)
```

### BacktestConfig
```python
BacktestConfig(
    market_ids=["253591", "253597"],
    kl_threshold=0.01,
    transaction_cost=0.015,
    solver_mode="adaptive",
    max_ticks=10000,
    progress_interval=1000,
)
```

---

## Key Design Decisions

1. **Focus on 3+ Market Partitions**: True combinatorial arbitrage, not simple 2-market constraints

2. **Two-Stage Detection**: Fast algebraic check → expensive solver fallback

3. **Correct Profit Model**: `profit = |violation_amount|`, not sum of price adjustments

4. **Adaptive Step Sizes**: 3x speedup over line search with comparable accuracy

5. **Schema Separation**: `llm/schema.py` vs `optimizer/schema.py` for different use cases

6. **Lazy Imports**: Root `__init__.py` uses `__getattr__` for startup optimization

---

## Future Considerations

1. **Schema Unification**: Merge `llm.schema.MarketCluster` and `optimizer.schema.MarketCluster`
2. **Solver Alternatives**: Consider OR-Tools or CVXPY instead of Gurobi
3. **Real-time Mode**: Add WebSocket support for live arbitrage detection
4. **Multi-exchange**: Support for multiple prediction market platforms

---

*Generated by Claude Code analysis agents*
