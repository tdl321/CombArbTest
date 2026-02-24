# Architecture Research

## Components

### 1. Data Layer (`data/`)
- **Purpose**: Load and query Polymarket parquet dataset
- **Inputs**: Parquet file paths
- **Outputs**: Market metadata, trade records, price time series
- **Key classes**: `MarketLoader`, `TradeLoader`, `PriceSeriesBuilder`

### 2. Market Clustering (`clustering/`)
- **Purpose**: Group related markets for arbitrage analysis
- **Inputs**: Market metadata (questions, slugs, dates)
- **Outputs**: Market clusters with relationship types
- **Key classes**: `SemanticClusterer`, `RelationshipClassifier`

### 3. LLM Integration (`llm/`)
- **Purpose**: Semantic analysis of market relationships
- **Inputs**: Market pairs, questions, context
- **Outputs**: Relationship classification, confidence scores
- **Key classes**: `LLMClient` (LiteLLM wrapper), `DependencyExtractor`, `SemanticRanker`

### 4. Statistical Analysis (`stats/`)
- **Purpose**: Lead-lag detection, correlation analysis
- **Inputs**: Price time series
- **Outputs**: Granger causality results, candidate pairs
- **Key classes**: `GrangerAnalyzer`, `CorrelationMatrix`

### 5. Optimization Engine (`optimizer/`)
- **Purpose**: Frank-Wolfe solver for arbitrage-free prices
- **Inputs**: Market prices, logical constraints (polytope)
- **Outputs**: Coherent price vector, KL divergence (profit)
- **Key classes**: `FrankWolfeSolver`, `LinearMinimizationOracle`, `MarketPolytope`

### 6. Backtester (`backtest/`)
- **Purpose**: Historical simulation and PnL calculation
- **Inputs**: Price series, optimizer results
- **Outputs**: Trade signals, PnL report
- **Key classes**: `Backtester`, `OpportunityDetector`, `PnLCalculator`

### 7. Reporting (`reports/`)
- **Purpose**: Output results and visualizations
- **Inputs**: Backtest results
- **Outputs**: Summary stats, charts, CSV exports
- **Key classes**: `ReportGenerator`, `OpportunityPlotter`

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  parquet files → MarketLoader → markets DataFrame                │
│  parquet files → TradeLoader → trades DataFrame                  │
│  trades → PriceSeriesBuilder → price time series                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CANDIDATE DISCOVERY                          │
│  markets → SemanticClusterer → candidate clusters                │
│  prices → GrangerAnalyzer → lead-lag pairs                       │
│  clusters + pairs → merged candidates                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM FILTERING                               │
│  candidates → DependencyExtractor → relationship graph           │
│  relationships → SemanticRanker → ranked opportunities           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION                                 │
│  relationship graph → MarketPolytope → constraint set            │
│  market prices + constraints → FrankWolfeSolver → coherent μ*    │
│  μ* vs market prices → KL divergence → profit estimate           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKTESTING                                  │
│  historical prices → Backtester → walk-forward simulation        │
│  opportunities → PnLCalculator → theoretical returns             │
│  results → ReportGenerator → output files                        │
└─────────────────────────────────────────────────────────────────┘
```

## Build Order

1. **Data Layer** — Foundation, everything depends on this
2. **Optimization Engine** — Core algorithm, can test with synthetic data
3. **Statistical Analysis** — Granger causality for candidate generation
4. **Market Clustering** — Semantic grouping (can start simple)
5. **LLM Integration** — Filtering layer, depends on clustering
6. **Backtester** — Ties everything together
7. **Reporting** — Final output layer

## Integration Points

| From | To | Interface |
|------|-----|-----------|
| Data → Stats | `PriceSeries` DataFrame | time-indexed prices |
| Data → Clustering | `Market` list | market metadata |
| Clustering → LLM | `MarketPair` | candidate pairs |
| LLM → Optimizer | `ConstraintGraph` | logical relationships |
| Optimizer → Backtest | `ArbitrageOpportunity` | profit estimates |

## Directory Structure

```
combarbbot/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── price_series.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   └── semantic.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── extractor.py
│   ├── stats/
│   │   ├── __init__.py
│   │   └── granger.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── frank_wolfe.py
│   │   └── polytope.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── engine.py
│   └── reports/
│       ├── __init__.py
│       └── generator.py
├── tests/
├── notebooks/
└── pyproject.toml
```
