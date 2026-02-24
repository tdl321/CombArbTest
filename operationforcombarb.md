# Combinatorial Arbitrage Engine: Technical Architecture

## Semantic Mapping and LLM Orchestration

### LLMs as Semantic Risk Managers

The architecture integrates an LLM-based semantic stage to re-rank statistical candidates (Granger causality pairs), prioritizing relationships with plausible real-world mechanisms. LLM filtering reduces downside risk by ~50% compared to purely statistical discovery.

The LLM assesses whether a proposed leader-follower pair (e.g., "Inflation Report" leads "S&P 500 Prediction") is coherent, filtering out "fragile" links likely to fail under changing market conditions.

### Guardrails and Dependency Extraction

Multi-layered guardrail system using RCFCE framework:

| Component | Implementation |
|-----------|----------------|
| **Role** | "You are an expert financial engineer and logical auditor" |
| **Context** | Latest market descriptions + resolved events from Gamma API |
| **Format** | JSON schema mapping markets to logical operators (AND, OR, NOT, XOR) + conditional probabilities |
| **Constraints** | "Do not hypothesize beyond provided text. If ambiguous, return 'unlinked'" |
| **Examples** | Few-shot examples for nested dependencies (championship vs. bracket wins) |

### Prompting Techniques

| Technique | Application | Benefit |
|-----------|-------------|---------|
| Zero-Shot CoT | Logical reasoning | Step-by-step dependency deduction |
| Few-Shot | Format control | JSON schema compliance |
| Chain-of-Verification | Fact-checking | Reduced hallucinations in dependency trees |
| Self-Consistency | Stability | Majority voting over multiple outputs |

---

## Code Architecture: Python Prototyping Layer

### Strategy Discovery Pipeline

1. **Lead-Lag Discovery**: Granger causality on historical market-implied probability time series
2. **Semantic Re-ranking**: Candidate pairs → LLM (Llama 3/GPT-4o) with strict guardrails
3. **Backtesting**: Frank-Wolfe solver validates theoretical profit against historical data

### LLM Interface

- **Structural Enforcement**: Guardrails AI or Pydantic for output integrity
- **Storage**: SQLite for "Canonical Cycle Paths" (deduplicated profitable arbitrage paths)
- **Data Sources**: Polymarket CLOB API + Gamma API

---

## Code Architecture: Rust Execution Engine

### System Architecture

Hexagonal architecture (Ports and Adapters) - core math logic decoupled from external API clients.

### Concurrency Model

| Layer | Implementation |
|-------|----------------|
| **Real-Time Data** | Tokio async runtime for concurrent WebSocket connections (RTDS + CLOB streams) |
| **Order Book Management** | DashMap (concurrent hash map) for bid-ask spreads and liquidity depths |
| **Parallel Search** | Rayon work-stealing parallelism for evaluating thousands of arbitrage cycles |

### Numerical Optimization Stack

| Crate | Domain | Utility |
|-------|--------|---------|
| `good_lp` | Optimization Modeling | DSL for marginal polytope constraints |
| `highs` | Linear/MIP Solver | LMO sub-problems at HFT speeds |
| `ndarray` | Linear Algebra | Matrix operations for gradients |
| `numrs` | Numerical Analysis | SIMD-optimized distance metrics |
| `polymarket-hft` | Execution | Low-latency CLOB interaction |

### Frank-Wolfe Solver Implementation

**Linear Minimization Oracle (LMO)**: `good_lp` + HiGHS backend

**Barrier Frank-Wolfe Variant** for numerical stability near polytope boundaries:
- Contracts marginal polytope M toward interior point μ₀
- Contracted polytope: M_ε = (1-ε)M + ε·μ₀

**Adaptive Contraction**:

| Phase | ε Value | Purpose |
|-------|---------|---------|
| Initialization | ~0.1 | High stability, rapid rough convergence |
| Mid-Optimization | ~0.01 | Refining coherent price vector |
| Convergence | ~0.001 | High-precision arbitrage detection |

**InitFW Process**:
1. Logical Settlement: IP solver checks if securities fixed to 0/1
2. Vertex Collection: IP solver finds extreme points
3. Centroid Calculation: μ₀ = average of vertices (guarantees relative interior)

---

## Execution Layer

### Order Sequencing

Multi-leg arbitrage execution prioritizes:
1. Lowest liquidity markets first (highest price movement risk)
2. "Leader" markets before "follower" legs

### Transaction Handling

- EIP-712 signing for Polygon settlement
- Pre-calculated/parallel order signing to reduce latency

### Slippage Calculation

1. Fetch aggregated bids/asks for target `token_id`
2. Walk book: iteratively subtract size at each price level
3. Calculate weighted average fill price
4. Compare against arbitrage-free price μ* from solver

### Tick Size Compliance

| Order Type | Precision |
|------------|-----------|
| Maker amounts | 2 decimal places |
| Taker amounts | 4 decimal places |
| Limit orders | Quantity = base conditional tokens |
| Market BUY | Quantity = quote notional (USDC) |

---

## Causal Discovery Pipeline

### Two-Stage Screener

| Stage | Method | Tooling | Objective |
|-------|--------|---------|-----------|
| Statistical | Granger Causality | `statsmodels` (Python) | Identify candidate leader-follower pairs |
| Semantic | LLM Plausibility | GPT-4o + Guardrails | Filter spurious correlations |
| Verification | Convex Opt (FW) | `good_lp` (Rust) | Quantify risk-free profit margin |
| Execution | HFT CLOB API | `polymarket-hft` (Rust) | Low-latency order placement |

### Granger Causality Model

Bivariate VAR for markets X, Y:
```
Y_t = α + Σβ_i·Y_{t-i} + Σγ_j·X_{t-j} + ε_t
```
If coefficients γ_j jointly significant → candidate for LLM re-ranking.

---

## Operational Security

### API Management

- Credentials: `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_PASSPHRASE`
- Rust `secrecy` crate prevents accidental logging
- Exponential backoff retry logic for rate-limiting

### LLM Security

| Layer | Protection |
|-------|------------|
| Input Guardrails | Scan for forbidden terms/jailbreak patterns |
| DLP | Prevent private keys/sensitive data in prompts |
| Output Validation | JSON schema enforcement |
