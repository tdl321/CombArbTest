# Combarbbot Roadmap

**Last Updated:** 2026-02-27

---

## Completed Phases

### v1.0 — Core Implementation (2026-02-24)
- [x] DATA-01 through DATA-05: Data layer with DuckDB loaders
- [x] LLM-01 through LLM-04: LLM analysis with Kimi 2.5 via OpenRouter
- [x] OPT-01 through OPT-05: Frank-Wolfe optimizer with HiGHS MILP LMO
- [x] BT-01 through BT-04: Walk-forward backtester
- [x] LOG-01 through LOG-04: Structured logging
- [x] VIZ-01 through VIZ-04: Simplex and Bregman visualizations

### v1.1 — Refinements (2026-02-25)
- [x] Marginal Polytope LMO v2 (HiGHS MILP, binary integrality)
- [x] ArbitrageExtractor (correct profit = violation magnitude)
- [x] Two-stage LLM clustering (partition detection)
- [x] Tick-level data iteration (CrossMarketIterator)
- [x] Backward compatibility wrappers

### v1.2 — Market Categorization (2026-02-25/26)
- [x] Rule-based market categorizer (src/llm/categorizer.py)
- [x] DuckDB category index (408,800 markets)
- [x] FastAPI market categories API (systemd service)
- [x] Tournament backtest system
- [x] LLM response caching (DuckDB-backed, TTL)

### v1.2.1 — Enhancement Pass (2026-02-27)
- [x] **CRITICAL**: Fixed solver -> trade extraction root cause
- [x] KL divergence / gradient alignment
- [x] Missing constraint type handlers (5 types)
- [x] Default centroid fix (1/N for multi-outcome)
- [x] Schema unification (optimizer inherits from LLM)
- [x] Path centralization (env var overrides)
- [x] sys.path hack removal
- [x] Test suite repair (130 tests passing)

---

## Future Work

### v1.3 — Production Readiness
- [ ] Remove duplicate class definitions from runner scripts
- [ ] Remove unused dependencies (litellm, instructor)
- [ ] Add Protocol types for dependency inversion
- [ ] Fix Pydantic V2 deprecation warnings
- [ ] Add input validation at boundaries (price range, market ID consistency)
- [ ] Use `instructor` library for structured LLM output validation
- [ ] Add LLM cost/token tracking

### v1.4 — Real-Time Trading Foundation
- [ ] WebSocket price feed (CRITICAL gap per STRATEGY_GAPS.md)
- [ ] Order book depth modeling
- [ ] Position sizing with Kelly criterion
- [ ] Gas-aware cost model
- [ ] Execution layer

### v2.0 — Rust Solver Migration (see RUST_SOLVER_MIGRATION.md)
- [ ] Scaffold Rust crate with PyO3 + maturin
- [ ] Port LMO to Rust (10-40x speedup expected)
- [ ] Port KL divergence + gradient (20-100x speedup)
- [ ] Port full Frank-Wolfe loop with rayon parallelism
- [ ] Integration tests (Rust vs corrected Python results)
- [ ] Python fallback toggle

### v3.0 — Advanced Features
- [ ] Multi-exchange arbitrage (cross-platform)
- [ ] Web dashboard for signal visualization
- [ ] Automated trade execution
- [ ] Multi-leg arbitrage path discovery (3+ markets)
- [ ] Confidence scoring for LLM-generated constraints
- [ ] Liquidity-adjusted profit estimates
