# Combarbbot Project State

**Last Updated:** 2026-02-27
**Status:** v1.2 Enhancement Pass Complete
**Branch:** main (10 commits)

---

## Current State

### Test Suite
- **130 unit tests pass** (0 failures)
- 12 integration tests (require LLM API / data files) deselected by default
- Test files: 8 test modules covering backtest, constraint checker, data, LLM, optimizer, partition, visualization

### Architecture
- **Schema unified**: `src/llm/schema.py` is the canonical source for shared types
  - `RelationshipType`, `MarketRelationship`, `MarketCluster`, `RelationshipGraph`
  - `src/optimizer/schema.py` inherits from LLM schema (no more duplication)
- **Paths centralized**: `src/config.py` has `DEFAULT_DATA_DIR`, `DEFAULT_DB_PATH`, `PROJECT_ROOT`
  - Override via env vars: `COMBARBBOT_DATA_DIR`, `COMBARBBOT_DB_PATH`, `COMBARBBOT_ROOT`
- **sys.path hacks removed**: `src/llm/client.py` uses proper relative imports

### Critical Bugs Fixed
1. **Solver -> Trade extraction** (ROOT CAUSE of zero-profit backtests):
   - `_marginal_to_legacy_result` now generates properly typed `ConstraintViolation` objects
   - Uses `RelationshipGraph` to map violations to: implies, mutex, binary, equivalent
   - `ArbitrageExtractor` can now process all solver-detected arbitrage
2. **KL divergence / gradient mismatch**: 
   - `categorical_kl` no longer normalizes internally (aligned with gradient)
3. **Missing constraint handlers**: 
   - prerequisite, exhaustive, incompatible, and, or now handled in LMO
4. **Default centroid**: 
   - Uses 1/N per market instead of 0.5 (correct for multi-outcome markets)

### Codebase Metrics
| Metric | Value |
|--------|-------|
| Source files | 35+ Python files in src/ |
| Total source lines | ~9,400 |
| Test lines | ~3,000 |
| Test files | 8 |
| Root runner scripts | 13 |
| Dependencies | 12 main + 2 dev |

### Services Running
- `market-categories-api.service` — FastAPI on localhost:8420
- 408,800 markets categorized in DuckDB

### Data
- 51GB Polymarket parquet dataset at `/root/prediction-market-analysis/data/polymarket`
- DuckDB database: `polymarket.db` (43MB) with `market_categories` and `relationship_cache`

---

## Recent Changes (2026-02-27 Enhancement Pass)

| Commit | Change |
|--------|--------|
| cleanup pass | Committed 21 uncommitted files, removed dead code |
| test repair | Fixed test_llm.py imports, rewrote test_backtest.py |
| critical fix | Solver violations mapped to correct constraint types |
| gradient fix | KL divergence objective aligned with gradient |
| constraint handlers | Added prerequisite, exhaustive, incompatible, and, or |
| schema unification | Optimizer types inherit from LLM schema |
| path centralization | Hardcoded paths moved to config.py |
| runner cleanup | Added canonical imports to duplicate runner scripts |

---

## Known Remaining Issues

1. **Pydantic V2 deprecation warning**: `MarketInfo` uses class-based Config (migrate to ConfigDict)
2. **Runner scripts still have local class definitions**: Imports added but duplicates not yet removed
3. **No real-time data feed**: Backtest-only (see STRATEGY_GAPS.md)
4. **Unused dependencies**: `litellm` and `instructor` in pyproject.toml
5. **API server**: No auth, no rate limiting (localhost-only, low risk)
