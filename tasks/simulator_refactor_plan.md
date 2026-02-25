# Simulator & Backtest Refactoring Plan

## Objective

Refactor the simulator and backtest system to:
1. Use correct arbitrage logic (ArbitrageTrade) instead of price-convergence model
2. Produce standardized reports with simplex visualizations
3. Separate signal detection from profit calculation cleanly

---

## Current State Analysis

### What Exists
```
src/
├── optimizer/
│   ├── frank_wolfe.py      ✓ Solver works correctly
│   ├── lmo.py              ✓ LMO works correctly
│   ├── divergence.py       ✓ KL divergence correct
│   └── schema.py           ✓ ArbitrageResult defined
├── arbitrage/
│   ├── __init__.py         ✓ NEW - created
│   └── extractor.py        ✓ NEW - ArbitrageTrade extraction
├── backtest/
│   ├── simulator.py        ✓ REFACTORED - uses ArbitrageExtractor
│   ├── pnl.py              ✓ DEPRECATED old + new functions active
│   ├── schema.py           ✓ REFACTORED - embeds ArbitrageTrade
│   ├── report.py           ⚠️ Old report format
│   └── report_generator.py ✓ NEW - correct report generation
└── visualization/
    ├── simplex.py          ✓ SimplexProjector works
    ├── bregman_plot.py     ✓ Dual-panel plots work
    └── signal_plot.py      ✓ Signal plotting works
```

### Current Flow (Wrong)
```
Simulator → ArbitrageResult → calculate_opportunity_pnl() → ArbitrageOpportunity
                                        │
                                        └── profit = Σ|coherent - market|  ❌
```

### Target Flow (Correct)
```
Simulator → ArbitrageResult → ArbitrageExtractor → ArbitrageTrade
                                                          │
                                                          └── profit = violation_magnitude  ✓
                                                          │
                                                          ▼
                                              ReportGenerator → Report + Visualizations
```

---

## Refactoring Tasks

### Phase 1: Schema Updates

#### Task 1.1: Update ArbitrageOpportunity schema
**File:** `src/backtest/schema.py`

Replace the old ArbitrageOpportunity with one based on ArbitrageTrade:

```python
# OLD
class ArbitrageOpportunity(BaseModel):
    theoretical_profit: float  # Σ|coherent - market| ❌
    trade_direction: dict[str, str]  # per-market toward coherent ❌

# NEW
class ArbitrageOpportunity(BaseModel):
    # Detection metadata
    timestamp: datetime
    block_number: int
    cluster_id: str
    
    # The actual arbitrage trade
    trade: ArbitrageTrade  # Contains positions, locked_profit, etc.
    
    # Solver output (for diagnostics)
    solver_result: Optional[ArbitrageResult] = None
    
    # Computed from trade
    @property
    def locked_profit(self) -> float:
        return self.trade.locked_profit
    
    @property
    def net_profit(self) -> float:
        return self.trade.net_profit(fee_per_leg=0.01)
    
    @property
    def positions(self) -> dict[str, str]:
        return self.trade.positions
```

**Checklist:**
- [x] Backup existing schema
- [x] Update ArbitrageOpportunity to embed ArbitrageTrade
- [x] Remove old fields (theoretical_profit, coherent_prices diff logic)
- [x] Add backward-compatible properties if needed
- [x] Update BacktestReport to match new opportunity structure

---

#### Task 1.2: Create unified BacktestOutput schema
**File:** `src/backtest/schema.py`

Add a new schema for complete backtest output:

```python
class BacktestOutput(BaseModel):
    """Complete output from a backtest run."""
    # Metadata
    run_id: str
    run_timestamp: datetime
    config: BacktestConfig
    
    # Results
    report: ArbitrageBacktestReport
    opportunities: list[ArbitrageOpportunity]
    
    # Output paths
    report_text_path: Optional[str] = None
    visualization_paths: list[str] = []
```

**Checklist:**
- [x] Define BacktestOutput schema
- [x] Include all necessary metadata for reproducibility
- [x] Add serialization methods (to_json, to_dict)

---

### Phase 2: Simulator Refactoring

#### Task 2.1: Create new _extract_arbitrage method [COMPLETE]
**File:** `src/backtest/simulator.py`

Add method to convert solver output to ArbitrageTrade:

```python
def _extract_arbitrage(
    self,
    result: ArbitrageResult,
    snapshot: CrossMarketSnapshot,
    cluster: MarketCluster,
    config: BacktestConfig,
) -> Optional[ArbitrageOpportunity]:
    """Extract arbitrage trade from solver result."""
    from src.arbitrage.extractor import ArbitrageExtractor
    
    extractor = ArbitrageExtractor(
        min_profit_threshold=config.min_profit,
        fee_per_leg=config.transaction_cost,
    )
    
    trades = extractor.extract_trades(result)
    
    if not trades:
        return None
    
    # Take best trade
    best_trade = max(trades, key=lambda t: t.locked_profit)
    
    # Check profitability threshold
    if best_trade.net_profit(config.transaction_cost) < config.min_profit:
        return None
    
    return ArbitrageOpportunity(
        timestamp=snapshot.timestamp or datetime.now(),
        block_number=snapshot.position.block_number,
        cluster_id=cluster.cluster_id,
        trade=best_trade,
        solver_result=result,
    )
```

**Checklist:**
- [x] Add _extract_arbitrage method
- [x] Integrate ArbitrageExtractor
- [x] Handle edge cases (no trades, unprofitable)

---

#### Task 2.2: Refactor _check_arbitrage to use new extraction [PARTIAL - backward compat maintained]
**File:** `src/backtest/simulator.py`

Update the main detection method:

```python
def _check_arbitrage(
    self,
    snapshot: CrossMarketSnapshot,
    cluster: MarketCluster,
    prices: dict[str, float],
    graph: RelationshipGraph,
    config: BacktestConfig,
) -> Optional[ArbitrageOpportunity]:
    """Check for arbitrage opportunity in a cluster."""
    
    # Stage 1: Fast partition check
    if is_partition_constraint(cluster):
        partition_ids = get_partition_market_ids(cluster)
        violation = check_partition(partition_ids, prices)
        
        if violation:
            return self._create_partition_opportunity_v2(
                snapshot, cluster, violation, config
            )
    
    # Stage 2: Solver for complex constraints
    result = find_arbitrage(market_prices=prices, relationships=graph)
    
    if result.kl_divergence < config.kl_threshold:
        return None
    
    # NEW: Use extractor instead of old PnL logic
    return self._extract_arbitrage(result, snapshot, cluster, config)
```

**Checklist:**
- [x] Replace calculate_opportunity_pnl calls with _extract_arbitrage
- [x] Update _create_partition_opportunity to use ArbitrageTrade
- [x] Update _check_arbitrage_with_solver similarly
- [x] Remove imports of old PnL functions

---

#### Task 2.3: Refactor _create_partition_opportunity [COMPLETE]
**File:** `src/backtest/simulator.py`

Update to create ArbitrageTrade directly:

```python
def _create_partition_opportunity_v2(
    self,
    snapshot: CrossMarketSnapshot,
    cluster: MarketCluster,
    violation: PartitionViolation,
    config: BacktestConfig,
) -> Optional[ArbitrageOpportunity]:
    """Create opportunity from partition violation using ArbitrageTrade."""
    from src.arbitrage.extractor import ArbitrageTrade
    
    # Determine direction
    if violation.direction == "underpriced":
        positions = {m: "BUY" for m in violation.prices}
        direction_word = "BUY ALL"
    else:
        positions = {m: "SELL" for m in violation.prices}
        direction_word = "SELL ALL"
    
    locked_profit = abs(violation.violation_amount)
    
    trade = ArbitrageTrade(
        constraint_type="partition",
        positions=positions,
        violation_amount=locked_profit,
        locked_profit=locked_profit,
        market_prices=violation.prices,
        description=f"Partition arb: {direction_word} at {violation.total:.4f}",
    )
    
    # Check profitability
    if trade.net_profit(config.transaction_cost) < config.min_profit:
        return None
    
    return ArbitrageOpportunity(
        timestamp=snapshot.timestamp or datetime.now(),
        block_number=snapshot.position.block_number,
        cluster_id=cluster.cluster_id,
        trade=trade,
        solver_result=None,  # No solver used for partition
    )
```

**Checklist:**
- [x] Rewrite _create_partition_opportunity
- [x] Create ArbitrageTrade directly from violation
- [x] Remove coherent_prices computation (not needed)

---

### Phase 3: Report Integration

#### Task 3.1: Create run_backtest_with_report function [COMPLETE]
**File:** `src/backtest/simulator.py` (or new file `src/backtest/runner.py`)

Top-level function that runs simulation and generates report:

```python
def run_backtest_with_report(
    market_ids: list[str],
    relationship_graph: RelationshipGraph,
    market_loader: MarketLoader,
    trade_loader: TradeLoader,
    block_loader: BlockLoader,
    config: BacktestConfig,
    output_dir: str,
) -> BacktestOutput:
    """Run complete backtest with report generation."""
    
    # Run simulation
    simulator = WalkForwardSimulator(market_loader, trade_loader, block_loader)
    opportunities = list(simulator.run(market_ids, relationship_graph, config))
    
    # Extract trades for report
    trades = [opp.trade for opp in opportunities]
    
    # Determine period
    if opportunities:
        period_start = min(o.timestamp for o in opportunities)
        period_end = max(o.timestamp for o in opportunities)
    else:
        period_start = period_end = None
    
    # Generate report
    from .report_generator import generate_full_report
    
    report, report_text, viz_files = generate_full_report(
        trades=trades,
        output_dir=output_dir,
        period_start=period_start,
        period_end=period_end,
        markets_analyzed=len(market_ids),
        clusters_monitored=len(relationship_graph.clusters),
        fee_per_leg=config.transaction_cost,
    )
    
    # Save report text path
    report_text_path = f"{output_dir}/backtest_report.txt"
    
    return BacktestOutput(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        run_timestamp=datetime.now(),
        config=config,
        report=report,
        opportunities=opportunities,
        report_text_path=report_text_path,
        visualization_paths=viz_files,
    )
```

**Checklist:**
- [x] Create run_backtest_with_report function
- [x] Wire up report_generator
- [x] Handle empty results case
- [x] Return complete BacktestOutput

---

#### Task 3.2: Update report_generator for richer visualizations [COMPLETE]
**File:** `src/backtest/report_generator.py`

Enhance visualization generation:

```python
def generate_simplex_visualizations(
    report: ArbitrageBacktestReport,
    output_dir: str,
    max_plots: int = 10,
) -> list[str]:
    """Generate both Bregman dual-panel AND simplex geometry plots."""
    
    saved_files = []
    
    for i, trade in enumerate(report.top_opportunities[:max_plots], 1):
        # 1. Bregman dual-panel (existing)
        bregman_path = _generate_bregman_plot(trade, i, output_dir)
        saved_files.append(bregman_path)
        
        # 2. Simplex geometry plot (new)
        simplex_path = _generate_simplex_plot(trade, i, output_dir)
        saved_files.append(simplex_path)
    
    # 3. Summary grid of all top opportunities
    grid_path = _generate_summary_grid(report.top_opportunities, output_dir)
    saved_files.append(grid_path)
    
    return saved_files


def _generate_simplex_plot(trade: ArbitrageTrade, index: int, output_dir: str) -> str:
    """Generate simplex geometry visualization showing market vs coherent."""
    from ..visualization.signal_plot import plot_arbitrage_signal
    from ..visualization.schema import ArbitrageSignal
    
    # Convert trade to signal format
    markets = list(trade.market_prices.keys())
    total = sum(trade.market_prices.values())
    coherent = {m: p / total for m, p in trade.market_prices.items()}
    
    signal = ArbitrageSignal(
        timestamp=datetime.now(),
        cluster_id=f"Opportunity #{index}",
        markets=markets,
        constraint_type=trade.constraint_type,
        market_prices=trade.market_prices,
        coherent_prices=coherent,
        edge_magnitude=trade.violation_amount,
        kl_divergence=trade.violation_amount,
        direction={m: coherent[m] - trade.market_prices[m] for m in markets},
        constraint_violation=trade.description,
        block_number=0,
    )
    
    save_path = f"{output_dir}/simplex_{index:02d}_{trade.constraint_type}.png"
    plot_arbitrage_signal(signal, save_path=save_path)
    
    return save_path
```

**Checklist:**
- [x] Add _generate_simplex_plot helper
- [x] Add _generate_summary_grid for overview
- [x] Update generate_simplex_visualizations to produce both plot types
- [x] Ensure consistent styling across plots

---

### Phase 4: Cleanup & Deprecation (COMPLETED 2026-02-25)

#### Task 4.1: Mark old PnL functions as deprecated [DONE]
**File:** `src/backtest/pnl.py`

```python
import warnings

def calculate_theoretical_profit(...):
    """DEPRECATED: Use ArbitrageExtractor instead."""
    warnings.warn(
        "calculate_theoretical_profit is deprecated. "
        "Use ArbitrageExtractor.extract_trades() for correct arbitrage profit.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing code for backward compatibility
```

**Checklist:**
- [x] Add deprecation warnings to old functions
- [x] Document migration path in docstrings
- [x] Keep functions for backward compatibility (1 version)

---

#### Task 4.2: Update __init__.py exports [DONE]
**File:** `src/backtest/__init__.py`

```python
from .simulator import (
    WalkForwardSimulator,
    run_simulation,
    run_backtest_with_report,  # NEW
)
from .schema import (
    ArbitrageOpportunity,
    BacktestConfig,
    BacktestOutput,  # NEW
)
from .report_generator import (
    ArbitrageBacktestReport,
    generate_full_report,
    generate_report_text,
)
# Deprecated but kept for compatibility
from .pnl import calculate_opportunity_pnl
```

**Checklist:**
- [x] Update exports in __init__.py
- [x] Add new classes/functions
- [x] Note deprecated exports

---

### Phase 5: Testing & Validation

#### Task 5.1: Create test for new flow
**File:** `tests/test_backtest_refactor.py`

```python
def test_arbitrage_extraction():
    """Test that ArbitrageExtractor produces correct trades."""
    result = ArbitrageResult(
        market_prices={"A": 0.60, "B": 0.50},
        coherent_prices={"A": 0.55, "B": 0.55},
        kl_divergence=0.01,
        constraints_violated=[
            ConstraintViolation(
                constraint_type="implies",
                from_market="A",
                to_market="B",
                violation_amount=0.10,
                description="implies(A->B)"
            )
        ],
        converged=True,
        iterations=10,
    )
    
    trades = extract_arbitrage_from_result(result)
    
    assert len(trades) >= 1
    trade = trades[0]
    assert trade.constraint_type == "implies"
    assert trade.positions == {"A": "SELL", "B": "BUY"}
    assert trade.locked_profit == 0.10


def test_report_generation():
    """Test that report generates correctly."""
    trades = [
        ArbitrageTrade(
            constraint_type="partition",
            positions={"X": "BUY", "Y": "BUY"},
            violation_amount=0.05,
            locked_profit=0.05,
            market_prices={"X": 0.45, "Y": 0.50},
            description="test",
        )
    ]
    
    report, text, viz = generate_full_report(
        trades=trades,
        output_dir="/tmp/test_report",
    )
    
    assert report.total_opportunities == 1
    assert "ARBITRAGE BACKTEST REPORT" in text
    assert len(viz) > 0
```

**Checklist:**
- [x] Test ArbitrageExtractor with each constraint type
- [x] Test report generation
- [x] Test visualization generation
- [x] Test end-to-end backtest run

---

#### Task 5.2: Run comparison backtest
**Script:** `scripts/compare_backtest.py`

Run old vs new on same data to validate:

```python
# Run with old logic
old_results = run_simulation_old(...)

# Run with new logic  
new_results = run_backtest_with_report(...)

# Compare
print("Old total profit:", sum(o.theoretical_profit for o in old_results))
print("New total profit:", sum(o.trade.locked_profit for o in new_results))

# The numbers should be similar for partition constraints
# but may differ for implies/mutex (old was wrong)
```

**Checklist:**
- [x] Create comparison script
- [x] Run on historical data (synthetic scenarios)
- [x] Document differences
- [x] Validate new results make sense

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `src/backtest/schema.py` | MODIFY | Update ArbitrageOpportunity, add BacktestOutput |
| `src/backtest/simulator.py` | MODIFY | Add _extract_arbitrage, refactor _check_arbitrage |
| `src/backtest/pnl.py` | MODIFY | Deprecate old functions |
| `src/backtest/report_generator.py` | MODIFY | Enhance visualizations |
| `src/backtest/__init__.py` | MODIFY | Update exports |
| `src/arbitrage/extractor.py` | EXISTS | Already created |
| `tests/test_backtest_refactor.py` | CREATE | New tests |

---

## Execution Order

```
Phase 1: Schema Updates
  └─ 1.1 Update ArbitrageOpportunity ─────┐
  └─ 1.2 Create BacktestOutput ───────────┤
                                          │
Phase 2: Simulator Refactoring            │
  └─ 2.1 Add _extract_arbitrage ──────────┤ (depends on 1.1)
  └─ 2.2 Refactor _check_arbitrage ───────┤ (depends on 2.1)
  └─ 2.3 Refactor _create_partition ──────┤ (depends on 2.1)
                                          │
Phase 3: Report Integration               │
  └─ 3.1 Create run_backtest_with_report ─┤ (depends on 2.*)
  └─ 3.2 Enhance visualizations ──────────┤
                                          │
Phase 4: Cleanup                          │
  └─ 4.1 Deprecate old PnL ───────────────┤
  └─ 4.2 Update exports ──────────────────┤
                                          │
Phase 5: Testing                          │
  └─ 5.1 Unit tests ──────────────────────┤ (depends on all above)
  └─ 5.2 Comparison backtest ─────────────┘
```

---

## Success Criteria

1. **Correct Profit Calculation**
   - Profit = violation_magnitude, not Σ|coherent - market|
   - Partition: profit = |1 - sum(prices)|
   - Implies: profit = P(from) - P(to) when violated

2. **Proper Trade Structure**
   - Positions are hedged pairs, not per-market directions
   - Each trade is self-contained and executable

3. **Standardized Report Output**
   - Text report with consistent format
   - Breakdown by constraint type
   - Edge distribution
   - Top opportunities

4. **Simplex Visualizations**
   - Bregman dual-panel for each top opportunity
   - Simplex geometry plot showing market vs coherent
   - Summary grid overview

5. **Backward Compatibility**
   - Old code can still run (with deprecation warnings)
   - Migration path documented
