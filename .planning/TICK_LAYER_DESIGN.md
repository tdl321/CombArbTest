# Tick-Level Data Layer Design

## Problem Statement

OHLCV price series aggregates trades over time buckets (minutes/hours), losing:
1. **Exact timing** - When exactly did cross-market mispricing occur?
2. **Duration** - How long did the arbitrage window stay open?
3. **Ordering** - Which market moved first?

For arbitrage detection, we need trade-by-trade precision with deterministic ordering.

## Raw Data Schema

### Trades
| Field | Type | Purpose |
|-------|------|---------|
| block_number | BIGINT | Primary ordering (~2s on Polygon) |
| log_index | BIGINT | Ordering within block |
| maker_asset_id | VARCHAR | Token ID (0 = USDC) |
| taker_asset_id | VARCHAR | Token ID (0 = USDC) |
| maker_amount | BIGINT | 6 decimals |
| taker_amount | BIGINT | 6 decimals |

### Key Insight
- timestamp field on trades is NULL - must join with blocks table
- Within a block, log_index provides deterministic trade ordering
- Block timestamps are ~2 seconds apart on Polygon

## Implementation

### New File: src/data/tick_stream.py

**TickPosition** (frozen dataclass)
- (block_number, log_index) tuple
- Implements comparison for ordering
- All trades at same position are atomic

**Tick** (frozen dataclass)
- Contains computed price, side, size, usdc_volume
- Links to TickPosition for ordering
- Lazy timestamp from blocks join

**MarketStateSnapshot**
- Point-in-time state of a single market
- Tracks last_price, prev_price for momentum
- Flags staleness when no recent trades

**CrossMarketSnapshot**
- Synchronized state of multiple markets at one position
- has_all_prices() - check readiness for arbitrage check
- get_prices() - extract price dict

**TickStream**
- Iterator over trades for a single market
- Maintains block/log ordering
- Batched queries to manage memory

**CrossMarketIterator**
- Synchronized iteration across N markets
- Yields snapshot at every tick where ANY market trades
- Snapshot contains current state of ALL markets

## Integration with Arbitrage Detection

The tick layer enables:

1. **Exact moment detection** - Know precisely when prices diverge
2. **Window measurement** - Count ticks/time until prices converge
3. **First-mover detection** - See which market moved first
4. **Staleness handling** - Skip when markets have no recent activity

## Performance Considerations

- Batched queries (5000-10000 trades per batch)
- Lazy timestamp enrichment
- Memory-efficient iteration (yields, not lists)
- DuckDB 3GB memory limit for 8GB VPS

## Files Changed

- src/data/tick_stream.py - NEW: tick-level classes
- src/data/__init__.py - UPDATED: export tick classes
- test_tick_stream.py - NEW: tests

## Next Steps

1. Integrate with Frank-Wolfe solver for arbitrage-free price calculation
2. Add order book state tracking (from CLOB if available)
3. Implement arbitrage opportunity logging/persistence
4. Backtest on historical tick data

---
*Created: 2025-02-25*
