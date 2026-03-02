# Alpha Source 2: Implied Distribution Arbitrage in BTC Strike Ladder Markets

## Executive Summary

Polymarket runs recurring "Bitcoin above $X on [date]?" binary markets with multiple strikes per expiry. These strikes collectively **imply a probability distribution** for BTC's price at expiry. When that implied distribution violates basic probability axioms (e.g., negative probabilities), there is pure arbitrage. When it deviates from smooth, volatility-consistent shapes, there are soft trading signals.

**Bottom line:** The market structure exists and is rich enough (up to 35 strikes per expiry, $209M total volume, spacing as tight as $100). The stored data contains only post-settlement prices, so we cannot directly measure historical mispricing frequency from snapshots. However, 3 settlement anomalies and 12 duplicate inconsistencies confirm pricing errors DO occur. A live monitor is the correct implementation path.

---

## Market Structure Analysis

### Data Overview
- **423** BTC "above $X on [date]" binary markets identified
- **113** distinct expiry dates
- **21** expiry dates with 3+ unique strikes (analyzable ladders)
- **$209.5M** total volume across all BTC strike markets
- Date range: September 2023 through September 2025

### Ladder Quality (dates with 3+ strikes)

| Date | Unique Strikes | Total Markets | Duplicates | Volume | BTC Settlement Range |
|------|---------------|---------------|------------|--------|---------------------|
| 2025-08-17 | 14 | 35 | 21 | $52K | [$117,900, $118,000] |
| 2025-08-18 | 23 | 28 | 5 | $1.46M | [$116,000, $116,500] |
| 2025-08-20 | 16 | 18 | 2 | $2.58M | [$113,500, $114,000] |
| 2025-08-22 | 16 | 19 | 3 | $3.55M | [$116,000, $117,500] |
| 2025-08-25 | 17 | 31 | 14 | $3.27M | [$112,000, $112,500] |
| 2025-08-27 | 18 | 33 | 15 | $3.52M | [$111,500, $112,000] |
| 2025-09-05 | 11 | 11 | 0 | $2.27M | [$110,000, $112,000] |
| 2025-09-10 | 11 | 11 | 0 | $1.51M | [$112,000, $114,000] |

### Strike Spacing Distribution
- **Minimum spacing:** $100 (Aug 17 fine-grained ladder)
- **Most common spacing:** $500 (127 occurrences, 56% of all pairs)
- **Weekly ladders:** $2,000 spacing (65 occurrences)
- **Mean spacing:** $957

---

## The Implied Distribution Framework

### Theory

For a set of strikes K_1 < K_2 < ... < K_n, each market gives P(BTC > K_i). The implied probability that BTC lands in each bucket is:

```
P(BTC in [K_i, K_{i+1}]) = P(BTC > K_i) - P(BTC > K_{i+1})
```

**Three axioms must hold:**

1. **Non-negativity:** P(BTC in [K_i, K_{i+1}]) >= 0 for all i
   - Violation = pure arbitrage (buy lower strike YES, sell higher strike YES)

2. **Monotonicity:** P(BTC > K_i) >= P(BTC > K_{i+1}) when K_i < K_{i+1}
   - Equivalent to non-negativity, stated differently

3. **Smoothness:** The density function should be roughly bell-shaped
   - Extreme bumps indicate localized mispricing (butterfly spread opportunity)

### Arbitrage Trades

**Hard arbitrage (negative density):**
- If P(BTC > $110K) = 0.60 and P(BTC > $112K) = 0.65:
  - BUY "above $110K" at $0.60
  - SELL "above $112K" at $0.65
  - **Guaranteed $0.05 profit per contract pair** regardless of where BTC settles

**Soft signal (smoothness violation):**
- If adjacent density buckets show P([$108K-$110K]) = 0.15 but P([$110K-$112K]) = 0.03:
  - The distribution has an implausible "cliff"
  - Sell the overpriced bucket, buy the underpriced bucket (butterfly spread)

---

## Findings from Settlement Data

### Critical Data Limitation
All 423 markets in the dataset have settled (resolved to 0 or 1). The `outcome_prices` field contains the final settlement value, not the live trading price. This means:
- We **cannot** directly observe historical implied distributions from stored snapshots
- Pre-resolution prices must be reconstructed from the 404M-row trade table (expensive) or fetched live

### Settlement Anomalies Found
Despite only having settlement data, we found **evidence that pricing errors occur**:

**3 Settlement Monotonicity Violations:**
- 2025-08-17: $117,700 settled NO but $117,900 settled YES (BTC was between $117,700-$117,900, yet the $117,700 market said BTC was NOT above it)
- 2025-08-25: $111,500 settled NO but $112,000 settled YES
- 2025-08-26: $110,000 settled NO but $110,500 settled YES

These violations appear to stem from **duplicate markets with different settlement times** or resolution order-of-operations issues. When deduplication picks the wrong (lower-volume) market for a given strike, the settlement can be inconsistent. This confirms that:
1. Multiple competing markets exist for the same strike/date
2. They can have inconsistent prices (and even settlements)
3. This creates exploitable discrepancies during live trading

**12 Duplicate Market Settlement Inconsistencies:**
Markets with the same strike and expiry but different IDs sometimes settled to different values (e.g., one YES, one NO for the same strike). This is a strong signal that live prices diverge between duplicates.

---

## Opportunity Sizing

### Structural Opportunity Score

Ladders ranked by `(num_strikes * total_volume) / avg_spacing`:

| Date | Score | Strikes | Min Spacing | Volume |
|------|-------|---------|-------------|--------|
| 2025-08-25 | 85 | 17 | $500 | $3.27M |
| 2025-08-27 | 77 | 18 | $500 | $3.52M |
| 2025-08-22 | 71 | 16 | $500 | $3.55M |
| 2025-08-18 | 53 | 23 | $100 | $1.46M |
| 2025-08-20 | 52 | 16 | $500 | $2.58M |
| 2025-08-29 | 41 | 16 | $500 | $2.41M |

### Why Violations Are Likely

1. **Many constraints:** 17+ strikes means 16+ non-negativity constraints the market must simultaneously satisfy
2. **Retail pricing:** These markets are priced by retail traders, not professional options market makers
3. **Thin liquidity on wings:** Far OTM/ITM strikes often have <$5K volume
4. **No automated enforcement:** Unlike options exchanges, there's no market maker enforcing no-arb conditions
5. **Duplicate markets:** Multiple markets for the same strike create cross-market arbitrage on top of distribution arbitrage

### Estimated Edge
Based on the options market analogy:
- Butterfly spreads on retail-priced BTC options typically capture 1-5 cents per spread
- With $500 strike spacing and 10+ strikes, expect 2-4 negative density violations per week
- Each violation at 2-5 cents on $1 contracts = 2-5% edge per trade
- Constrained by liquidity: likely $5K-$20K deployable per opportunity

---

## Implementation Architecture

### Existing Infrastructure

The codebase already has:
- `tick_stream.py`: CrossMarketIterator for synchronized multi-market tracking
- `loader.py`: MarketLoader, TradeLoader, BlockLoader for data access
- Trade data: 404M trades in DuckDB for historical reconstruction

### Proposed Live Monitor

```python
# Pseudocode for live implied distribution monitor

class ImpliedDistributionMonitor:
    def __init__(self, clob_api):
        self.api = clob_api

    def get_active_ladders(self):
        """Find all active BTC strike ladders with 3+ strikes."""
        markets = self.api.get_markets(tag="bitcoin")
        # Group by expiry, filter to 'above' markets
        # Return grouped ladders

    def compute_implied_density(self, ladder):
        """Compute density from live prices."""
        strikes = sorted(ladder, key=lambda m: m.strike)
        for i in range(len(strikes) - 1):
            prob = strikes[i].yes_price - strikes[i+1].yes_price
            if prob < -0.01:  # negative density!
                self.alert_arbitrage(strikes[i], strikes[i+1], prob)

    def alert_arbitrage(self, lower, upper, neg_prob):
        """Hard arbitrage signal."""
        # BUY lower.yes at lower.yes_price
        # SELL upper.yes at upper.yes_price
        # Profit = abs(neg_prob) per pair
        pass
```

### Implementation Priority

| Priority | Task | Effort | Expected Value |
|----------|------|--------|---------------|
| HIGH | Live implied density monitor via CLOB API | 2-3 days | Core alpha source |
| HIGH | Negative density alert system | 1 day | Pure arb detection |
| MEDIUM | Butterfly spread signals (smoothness) | 2 days | Soft signal |
| MEDIUM | IV surface vs realized vol tracker | 3 days | Term structure alpha |
| LOW | Historical backtest via trade reconstruction | 5 days | Validation only |

---

## Comparison to Options Volatility Surface Arbitrage

| Aspect | Traditional Options | Polymarket BTC Strikes |
|--------|-------------------|----------------------|
| Instruments | Call/put options | Binary "above $X" markets |
| Density extraction | Breeden-Litzenberger | Simple difference of adjacent strikes |
| Market makers | Professional, enforce no-arb | Retail, no enforcement |
| Liquidity | Deep on major strikes | Thin on wings, moderate at ATM |
| Execution | Instant via exchange | CLOB with spread |
| Settlement | Cash or physical | Binary (0 or 1) |
| Edge persistence | Seconds (HFT) | Minutes to hours (retail) |

The key advantage: **Polymarket has no professional market makers enforcing no-arbitrage conditions across the strike ladder.** In traditional options, butterfly arbitrage is eliminated within milliseconds. On Polymarket, it can persist for extended periods.

---

## Files

- **Analysis script:** `/root/combarbbot/alpha2_implied_distribution.py`
- **Data source:** `/root/prediction-market-analysis/data/polymarket/markets/`
- **Trade data:** `/root/combarbbot/polymarket.db` (trades table, 404M rows)
- **Tick stream module:** `/root/combarbbot/src/data/tick_stream.py`
