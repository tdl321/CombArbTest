# Alpha Source 1: Monotonicity Violations in BTC Strike Ladders

## Overview

Polymarket runs daily "Bitcoin above $X on [date]?" binary markets at multiple strike prices.
Mathematical constraint: if strike_A > strike_B, then P(BTC > strike_A) <= P(BTC > strike_B).
Violations represent risk-free arbitrage opportunities.

## Data Summary

- **Total BTC threshold markets found**: 2,716 (filtered to vol > $1K, above direction only)
- **Markets with valid strike (>= $10K) and parsed expiry**: 2,414
- **Ladders with 3+ unique strikes**: 166
- **Ladders analyzed at trade level (top 15 by volume)**: 14

## Critical Finding: Two Types of Violations

### Type 1: Spurious (Dominant, ~95% of violations)

The largest violations (magnitudes 0.5-0.98) are caused by **weekly highlight markets** being
grouped into the same ladder as the standard strikes. For each expiry, Polymarket often has:

- Standard ladder: $80K, $82K, $84K, ..., $98K (even $2K increments, moderate volume)
- Weekly highlight: a single prominent strike like $94K, $95K, $100K, $103K, $105K 
  (very high volume, 10-20x the ladder markets)

The highlight market prices independently of the ladder -- it has its own liquidity pool and
market dynamics. When the highlight strike falls between ladder strikes, it creates apparent
"violations" that are not actually arbitrageable because they are separate market ecosystems.

**Evidence**: On Jan 24, the $103K market had 17,397 price observations while adjacent
$102K had only 30 and $104K had 22. The $103K market traded at 0.99 while $96K traded
at 0.44 -- this isn't a pricing error, the $103K market is the weekly flagship with
independent market-making.

Affected ladders and their highlight strikes:
| Expiry | Highlight Strike | Violation Count | Notes |
|--------|-----------------|-----------------|-------|
| Jan 24 | $103,000 | 16,590 | 17K trades vs 22-30 for neighbors |
| Jan 3 | $94,000 | 11,047 | 24K trades vs 74-1925 for neighbors |
| Dec 6 | $98,000 | 11,706 | Flagship market |
| Dec 13 | $100,000 | 7,503 | Flagship market |
| Jan 17 | $95,000 | 6,660 | Flagship market |
| Jan 31 | $105,000 | 1,031 | Flagship market |

### Type 2: Genuine Violations (~5% of count, the real signal)

After filtering out highlight-market artifacts, genuine violations exist between
standard ladder strikes. These are smaller but potentially tradeable:

| Expiry | Pair | Gap | Observation |
|--------|------|-----|-------------|
| Nov 29 | $88K@0.39 < $90K@0.45 | 6.0c | Persistent across ~188 checks |
| Dec 27 | $97K@0.009 < $98K@0.08 | 7.1c | Low-probability tail strikes |
| Jan 10 | $97K@0.006 < $98K@0.10 | 9.4c | Same tail pattern |
| Dec 20 | $82K@0.887 < $84K@0.91 | 2.3c | Tight, near fees threshold |
| Jan 23 | $86K@0.907 < $88K@0.93 | 2.3c | Tight, near fees threshold |
| Jan 23 | $94K@0.40 < $96K@0.43 | 3.0c | Moderate gap |

**Key pattern**: Genuine violations tend to appear at:
1. **Tail strikes** (very low probability, < 10c): $97K vs $98K when both are unlikely.
   These have poor liquidity and wide spreads, making arb execution difficult.
2. **ATM strikes** (near current BTC price): $88K vs $90K. More liquid but gaps are 
   small (2-6c), often at or below fee thresholds.

## Quantified Opportunity

### All Violations (Including Spurious)
- **Overall violation rate**: 3.26% of adjacent-pair checks
- **Total violations**: 56,169 across 1.72M checks
- **Weighted avg magnitude**: 0.84 (dominated by spurious Type 1)

### Genuine Violations Only (Estimated)
- **Genuine violation rate**: ~0.2-0.3% of checks
- **Typical magnitude**: 2-10 cents
- **Persistence**: Violations persist across multiple blocks (10-50+ trades)
- **Affected ladders**: ~4-5 out of 14 analyzed (Dec 20, Dec 27, Jan 10, Jan 23, Nov 29)

### Profitability Assessment

| Factor | Value | Impact |
|--------|-------|--------|
| Typical gap | 2-10c | Revenue per share |
| Polymarket fee | ~1-2% per side | 2-4c cost on 50c positions |
| Spread cost | ~1-3c per side | Need to cross spread |
| Min profitable gap | ~5-8c | After all costs |
| Violations > 5c | ~30% of genuine | Small subset tradeable |
| Liquidity at violation | Low-moderate | $50-$500 per trade |
| Duration | 10-50 blocks (~2-10 min) | Tight execution window |

**Expected value per opportunity**: ~2-5c profit x $100-500 size = $2-25 per trade
**Frequency**: ~1-3 genuine violations per ladder per day
**Daily potential**: $10-75 across all active ladders

## Conclusion

**Verdict: Low-yield alpha, not worth building a dedicated system for.**

The monotonicity violations that exist in BTC strike ladders fall into two categories:
1. Spurious violations from mixed market types (not tradeable)
2. Genuine violations that are typically 2-10c, persist briefly, and occur at 
   illiquid strikes or near-ATM strikes where fees eat most of the profit

The opportunity is real but small. It would be more practical as an **addition to an 
existing market-making system** rather than a standalone strategy. A market maker already
quoting these strikes could add monotonicity enforcement as a secondary signal.

### Recommendations
1. **Do not build standalone arb bot** -- returns don't justify the engineering
2. **Add as signal to broader system** -- if building a BTC binary market maker,
   use monotonicity violations as a secondary pricing constraint
3. **Monitor for regime changes** -- if Polymarket adds more strike granularity or
   the markets get less efficient, revisit this analysis
4. **Better opportunity**: Look at cross-expiry violations or volatility surface
   inconsistencies which may have larger magnitudes

## Technical Notes

- Strike parsing: regex on market questions, filtered to >= $10K
- Expiry parsing: "on [Month] [Day]" pattern  
- Deduplication: when multiple markets share a strike, kept highest volume
- Trade prices: maker_amount / taker_amount, normalized to [0,1]
- Violation threshold: 2 cents (to filter noise)
- Only adjacent strike pairs checked (most conservative measure)
- Data source: Polymarket on-chain trade data via parquet files
- Analysis covers Nov 2024 - Feb 2025 data

Generated: 2025-02-27
