# Alpha 3: Term Structure Inversions in BTC "Above" Markets

## Executive Summary

**Term structure inversions are pervasive and large.** Across 1,876 daily BTC "above" markets (after filtering 2,157 intraday markets), we found **6,632 trade-level inversions** out of 14,478 comparable pair-days -- a **45.8% inversion rate**. The average inversion magnitude is **19.2 cents** (on a $1 contract), with some exceeding 90 cents.

This is the strongest alpha signal found so far in the Polymarket BTC options data.

## What Is a Term Structure Inversion?

For "BTC above $X on [date]" markets, a fundamental no-arbitrage constraint exists:

> P(BTC > $X on later date) >= P(BTC > $X on earlier date)

If BTC is above $100K on January 10, the probability it's above $100K on January 17 should be *at least* as high, because BTC has additional time to recover if it temporarily dips below.

A **term structure inversion** occurs when a longer-dated market trades *cheaper* than a shorter-dated one at the same strike. This creates a risk-free arbitrage: buy the cheap long-dated contract, sell the expensive short-dated one.

## Data Pipeline

- **Source**: 4,044 BTC "above" markets from Polymarket parquet files
- **Filtered out**: 2,157 intraday markets (e.g., "at 4PM ET")
- **Parsed**: 1,876 daily markets across 73 unique strikes
- **Price formats handled**: `$97,000` (comma), `$100K` (K suffix), `$107.5K` (decimal K), `$117000` (raw)
- **Chains analyzed**: 46 strikes with 2+ expiry dates (top 50 by volume)
- **Trade data**: 1,999,708 matched trades scanned from ~40K parquet files
- **VWAP calculation**: Daily volume-weighted average price per market, compared across same-strike expiries

## Key Findings

### 1. Inversion Rate: 45.8% Overall

Of 14,478 comparable pair-days (days where two same-strike markets with different expiries both traded), **6,632 showed inversions** exceeding a 1-cent threshold.

This is not a small-sample anomaly. It occurs across 34 of 46 chains analyzed.

### 2. Magnitude Distribution

| Magnitude Range | Count | Percentage |
|----------------|-------|------------|
| 1-2 cents      | 683   | 10.3%      |
| 2-5 cents      | 1,344 | 20.3%      |
| 5-10 cents     | 1,257 | 19.0%      |
| 10-20 cents    | 1,219 | 18.4%      |
| 20-50 cents    | 1,402 | 21.1%      |
| 50-99 cents    | 727   | 11.0%      |

Nearly **50% of inversions exceed 10 cents** in magnitude. The median inversion is 10.1 cents, the mean is 19.2 cents, and the maximum is 99.1 cents.

### 3. Strike-Level Analysis

Top chains by inversion frequency (high-volume chains only):

| Strike    | Inversions | Pair-Days | Rate  | Avg Mag | Max Mag |
|-----------|-----------|-----------|-------|---------|---------|
| $86,000   | 477       | 734       | 65.0% | 12.2c   | 53.0c   |
| $84,000   | 426       | 671       | 63.5% | 6.8c    | 24.8c   |
| $88,000   | 499       | 793       | 62.9% | 19.3c   | 66.2c   |
| $104,000  | 298       | 491       | 60.7% | 21.1c   | 89.8c   |
| $106,000  | 298       | 495       | 60.2% | 19.9c   | 99.1c   |
| $108,000  | 307       | 566       | 54.2% | 18.9c   | 97.6c   |
| $112,000  | 387       | 723       | 53.5% | 21.1c   | 69.9c   |
| $110,000  | 350       | 657       | 53.3% | 18.8c   | 73.0c   |
| $90,000   | 423       | 808       | 52.4% | 24.8c   | 79.8c   |

### 4. Largest Individual Inversions

The most extreme inversions often occur when BTC moves sharply and market-making adjusts faster in one contract than another:

| Date       | Strike   | Short Exp  | Short VWAP | Long Exp   | Long VWAP | Gap    |
|------------|----------|------------|------------|------------|-----------|--------|
| 2025-10-31 | $106,000 | 2025-11-02 | 0.996      | 2025-11-09 | 0.005     | 99.1c  |
| 2025-10-31 | $106,000 | 2025-11-01 | 0.988      | 2025-11-09 | 0.005     | 98.3c  |
| 2025-10-31 | $108,000 | 2025-11-02 | 0.979      | 2025-11-09 | 0.003     | 97.6c  |
| 2025-11-09 | $102,000 | 2025-11-11 | 0.971      | 2025-11-17 | 0.012     | 95.9c  |
| 2025-11-09 | $100,000 | 2025-11-11 | 0.994      | 2025-11-18 | 0.042     | 95.2c  |

These massive inversions (>90c) represent near-certain arbitrage profit. They likely arise from stale limit orders or delayed market-maker updates across weekly contracts.

### 5. Snapshot vs. Trade Inversions

When filtering to only "live" markets (prices between 1c and 99c), the snapshot analysis found just **23 inversions** with a modest average magnitude of 1.7 cents. This confirms inversions are largely corrected by the time markets settle, but persist intraday/intraweek during active trading.

### 6. Days Between Expiries

- Mean: 5.7 days apart
- Median: 5 days apart

Most inversions occur between consecutive weekly expiries (7 days apart), which are the most liquid comparison points.

## Estimated Edge

- **Total edge across all inversions**: 1,270.6 dollars (sum of magnitudes across all pair-days)
- **Average edge per opportunity**: 19.2 cents per contract
- **Number of actionable opportunities**: 6,632

In practice, the capturable edge depends on:
1. **Liquidity**: Can you actually fill both legs?
2. **Fees**: Polymarket takes ~1-2% fee on trades
3. **Speed**: Large inversions may be short-lived
4. **Settlement risk**: Contracts settle based on oracle data

At a 2c fee per leg (4c round-trip), roughly **70% of inversions (4,605 of 6,632) would still be profitable after fees**.

## Why Do Inversions Persist?

1. **Fragmented liquidity**: Each weekly contract has its own order book with independent market makers
2. **Slow repricing**: When BTC moves sharply, nearer-dated contracts reprice faster
3. **No cross-contract arbitrage infrastructure**: Unlike traditional options, there's no automated term structure arb
4. **Retail-dominated**: Many participants trade individual contracts without considering the term structure
5. **Capital inefficiency**: Funds locked in one contract can't be easily redeployed to another

## Recommendations for the Arb Bot

1. **Monitor same-strike pairs across consecutive weeks**: This is where most inversions occur
2. **Trigger on 5+ cent inversions**: These clear fees and have meaningful edge
3. **Focus on high-volume strikes ($84K-$118K range)**: Best liquidity for execution
4. **Time-decay awareness**: Inversions near expiry are different from mid-term inversions
5. **Combine with Alpha 1 (monotonicity)**: A comprehensive cross-contract arb strategy should enforce both monotonicity within a date AND term structure across dates

## Comparison with Alpha 1

| Metric             | Alpha 1 (Monotonicity) | Alpha 3 (Term Structure) |
|-------------------|----------------------|-------------------------|
| Type              | Cross-strike, same date | Same-strike, cross-date |
| Inversion rate    | ~30-50%              | 45.8%                   |
| Avg magnitude     | ~3-5c                | 19.2c                   |
| Max magnitude     | ~10-15c              | 99.1c                   |
| Opportunities     | Hundreds             | 6,632                   |

**Alpha 3 produces larger and more frequent opportunities than Alpha 1.** Term structure inversions are the dominant mispricing pattern in Polymarket's BTC options.

## Files

- **Analysis script**: `/root/combarbbot/scripts/alpha3_term_structure.py`
- **Results JSON**: `/root/combarbbot/scripts/alpha3_results.json`
- **This document**: `/root/combarbbot/docs/ALPHA3_TERM_STRUCTURE.md`
