"""Backtest runner for arbitrage detection on historical data."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from ..data.category_index import CategoryType, SubcategoryType
from ..data.loader import CategoryAwareMarketLoader, Market, Trade
from ..optimizer import find_arbitrage, detect_arbitrage_simple
from ..optimizer.schema import ArbitrageResult, MarginalArbitrageResult, RelationshipGraph


@dataclass
class BacktestConfig:
    category: CategoryType | None = None
    subcategory: SubcategoryType | None = None
    market_ids: list[str] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    time_step: timedelta = field(default_factory=lambda: timedelta(hours=1))
    min_volume: float = 0
    arbitrage_threshold: float = 0.01


@dataclass
class ArbitrageOpportunity:
    timestamp: datetime
    market_ids: list[str]
    kl_divergence: float
    market_prices: dict[str, float]
    coherent_prices: dict[str, float]


@dataclass
class BacktestResult:
    config: BacktestConfig
    opportunities: list[ArbitrageOpportunity] = field(default_factory=list)
    total_snapshots: int = 0
    markets_analyzed: int = 0
    run_time_seconds: float = 0.0

    @property
    def opportunity_count(self) -> int:
        return len(self.opportunities)


class BacktestRunner:
    """Run arbitrage detection backtests on historical data."""

    def __init__(self, loader: CategoryAwareMarketLoader | None = None):
        self.loader = loader or CategoryAwareMarketLoader()

    def run(self, config: BacktestConfig, relationships: RelationshipGraph | None = None) -> BacktestResult:
        import time
        start = time.time()
        
        markets = self._get_markets(config)
        if not markets:
            return BacktestResult(config=config)
        
        market_ids = [m.id for m in markets]
        trades = self.loader.get_trades(market_ids, config.start_time, config.end_time)
        
        if not trades:
            return BacktestResult(config=config, markets_analyzed=len(markets))
        
        snapshots = self._create_snapshots(trades, config.time_step)
        opportunities = []
        
        for timestamp, prices in snapshots.items():
            result = self._detect_arbitrage(prices, relationships)
            if result and result.kl_divergence >= config.arbitrage_threshold:
                opportunities.append(ArbitrageOpportunity(
                    timestamp=timestamp, market_ids=list(prices.keys()),
                    kl_divergence=result.kl_divergence,
                    market_prices=result.market_prices,
                    coherent_prices=result.coherent_prices
                ))
        
        return BacktestResult(
            config=config, opportunities=opportunities,
            total_snapshots=len(snapshots), markets_analyzed=len(markets),
            run_time_seconds=time.time() - start
        )

    def run_simple(self, market_prices: dict[str, list[float]], 
                   implications: list[tuple[str, str]] | None = None) -> MarginalArbitrageResult:
        return detect_arbitrage_simple(market_prices, implications or [])

    def _get_markets(self, config: BacktestConfig) -> list[Market]:
        if config.market_ids:
            return self.loader.get_markets_batch(config.market_ids)
        if config.category:
            return self.loader.query_by_category(config.category, config.subcategory, config.min_volume)
        return []

    def _create_snapshots(self, trades: list[Trade], time_step: timedelta) -> dict[datetime, dict[str, float]]:
        snapshots = {}
        current_prices = {}
        if not trades:
            return snapshots
        
        current_bucket = trades[0].timestamp.replace(minute=0, second=0, microsecond=0)
        max_time = trades[-1].timestamp
        trade_idx = 0
        
        while current_bucket <= max_time:
            bucket_end = current_bucket + time_step
            while trade_idx < len(trades) and trades[trade_idx].timestamp < bucket_end:
                current_prices[trades[trade_idx].market_id] = trades[trade_idx].price
                trade_idx += 1
            if current_prices:
                snapshots[current_bucket] = dict(current_prices)
            current_bucket = bucket_end
        
        return snapshots

    def _detect_arbitrage(self, prices: dict[str, float], relationships: RelationshipGraph | None) -> ArbitrageResult | None:
        if len(prices) < 2:
            return None
        try:
            return find_arbitrage(prices, relationships)
        except Exception:
            return None
