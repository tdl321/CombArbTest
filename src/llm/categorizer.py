"""Market categorization using rules or LLM."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MarketInfo:
    """Minimal market info needed for categorization."""
    id: str
    question: str
    slug: str


@dataclass
class CategoryResult:
    """Result of categorization."""
    market_id: str
    category: str
    subcategory: str
    confidence: float = 1.0


class RuleBasedCategorizer:
    """Fast rule-based categorizer using keyword matching."""

    RULES: list[tuple[list[str], str, str]] = [
        (["trump", "biden", "election", "president", "republican", "democrat", "gop", "dnc"], "politics", "us-election"),
        (["congress", "senate", "house of rep", "speaker"], "politics", "us-congress"),
        (["nba", "basketball", "lebron", "lakers", "celtics"], "sports", "nba"),
        (["nfl", "football", "super bowl", "touchdown"], "sports", "nfl"),
        (["mlb", "baseball", "home run", "pitcher"], "sports", "mlb"),
        (["nhl", "hockey", "stanley cup"], "sports", "nhl"),
        (["soccer", "fifa", "premier league", "world cup"], "sports", "soccer"),
        (["ufc", "mma", "boxing", "fight"], "sports", "mma"),
        (["bitcoin", "btc"], "crypto", "bitcoin"),
        (["ethereum", "eth", "vitalik"], "crypto", "ethereum"),
        (["solana", "xrp", "doge", "crypto", "token", "coin"], "crypto", "altcoins"),
        (["fed", "federal reserve", "interest rate", "fomc"], "finance", "fed"),
        (["stock", "sp500", "nasdaq", "dow"], "finance", "markets"),
        (["temperature", "weather", "rain", "snow", "hurricane"], "weather", "temperature"),
        (["oscar", "grammy", "emmy", "golden globe"], "entertainment", "awards"),
        (["spacex", "nasa", "rocket", "mars"], "science", "space"),
        (["ai", "gpt", "openai", "chatgpt"], "science", "ai"),
    ]

    def categorize(self, market: MarketInfo) -> CategoryResult:
        text = f"{market.question} {market.slug}".lower()
        for keywords, category, subcategory in self.RULES:
            if any(kw in text for kw in keywords):
                return CategoryResult(market.id, category, subcategory, 0.9)
        return CategoryResult(market.id, "other", "misc", 0.5)

    def categorize_batch(self, markets: list[MarketInfo]) -> list[CategoryResult]:
        return [self.categorize(m) for m in markets]


class MarketCategorizer:
    """Hybrid categorizer (rule-based with optional LLM fallback)."""

    def __init__(self, use_llm_fallback: bool = False):
        self.rule_categorizer = RuleBasedCategorizer()
        self.use_llm_fallback = use_llm_fallback

    def categorize(self, market: MarketInfo) -> CategoryResult:
        return self.rule_categorizer.categorize(market)

    def categorize_batch(self, markets: list[MarketInfo]) -> list[CategoryResult]:
        return self.rule_categorizer.categorize_batch(markets)
