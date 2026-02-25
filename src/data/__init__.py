"""Data loading and category management modules."""

from .category_index import CategoryIndex
from .loader import CategoryAwareMarketLoader

__all__ = ["CategoryIndex", "CategoryAwareMarketLoader"]
