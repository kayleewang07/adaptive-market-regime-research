"""Validated market-data access layer."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .cache import MarketDataCache
from .cleaning import clean_market_frame
from .providers import YFinanceProvider
from .validation import validate_market_frame


class MarketDataRepository:
    """Fetch, clean, validate, and cache market data."""

    def __init__(self, provider: YFinanceProvider, cache: MarketDataCache) -> None:
        """Configure the provider and cache used by the repository."""
        self.provider, self.cache = provider, cache

    def get(self, symbol: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
        """Return a validated daily OHLCV frame for one symbol."""
        frame = None if refresh else self.cache.load(symbol, start, end)
        if frame is None:
            frame = self.provider.download(symbol, start, end)
            frame = clean_market_frame(frame)
            validate_market_frame(frame)
            self.cache.save(frame, symbol, start, end)
        else:
            frame = clean_market_frame(frame)
            validate_market_frame(frame)
        return frame

    def get_many(self, symbols: Iterable[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Return validated market frames keyed by symbol."""
        return {symbol: self.get(symbol, start, end) for symbol in symbols}
