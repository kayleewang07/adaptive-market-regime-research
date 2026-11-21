"""Market-data provider adapters."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    """Anything that can return a daily OHLCV frame for one symbol."""

    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return daily OHLCV data for ``[start, end)``."""
        ...


class YFinanceProvider:
    """Download daily split- and dividend-adjusted OHLCV data through yfinance."""

    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download one symbol for ``[start, end)``; raise if nothing comes back."""
        import yfinance as yf

        frame: pd.DataFrame = yf.download(
            symbol, start=start, end=end, auto_adjust=True, progress=False
        )
        if frame is None or frame.empty:
            raise ValueError(
                f"No data returned for {symbol!r} between {start} and {end}. Check the "
                "ticker, the dates and your internet connection (Yahoo Finance must be reachable)."
            )
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame
