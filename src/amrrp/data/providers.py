"""Market-data provider adapters."""

from __future__ import annotations

import pandas as pd


class YFinanceProvider:
    """Download daily adjusted OHLCV data through yfinance."""

    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download one symbol for ``[start, end]``; end is converted for yfinance."""
        try:
            import yfinance as yf
        except ImportError as error:
            raise ImportError("Install yfinance to download market data.") from error
        frame = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame
