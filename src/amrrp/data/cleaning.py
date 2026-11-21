"""Market-data cleaning and normalization."""

from __future__ import annotations

import pandas as pd


def clean_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, timezone-naive OHLCV data with a unique datetime index."""
    if frame.empty:
        raise ValueError("Market data is empty.")
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_convert(None)
    result.index = index.normalize()
    result.index.name = "Date"
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result.columns = [str(column).title() for column in result.columns]
    return result
