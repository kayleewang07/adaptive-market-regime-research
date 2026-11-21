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
    return _snap_rounding_errors(result)


def _snap_rounding_errors(frame: pd.DataFrame, tolerance: float = 1e-9) -> pd.DataFrame:
    """Fix High/Low values that are off by floating-point noise only.

    Dividend-adjusted prices are rescaled independently, so High can end up ~1e-16
    below Close. Gaps within ``tolerance`` (relative) are snapped; anything larger is
    left alone for validation to reject.
    """
    prices = ["Open", "High", "Low", "Close"]
    if not set(prices) <= set(frame.columns):
        return frame
    frame[prices] = frame[prices].astype(float)
    others_high = frame[["Open", "Close", "Low"]].max(axis=1)
    others_low = frame[["Open", "Close", "High"]].min(axis=1)
    tiny_high = (frame["High"] < others_high) & (
        others_high - frame["High"] <= tolerance * others_high.abs()
    )
    tiny_low = (frame["Low"] > others_low) & (
        frame["Low"] - others_low <= tolerance * others_low.abs()
    )
    frame.loc[tiny_high, "High"] = others_high[tiny_high]
    frame.loc[tiny_low, "Low"] = others_low[tiny_low]
    return frame
