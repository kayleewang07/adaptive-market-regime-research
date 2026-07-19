"""Market-data validation rules."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def validate_market_frame(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` if an OHLCV frame violates the platform data contract."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("Data must have a unique DatetimeIndex.")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("OHLCV data contains missing values.")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    if (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("High price is below an intraday price.")
    if (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Low price is above an intraday price.")
