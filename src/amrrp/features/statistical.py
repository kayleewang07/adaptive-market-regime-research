"""Statistical feature transformers."""

from __future__ import annotations

import pandas as pd

from .base import DataFrameTransformer


class StatisticalFeatures(DataFrameTransformer):
    """Generate configurable distributional and drawdown features."""

    def __init__(self, window: int = 63) -> None:
        """Store the rolling observation window."""
        self.window = window

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return lag-safe return distribution and drawdown features."""
        returns, close = X["Close"].pct_change(), X["Close"].astype(float)
        peak = close.cummax()
        return pd.DataFrame(
            {
                "drawdown": close / peak - 1,
                "return_skew": returns.rolling(self.window).skew(),
                "return_kurtosis": returns.rolling(self.window).kurt(),
            },
            index=X.index,
        ).shift(1)
