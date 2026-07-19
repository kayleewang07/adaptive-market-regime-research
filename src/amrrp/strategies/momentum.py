"""Momentum strategy."""

import pandas as pd

from .base import Strategy


class Momentum(Strategy):
    """Hold the asset after positive trailing return."""

    name = "momentum"

    def __init__(self, lookback: int = 126) -> None:
        """Store momentum lookback sessions."""
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a lagged long/cash momentum signal."""
        return (
            (data["Close"].pct_change(self.lookback) > 0)
            .astype(float)
            .shift(1)
            .fillna(0)
            .rename("target_weight")
        )
