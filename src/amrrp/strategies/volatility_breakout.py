"""Volatility-breakout strategy."""

import pandas as pd

from .base import Strategy


class VolatilityBreakout(Strategy):
    """Enter long when price clears an upper rolling volatility band."""

    name = "volatility_breakout"

    def __init__(self, window: int = 20, multiplier: float = 2.0) -> None:
        """Store breakout-band parameters."""
        self.window, self.multiplier = window, multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a lagged upper-band breakout signal."""
        close = data["Close"]
        upper = (
            close.rolling(self.window).mean() + self.multiplier * close.rolling(self.window).std()
        )
        return (close > upper).astype(float).shift(1).fillna(0).rename("target_weight")
