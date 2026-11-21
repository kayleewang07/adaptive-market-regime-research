"""Volatility-breakout strategy."""

import pandas as pd

from .base import Strategy


class VolatilityBreakout(Strategy):
    """Go long when price breaks above an upper volatility band.

    Enter when the close exceeds ``mean + multiplier * std`` over ``window`` sessions and
    stay long until the close falls back below the rolling mean.
    """

    name = "volatility_breakout"

    def __init__(self, window: int = 20, multiplier: float = 2.0) -> None:
        """Store breakout-band parameters."""
        self.window, self.multiplier = window, multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return long (1) / cash (0) weights decided at each close."""
        close = data["Close"]
        mean, std = close.rolling(self.window).mean(), close.rolling(self.window).std()
        state = pd.Series(float("nan"), index=data.index)
        state[close > mean + self.multiplier * std] = 1.0
        state[close < mean] = 0.0
        return state.ffill().fillna(0.0).rename("target_weight")
