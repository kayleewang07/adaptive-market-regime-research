"""Time-series momentum strategy."""

import pandas as pd

from .base import Strategy


class Momentum(Strategy):
    """Hold the asset while its trailing return is positive."""

    name = "momentum"

    def __init__(self, lookback: int = 126) -> None:
        """Store momentum lookback in trading sessions."""
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return long (1) / cash (0) weights decided at each close."""
        trailing = data["Close"].pct_change(self.lookback)
        return (trailing > 0).astype(float).rename("target_weight")
