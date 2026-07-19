"""Mean-reversion strategy."""

import pandas as pd

from .base import Strategy


class MeanReversion(Strategy):
    """Enter long after unusually negative rolling z-scores."""

    name = "mean_reversion"

    def __init__(self, window: int = 20, entry_z: float = -1.5) -> None:
        """Store the z-score horizon and long-entry threshold."""
        self.window, self.entry_z = window, entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a lagged long/cash mean-reversion signal."""
        close = data["Close"]
        z = (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()
        return (z < self.entry_z).astype(float).shift(1).fillna(0).rename("target_weight")
