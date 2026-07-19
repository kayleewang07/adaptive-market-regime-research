"""Moving-average crossover strategy."""

import pandas as pd

from .base import Strategy


class MovingAverageCrossover(Strategy):
    """Hold the asset while a fast average exceeds a slow average."""

    name = "moving_average_crossover"

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        """Store moving-average windows."""
        self.fast, self.slow = fast, slow

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return lagged long/cash signals from close prices."""
        close = data["Close"]
        return (
            (close.rolling(self.fast).mean() > close.rolling(self.slow).mean())
            .astype(float)
            .shift(1)
            .fillna(0)
            .rename("target_weight")
        )
