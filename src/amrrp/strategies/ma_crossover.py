"""Moving-average crossover strategy."""

import pandas as pd

from .base import Strategy


class MovingAverageCrossover(Strategy):
    """Hold the asset while a fast moving average is above a slow one."""

    name = "moving_average_crossover"

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        """Store moving-average windows."""
        if fast >= slow:
            raise ValueError("The fast window must be shorter than the slow window.")
        self.fast, self.slow = fast, slow

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return long (1) / cash (0) weights decided at each close."""
        close = data["Close"]
        signal = close.rolling(self.fast).mean() > close.rolling(self.slow).mean()
        return signal.astype(float).rename("target_weight")
