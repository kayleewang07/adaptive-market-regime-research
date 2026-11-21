"""Mean-reversion strategy."""

import pandas as pd

from .base import Strategy


class MeanReversion(Strategy):
    """Go long on oversold closes and exit once price reverts to its average.

    Enter when the rolling z-score of the close falls below ``entry_z``; stay long until
    it rises back above ``exit_z``.
    """

    name = "mean_reversion"

    def __init__(self, window: int = 20, entry_z: float = -1.5, exit_z: float = 0.0) -> None:
        """Store the z-score horizon and entry/exit thresholds."""
        if entry_z >= exit_z:
            raise ValueError("entry_z must be below exit_z.")
        self.window, self.entry_z, self.exit_z = window, entry_z, exit_z

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return long (1) / cash (0) weights decided at each close."""
        close = data["Close"]
        z = (close - close.rolling(self.window).mean()) / close.rolling(self.window).std()
        state = pd.Series(float("nan"), index=data.index)
        state[z < self.entry_z] = 1.0
        state[z > self.exit_z] = 0.0
        return state.ffill().fillna(0.0).rename("target_weight")
