"""Buy-and-hold strategy."""

import pandas as pd

from .base import Strategy


class BuyAndHold(Strategy):
    """Remain fully invested from the first close onward."""

    name = "buy_and_hold"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a constant long target weight."""
        return pd.Series(1.0, index=data.index, name="target_weight")
