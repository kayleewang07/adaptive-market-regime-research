"""Regime-transition analysis."""

import pandas as pd


def transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Return row-normalized one-step transition probabilities (row = from, column = to)."""
    previous = regimes.shift().rename("from")
    current = regimes.rename("to")
    valid = previous.notna()
    return pd.crosstab(previous[valid].astype(int), current[valid], normalize="index")
