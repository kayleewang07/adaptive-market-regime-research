"""Regime-transition analysis."""

import pandas as pd


def transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Return row-normalized one-step regime transition probabilities."""
    return pd.crosstab(regimes.shift(), regimes, normalize="index")
