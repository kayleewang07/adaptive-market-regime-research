"""Aggregate and rolling performance analysis."""

import pandas as pd

from amrrp.backtesting.metrics import performance_metrics


def compare_strategies(returns: pd.DataFrame) -> pd.DataFrame:
    """Return a metric table with one row per strategy return series."""
    return pd.DataFrame({name: performance_metrics(returns[name]) for name in returns}).T
