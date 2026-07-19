"""Research table generation."""

import pandas as pd


def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return presentation-ready rounded performance metrics."""
    return metrics.copy().round(4)
