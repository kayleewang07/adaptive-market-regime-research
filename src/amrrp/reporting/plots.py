"""Research plot generation."""

from typing import Any

import pandas as pd


def equity_figure(returns: pd.DataFrame) -> Any:
    """Return an interactive compounded-equity Plotly figure (growth of $1)."""
    import plotly.express as px

    return px.line((1 + returns.fillna(0)).cumprod(), title="Growth of $1")


def correlation_heatmap(features: pd.DataFrame) -> Any:
    """Return an interactive feature-correlation heatmap."""
    import plotly.express as px

    return px.imshow(features.corr(), title="Feature correlation")
