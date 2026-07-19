"""Research plot generation."""

import pandas as pd


def equity_figure(returns: pd.DataFrame):
    """Return an interactive cumulative-equity Plotly figure."""
    import plotly.express as px

    return px.line((1 + returns).cumprod(), title="Strategy equity curves")


def correlation_heatmap(features: pd.DataFrame):
    """Return an interactive feature-correlation heatmap."""
    import plotly.express as px

    return px.imshow(features.corr(), title="Feature correlation")
