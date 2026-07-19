"""Regime-model visualizations."""

from __future__ import annotations

import pandas as pd


def regime_figure(regimes: pd.Series):
    """Build an interactive Plotly regime timeline."""
    try:
        import plotly.express as px
    except ImportError as error:
        raise ImportError("Install plotly to visualize regimes.") from error
    frame = regimes.rename("regime").to_frame().reset_index()
    frame.columns = ["date", "regime"]
    return px.scatter(frame, x="date", y="regime", color="regime", title="Market regime history")
