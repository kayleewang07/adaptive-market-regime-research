"""Regime-model visualizations."""

from __future__ import annotations

from typing import Any

import pandas as pd


def regime_figure(regimes: pd.Series) -> Any:
    """Build an interactive Plotly regime timeline."""
    import plotly.express as px

    frame = regimes.rename("regime").to_frame().reset_index()
    frame.columns = ["date", "regime"]
    frame["regime"] = frame["regime"].astype(str)
    return px.scatter(frame, x="date", y="regime", color="regime", title="Market regime history")
