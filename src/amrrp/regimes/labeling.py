"""Stable, interpretable regime labels."""

from __future__ import annotations

import pandas as pd


def label_regimes(regimes: pd.Series, returns: pd.Series) -> pd.Series:
    """Map numeric regimes to stable return/volatility descriptive labels."""
    stats = (
        pd.DataFrame({"regime": regimes, "returns": returns})
        .groupby("regime")["returns"]
        .agg(["mean", "std"])
    )
    ordered = stats.sort_values(["mean", "std"]).index
    labels = {regime: f"Regime {position + 1}" for position, regime in enumerate(ordered)}
    return regimes.map(labels).rename("regime_label")
