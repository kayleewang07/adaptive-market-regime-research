"""Strategy comparison conditional on market regime."""

from __future__ import annotations

import pandas as pd

from amrrp.backtesting.metrics import performance_metrics


def compare_by_regime(strategy_returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Compute each strategy's performance metrics within every observed regime."""
    joined = strategy_returns.join(regimes.rename("regime"), how="inner")
    rows = []
    for regime, group in joined.groupby("regime"):
        for strategy in strategy_returns.columns:
            rows.append(
                {"regime": regime, "strategy": strategy, **performance_metrics(group[strategy])}
            )
    return pd.DataFrame(rows).set_index(["regime", "strategy"]).sort_index()
