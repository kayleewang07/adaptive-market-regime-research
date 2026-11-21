"""Strategy comparison conditional on market regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from amrrp.backtesting.metrics import performance_metrics


def compare_by_regime(strategy_returns: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Compute each strategy's metrics using only the days labeled with each regime.

    Days within a regime are generally not contiguous, so drawdown and CAGR describe
    the returns earned *while* in that regime, stitched together.
    """
    joined = strategy_returns.join(regimes.rename("regime"), how="inner")
    rows = []
    for regime, group in joined.groupby("regime"):
        for strategy in strategy_returns.columns:
            rows.append(
                {
                    "regime": regime,
                    "strategy": strategy,
                    "days": len(group),
                    **performance_metrics(group[strategy]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["regime", "strategy"]).set_index(["regime", "strategy"])
    return pd.DataFrame(rows).set_index(["regime", "strategy"]).sort_index()


def regime_summary(asset_returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Describe the underlying asset's behaviour within each regime."""
    joined = pd.concat(
        [asset_returns.rename("ret"), regimes.rename("regime")], axis=1, join="inner"
    )
    grouped = joined.groupby("regime")["ret"]
    return pd.DataFrame(
        {
            "days": grouped.size(),
            "share_of_days": grouped.size() / len(joined),
            "annualized_return": grouped.mean() * 252,
            "annualized_volatility": grouped.std() * np.sqrt(252),
        }
    )
