"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Compute standard portfolio and downside-risk metrics from periodic returns."""
    values = returns.dropna()
    if values.empty:
        return {
            key: float("nan")
            for key in ("annual_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown")
        }
    equity = (1 + values).cumprod()
    years = len(values) / periods_per_year
    annual = values.mean() * periods_per_year
    volatility = values.std(ddof=1) * np.sqrt(periods_per_year)
    downside = values[values < 0].std(ddof=1) * np.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1
    maximum = drawdown.min()
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years else float("nan")
    return {
        "annual_return": annual,
        "cagr": cagr,
        "sharpe": annual / volatility if volatility else np.nan,
        "sortino": annual / downside if downside else np.nan,
        "calmar": cagr / abs(maximum) if maximum else np.nan,
        "max_drawdown": maximum,
    }


def rolling_metrics(
    returns: pd.Series, window: int = 63, periods_per_year: int = 252
) -> pd.DataFrame:
    """Return rolling annual return, volatility, Sharpe, and drawdown series."""
    annual = returns.rolling(window).mean() * periods_per_year
    volatility = returns.rolling(window).std() * np.sqrt(periods_per_year)
    equity = (1 + returns).cumprod()
    return pd.DataFrame(
        {
            "annual_return": annual,
            "volatility": volatility,
            "sharpe": annual / volatility,
            "drawdown": equity / equity.cummax() - 1,
        }
    )
