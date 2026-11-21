"""Portfolio performance metrics.

All ratios assume a zero risk-free rate. ``annual_return`` is the arithmetic mean daily
return scaled by ``periods_per_year``; ``cagr`` is the geometric (compounded) rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = (
    "annual_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
)


def performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Compute standard portfolio and downside-risk metrics from periodic returns."""
    values = returns.dropna()
    if len(values) < 2:
        return dict.fromkeys(METRIC_NAMES, float("nan"))
    equity = (1 + values).cumprod()
    years = len(values) / periods_per_year
    annual = float(values.mean() * periods_per_year)
    volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    # Downside deviation: root-mean-square of negative returns over all periods.
    downside = float(np.sqrt((values.clip(upper=0) ** 2).mean() * periods_per_year))
    maximum = float((equity / equity.cummax() - 1).min())
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    return {
        "annual_return": annual,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": annual / volatility if volatility > 0 else float("nan"),
        "sortino": annual / downside if downside > 0 else float("nan"),
        "calmar": cagr / abs(maximum) if maximum < 0 else float("nan"),
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
