"""Daily close-to-close backtesting engine with vectorized accounting."""

from __future__ import annotations

import pandas as pd

from amrrp.core.types import BacktestConfig, BacktestResult

from .metrics import performance_metrics


class BacktestEngine:
    """Simulate a single-asset daily strategy with proportional trading costs.

    Timing: ``target_weights[t]`` is decided at the close of ``t`` and earns the
    close-to-close return from ``t`` to ``t + 1``. Changing the weight at ``t`` costs
    ``|w[t] - w[t-1]| * (transaction_cost_bps + slippage_bps) / 10_000`` of equity.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Use supplied or default execution assumptions."""
        self.config = config or BacktestConfig()

    def run(self, prices: pd.Series, target_weights: pd.Series) -> BacktestResult:
        """Return the portfolio's equity, returns, weights, turnover and metrics."""
        prices, target_weights = prices.astype(float).align(
            target_weights.astype(float), join="inner"
        )
        weights = target_weights.clip(-1, 1).fillna(0.0)
        turnover = weights.diff().abs().fillna(weights.abs())
        gross = weights.shift(1).fillna(0.0) * prices.pct_change().fillna(0.0)
        cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000
        returns = (gross - turnover * cost_rate).rename("portfolio_return")
        equity = (1 + returns).cumprod().mul(self.config.initial_capital).rename("equity")
        return BacktestResult(
            equity,
            returns,
            weights.rename("weight"),
            turnover.rename("turnover"),
            performance_metrics(returns, self.config.periods_per_year),
        )
