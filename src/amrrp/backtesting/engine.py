"""Daily event-driven backtesting engine with vectorized accounting."""

from __future__ import annotations

import pandas as pd

from amrrp.core.types import BacktestConfig, BacktestResult

from .metrics import performance_metrics


class BacktestEngine:
    """Simulate daily close-to-close strategy performance and trading friction."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Use supplied or default execution assumptions."""
        self.config = config or BacktestConfig()

    def run(self, prices: pd.Series, target_weights: pd.Series) -> BacktestResult:
        """Process daily position events and return portfolio history and metrics."""
        prices, target_weights = prices.astype(float).align(
            target_weights.astype(float), join="inner"
        )
        weights = target_weights.clip(-1, 1).fillna(0)
        turnover = weights.diff().abs().fillna(weights.abs())
        gross = weights.shift(1).fillna(0) * prices.pct_change().fillna(0)
        friction = turnover * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000
        returns = (gross - friction).rename("portfolio_return")
        equity = (1 + returns).cumprod().mul(self.config.initial_capital).rename("equity")
        return BacktestResult(
            equity,
            returns,
            weights.rename("weight"),
            turnover.rename("turnover"),
            performance_metrics(returns, self.config.periods_per_year),
        )
