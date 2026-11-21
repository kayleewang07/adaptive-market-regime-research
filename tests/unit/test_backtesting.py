"""Tests for vectorized portfolio accounting."""

import pandas as pd
import pytest

from amrrp.backtesting.engine import BacktestEngine
from amrrp.core.types import BacktestConfig


def test_weight_earns_the_next_sessions_return_and_pays_costs() -> None:
    """A weight decided at close t earns t->t+1, and each trade costs 7 bps by default."""
    index = pd.date_range("2024-01-01", periods=3)
    prices = pd.Series([100.0, 110.0, 121.0], index=index)
    result = BacktestEngine().run(prices, pd.Series([1.0, 0.0, 0.0], index=index))
    cost = 7 / 10_000
    assert result.returns.tolist() == pytest.approx([-cost, 0.10 - cost, 0.0])
    assert result.turnover.tolist() == [1.0, 1.0, 0.0]


def test_zero_cost_buy_and_hold_tracks_the_asset() -> None:
    """With no friction, a constant full position compounds to the price ratio."""
    index = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([100.0, 102.0, 99.0, 105.0], index=index)
    engine = BacktestEngine(
        BacktestConfig(initial_capital=1.0, transaction_cost_bps=0, slippage_bps=0)
    )
    result = engine.run(prices, pd.Series(1.0, index=index))
    assert result.equity.iloc[-1] == pytest.approx(105 / 100)


def test_weights_are_clipped_to_unit_leverage() -> None:
    """Target weights outside [-1, 1] are capped."""
    index = pd.date_range("2024-01-01", periods=2)
    result = BacktestEngine().run(
        pd.Series([1.0, 2.0], index=index), pd.Series([3.0, -5.0], index=index)
    )
    assert result.weights.tolist() == [1.0, -1.0]
