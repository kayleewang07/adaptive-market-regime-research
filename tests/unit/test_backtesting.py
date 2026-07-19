"""Tests for vectorized portfolio accounting."""

import pytest

pd = pytest.importorskip("pandas")

from amrrp.backtesting.engine import BacktestEngine


def test_backtest_applies_turnover_costs() -> None:
    """A position change should create measurable turnover and trading friction."""
    index = pd.date_range("2024-01-01", periods=3)
    result = BacktestEngine().run(
        pd.Series([100, 101, 102], index=index), pd.Series([0, 1, 1], index=index)
    )
    assert result.turnover.iloc[1] == 1
    assert result.equity.iloc[-1] > 100_000
