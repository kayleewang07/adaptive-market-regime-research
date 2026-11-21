"""Strategies must be causal and must not double-lag their signals."""

import pandas as pd
import pytest

from amrrp.experiments.registry import STRATEGIES
from amrrp.strategies.ma_crossover import MovingAverageCrossover
from amrrp.strategies.momentum import Momentum


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_signals_do_not_use_future_prices(name: str, market: pd.DataFrame) -> None:
    """Changing prices after a cutoff must not change any signal before it."""
    cutoff = market.index[1000]
    altered = market.copy()
    altered.loc[altered.index > cutoff, ["Open", "High", "Low", "Close"]] *= 0.5
    strategy = STRATEGIES[name]()
    original = strategy.generate_signals(market).loc[:cutoff]
    changed = strategy.generate_signals(altered).loc[:cutoff]
    pd.testing.assert_series_equal(original, changed)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_signals_are_long_or_flat(name: str, market: pd.DataFrame) -> None:
    """All bundled strategies are long-only with weights in {0, 1}."""
    signals = STRATEGIES[name]().generate_signals(market)
    assert signals.index.equals(market.index)
    assert set(signals.unique()) <= {0.0, 1.0}


def test_moving_average_signal_is_not_lagged_twice() -> None:
    """The signal at t reflects the crossover state at t; the engine adds the one-day delay."""
    index = pd.date_range("2024-01-01", periods=6)
    data = pd.DataFrame({"Close": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]}, index=index)
    signals = MovingAverageCrossover(fast=1, slow=3).generate_signals(data)
    assert signals.iloc[3] == 1.0  # fast (20) > slow (13.3) on the day of the jump


def test_momentum_follows_trailing_return() -> None:
    """Momentum is long exactly when the trailing return is positive."""
    index = pd.date_range("2024-01-01", periods=5)
    data = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 11.0, 9.0]}, index=index)
    assert Momentum(lookback=2).generate_signals(data).tolist() == [0, 0, 1, 0, 0]


def test_invalid_parameters_are_rejected() -> None:
    """Nonsensical windows fail loudly."""
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=50, slow=20)
