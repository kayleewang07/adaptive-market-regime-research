"""Regime-aware allocation."""

import numpy as np
import pandas as pd

from amrrp.strategies.regime_switching import (
    CASH,
    regime_switching_weights,
    select_strategy_per_regime,
)


def test_selection_picks_best_strategy_cash_or_default_per_regime() -> None:
    """Best Sharpe wins; all-negative regimes go to cash; rare regimes use the default."""
    rng = np.random.default_rng(0)
    regimes = pd.Series([0] * 300 + [1] * 300 + [2] * 300 + [3] * 20)
    noise = rng.normal(0, 0.01, (len(regimes), 2))
    drift = np.stack(
        [
            regimes.map({0: 0.002, 1: -0.001, 2: -0.002, 3: 0.003}),
            regimes.map({0: -0.001, 1: 0.002, 2: -0.002, 3: 0.0}),
        ],
        axis=1,
    )
    returns = pd.DataFrame(drift + noise, columns=["trend", "reversion"])
    returns["buy_and_hold"] = 0.0
    mapping = select_strategy_per_regime(returns, regimes, min_days=60)
    assert mapping == {0: "trend", 1: "reversion", 2: CASH, 3: "buy_and_hold"}


def test_weights_follow_the_mapped_strategy() -> None:
    """Each day takes its weight from the strategy mapped to that day's regime."""
    index = pd.date_range("2024-01-01", periods=5)
    weights = pd.DataFrame({"buy_and_hold": 1.0, "trend": [0.0, 1.0, 1.0, 0.0, 1.0]}, index=index)
    regimes = pd.Series([0, 0, 1, 1], index=index[1:])  # first day has no label yet
    combined = regime_switching_weights(weights, regimes, {0: "trend", 1: CASH})
    assert combined.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
