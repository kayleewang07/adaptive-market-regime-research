"""Regime-aware allocation: run a different strategy in each market regime."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from amrrp.analytics.regime_comparison import compare_by_regime

CASH = "cash"


def select_strategy_per_regime(
    train_returns: pd.DataFrame,
    train_regimes: pd.Series,
    min_days: int = 60,
    default: str = "buy_and_hold",
) -> dict[int, str]:
    """Pick, for each regime, the strategy with the best *training-period* Sharpe ratio.

    Holding cash (Sharpe 0) is always a candidate, so a regime where every strategy lost
    money in training is sat out. Regimes seen on fewer than ``min_days`` training days
    fall back to ``default``, because their Sharpe estimates are too noisy to act on.
    Only training data is used, so the choice can be evaluated honestly out of sample.
    """
    table = compare_by_regime(train_returns, train_regimes)
    mapping: dict[int, str] = {}
    for regime in table.index.get_level_values("regime").unique():
        group = table.xs(regime, level="regime")
        days = int(group["days"].iloc[0])
        sharpe = group["sharpe"].fillna(float("-inf"))
        best = str(sharpe.idxmax())
        key = int(regime)
        if days < min_days:
            mapping[key] = default
        elif sharpe[best] <= 0:
            mapping[key] = CASH
        else:
            mapping[key] = best
    return mapping


def regime_switching_weights(
    strategy_weights: pd.DataFrame,
    regimes: pd.Series,
    mapping: Mapping[int, str],
    default: str = "buy_and_hold",
) -> pd.Series:
    """Combine per-strategy target weights by following the strategy mapped to each regime.

    ``regimes`` at day ``t`` is computed from features lagged by one day, so it is known
    before the close of ``t`` and can be used for the decision made at that close. Days
    without a regime label (feature warm-up) or with an unmapped regime use ``default``.
    """
    labels = regimes.reindex(strategy_weights.index)
    choice = labels.map(lambda r: mapping.get(int(r), default) if pd.notna(r) else default)
    weights = pd.Series(0.0, index=strategy_weights.index, name="target_weight")
    for name in choice.unique():
        if name == CASH:
            continue
        selected = choice == name
        weights[selected] = strategy_weights.loc[selected, name]
    return weights
