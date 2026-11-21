"""Performance metrics checked against hand-calculated values."""

import math

import pandas as pd
import pytest

from amrrp.backtesting.metrics import performance_metrics


def test_metrics_match_hand_calculation() -> None:
    """Three periods with one period per 'year' make every metric easy to verify."""
    returns = pd.Series([0.10, -0.05, 0.02])
    m = performance_metrics(returns, periods_per_year=3)
    # Equity: 1.10 -> 1.045 -> 1.0659; one year elapses.
    assert m["cagr"] == pytest.approx(0.0659)
    assert m["annual_return"] == pytest.approx(0.07)  # mean 0.02333 * 3
    sample_std = math.sqrt(
        ((0.10 - 0.07 / 3) ** 2 + (-0.05 - 0.07 / 3) ** 2 + (0.02 - 0.07 / 3) ** 2) / 2
    )
    assert m["volatility"] == pytest.approx(sample_std * math.sqrt(3))
    assert m["sharpe"] == pytest.approx(0.07 / (sample_std * math.sqrt(3)))
    assert m["max_drawdown"] == pytest.approx(1.045 / 1.10 - 1)  # -5%
    downside = math.sqrt((0.05**2 / 3) * 3)
    assert m["sortino"] == pytest.approx(0.07 / downside)
    assert m["calmar"] == pytest.approx(0.0659 / 0.05)


def test_metrics_without_losses_have_no_drawdown_ratios() -> None:
    """Calmar and Sortino are undefined (NaN) when there are no losses."""
    m = performance_metrics(pd.Series([0.01, 0.02, 0.01]))
    assert m["max_drawdown"] == 0
    assert math.isnan(m["calmar"]) and math.isnan(m["sortino"])


def test_metrics_of_too_short_series_are_nan() -> None:
    """Fewer than two observations cannot produce a volatility."""
    assert all(math.isnan(v) for v in performance_metrics(pd.Series([0.01])).values())
