"""Smoke test for feature engineering, regime modeling, and portfolio simulation."""

import numpy as np
import pandas as pd

from amrrp.backtesting.engine import BacktestEngine
from amrrp.features.pipeline import FeaturePipeline
from amrrp.regimes.gmm import GMMRegimeModel
from amrrp.strategies.momentum import Momentum


def test_research_flow_with_constant_volume() -> None:
    """The workflow should remain usable when a feature column is all null."""
    index = pd.date_range("2023-01-01", periods=320, freq="B")
    close = pd.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0002, 0.01, len(index)))),
        index=index,
    )
    market = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    features = FeaturePipeline().fit_transform(market)
    regimes = GMMRegimeModel().fit(features).predict(features)
    result = BacktestEngine().run(market["Close"], Momentum(21).generate_signals(market))
    assert len(regimes) == len(features)
    assert len(result.equity) == len(market)
