"""Shared synthetic market data for tests (no network access needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_market(
    start: str = "2005-01-03", end: str = "2012-12-31", seed: int = 1, turbulent_years=(2008, 2011)
) -> pd.DataFrame:
    """Return a valid OHLCV frame with calm years (1% daily vol) and turbulent ones (3%)."""
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    vol = np.where(index.year.isin(turbulent_years), 0.03, 0.01)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, vol))), index=index)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, len(index)).astype(float),
        }
    )


@pytest.fixture
def market() -> pd.DataFrame:
    """Eight years of synthetic daily prices with two turbulent years."""
    return make_market()
