"""Tests for market-data cleanup, validation and caching."""

import pandas as pd
import pytest

from amrrp.data.cache import MarketDataCache
from amrrp.data.cleaning import clean_market_frame
from amrrp.data.repository import MarketDataRepository
from amrrp.data.validation import validate_market_frame


def valid_frame() -> pd.DataFrame:
    """Two valid daily bars."""
    return pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 20],
        },
        index=pd.to_datetime(["2024-01-02 14:30+00:00", "2024-01-03 14:30+00:00"]),
    )


def test_clean_and_validate_market_frame() -> None:
    """Cleanup should normalize dates and preserve a valid OHLCV frame."""
    cleaned = clean_market_frame(valid_frame())
    validate_market_frame(cleaned)
    assert cleaned.index.tz is None
    assert list(cleaned.index.strftime("%H:%M")) == ["00:00", "00:00"]


def test_validate_rejects_missing_values() -> None:
    """Validation should refuse incomplete OHLCV observations."""
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [float("nan")], "Volume": [1]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    with pytest.raises(ValueError, match="missing"):
        validate_market_frame(frame)


def test_validate_rejects_inconsistent_high_low() -> None:
    """A high below the close is impossible."""
    frame = clean_market_frame(valid_frame())
    frame.loc[frame.index[0], "High"] = 1.0
    with pytest.raises(ValueError, match="High"):
        validate_market_frame(frame)


def test_cleaning_snaps_floating_point_noise_but_not_real_errors() -> None:
    """A high 1e-16 below the close is rounding noise; a high 10% below is bad data."""
    frame = clean_market_frame(valid_frame())
    noisy = frame.copy()
    noisy.loc[noisy.index[0], "High"] = noisy.loc[noisy.index[0], "Close"] * (1 - 1e-16) - 1e-16
    validate_market_frame(clean_market_frame(noisy))
    broken = frame.copy()
    broken.loc[broken.index[0], "High"] = broken.loc[broken.index[0], "Close"] * 0.9
    with pytest.raises(ValueError, match="High"):
        validate_market_frame(clean_market_frame(broken))


def test_empty_download_fails_with_a_clear_message() -> None:
    """An empty frame is rejected before it reaches any model."""
    with pytest.raises(ValueError, match="empty"):
        clean_market_frame(pd.DataFrame())


class CountingProvider:
    """Fake provider that records how often it is called."""

    def __init__(self) -> None:
        self.calls = 0

    def download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        self.calls += 1
        return valid_frame()


def test_repository_caches_downloads(tmp_path) -> None:
    """The second request is served from the parquet cache."""
    provider = CountingProvider()
    repository = MarketDataRepository(provider, MarketDataCache(tmp_path))
    first = repository.get("spy", "2024-01-01", "2024-02-01")
    second = repository.get("spy", "2024-01-01", "2024-02-01")
    assert provider.calls == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    repository.get("spy", "2024-01-01", "2024-02-01", refresh=True)
    assert provider.calls == 2
