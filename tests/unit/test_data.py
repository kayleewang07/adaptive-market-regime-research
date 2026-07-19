"""Tests for market-data cleanup and validation."""

import pytest

pd = pytest.importorskip("pandas")

from amrrp.data.cleaning import clean_market_frame
from amrrp.data.validation import validate_market_frame


def test_clean_and_validate_market_frame() -> None:
    """Cleanup should normalize dates and preserve a valid OHLCV frame."""
    frame = pd.DataFrame(
        {
            "Open": [1, 2],
            "High": [2, 3],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [10, 20],
        },
        index=pd.to_datetime(["2024-01-02 14:30+00:00", "2024-01-03 14:30+00:00"]),
    )
    cleaned = clean_market_frame(frame)
    validate_market_frame(cleaned)
    assert cleaned.index.tz is None


def test_validate_rejects_missing_values() -> None:
    """Validation should refuse incomplete OHLCV observations."""
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [float("nan")], "Volume": [1]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    with pytest.raises(ValueError, match="missing"):
        validate_market_frame(frame)
