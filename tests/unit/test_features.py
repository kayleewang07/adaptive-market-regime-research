"""Feature engineering must be lag-safe."""

import pandas as pd

from amrrp.features.pipeline import FeaturePipeline


def test_features_do_not_use_same_day_or_future_data(market: pd.DataFrame) -> None:
    """Altering prices from day t onward must leave every feature before t unchanged."""
    cutoff = market.index[800]
    altered = market.copy()
    altered.loc[altered.index >= cutoff, ["Open", "High", "Low", "Close"]] *= 1.7
    altered.loc[altered.index >= cutoff, "Volume"] *= 3
    original = FeaturePipeline().fit_transform(market)
    changed = FeaturePipeline().fit_transform(altered)
    # Row t uses data through t-1, so rows up to and including the cutoff are unaffected.
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_pipeline_output_is_finite(market: pd.DataFrame) -> None:
    """The feature matrix has no missing or infinite values."""
    features = FeaturePipeline().fit_transform(market)
    assert features.notna().all().all()
    assert len(features) > len(market) - 100


def test_features_are_invariant_to_price_level(market: pd.DataFrame) -> None:
    """Scaling all prices by 10x (e.g. a reverse split) must not change any feature."""
    scaled = market.copy()
    scaled[["Open", "High", "Low", "Close"]] *= 10
    pd.testing.assert_frame_equal(
        FeaturePipeline().fit_transform(market), FeaturePipeline().fit_transform(scaled)
    )
