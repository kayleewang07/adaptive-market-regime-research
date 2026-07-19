"""Feature pipeline assembly."""

from __future__ import annotations

import pandas as pd

from .statistical import StatisticalFeatures
from .technical import TechnicalFeatures


class FeaturePipeline:
    """Composable sklearn-compatible feature engineering pipeline."""

    def __init__(
        self,
        technical: TechnicalFeatures | None = None,
        statistical: StatisticalFeatures | None = None,
    ) -> None:
        """Configure default or supplied feature transformers."""
        self.technical = technical or TechnicalFeatures()
        self.statistical = statistical or StatisticalFeatures()

    def fit_transform(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create a clean feature matrix for regime model training."""
        result = pd.concat(
            [
                self.technical.fit_transform(market_data),
                self.statistical.fit_transform(market_data),
            ],
            axis=1,
        )
        result = result.replace([float("inf"), float("-inf")], float("nan"))
        return result.dropna(axis=1, how="all").dropna()

    def transform(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create an evaluation feature matrix using configured transformers."""
        return self.fit_transform(market_data)
