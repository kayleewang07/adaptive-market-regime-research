"""Gaussian Mixture Model regime adapter."""

from __future__ import annotations

import pandas as pd

from .base import BaseRegimeModel
from .visualization import regime_figure


class GMMRegimeModel(BaseRegimeModel):
    """Gaussian mixture regime classifier."""

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        """Store mixture configuration."""
        self.n_regimes, self.random_state = n_regimes, random_state

    def fit(self, features: pd.DataFrame) -> GMMRegimeModel:
        """Fit a scaled Gaussian mixture model."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        self.feature_names_ = list(features.columns)
        self.scaler_ = StandardScaler().fit(features)
        self.model_ = GaussianMixture(self.n_regimes, random_state=self.random_state).fit(
            self.scaler_.transform(features)
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return most likely regime per observation."""
        values = self._prepare(features)
        return pd.Series(
            self.model_.predict(self.scaler_.transform(values)), index=values.index, name="regime"
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return posterior mixture responsibilities."""
        values = self._prepare(features)
        return pd.DataFrame(
            self.model_.predict_proba(self.scaler_.transform(values)),
            index=values.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )

    def visualize(self, features: pd.DataFrame):
        """Return a regime timeline figure."""
        return regime_figure(self.predict(features))
