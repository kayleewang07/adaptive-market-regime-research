"""K-Means regime adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseRegimeModel
from .visualization import regime_figure


class KMeansRegimeModel(BaseRegimeModel):
    """Centroid-based regime classifier with distance-derived affinities.

    Each day is assigned to its nearest centroid independently, so labels are causal.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        """Store clustering configuration."""
        self.n_regimes, self.random_state = n_regimes, random_state

    def fit(self, features: pd.DataFrame) -> KMeansRegimeModel:
        """Fit standardized K-Means clusters."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        self.feature_names_ = list(features.columns)
        self.scaler_ = StandardScaler().fit(features)
        self.model_ = KMeans(self.n_regimes, n_init=10, random_state=self.random_state).fit(
            self.scaler_.transform(features)
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return nearest-centroid regime labels."""
        values = self._prepare(features)
        return pd.Series(
            self.model_.predict(self.scaler_.transform(values)), index=values.index, name="regime"
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return a softmax over negative centroid distances (not a calibrated probability)."""
        values = self._prepare(features)
        distance = self.model_.transform(self.scaler_.transform(values))
        affinity = np.exp(-(distance - distance.min(axis=1, keepdims=True)))
        return pd.DataFrame(
            affinity / affinity.sum(axis=1, keepdims=True),
            index=values.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )

    def visualize(self, features: pd.DataFrame) -> Any:
        """Return a regime timeline figure."""
        return regime_figure(self.predict(features))
