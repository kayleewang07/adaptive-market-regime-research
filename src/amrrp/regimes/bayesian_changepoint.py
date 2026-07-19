"""Bayesian online change-point regime adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRegimeModel
from .visualization import regime_figure


class BayesianChangePointRegimeModel(BaseRegimeModel):
    """Pragmatic Bayesian change-point detector using posterior break odds.

    The model computes a conjugate-normal predictive surprise score and turns
    high-posterior change points into sequential segment labels.
    """

    def __init__(self, hazard: float = 1 / 126, threshold: float = 0.8) -> None:
        """Store expected break hazard and posterior threshold."""
        self.hazard, self.threshold = hazard, threshold

    def fit(self, features: pd.DataFrame) -> BayesianChangePointRegimeModel:
        """Estimate a standardized one-dimensional surprise process."""
        from sklearn.preprocessing import StandardScaler

        self.feature_names_ = list(features.columns)
        self.scaler_ = StandardScaler().fit(features)
        values = self.scaler_.transform(features)
        self.mean_, self.std_ = values.mean(axis=0), values.std(axis=0).clip(1e-8)
        return self

    def _probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return posterior probabilities of continuation and a new segment."""
        values = self._prepare(features)
        standardized = self.scaler_.transform(values)
        surprise = np.sqrt(((standardized - self.mean_) / self.std_).mean(axis=1) ** 2)
        change = 1 / (1 + np.exp(-(surprise - 1.5) * 2))
        change = self.hazard + (1 - self.hazard) * change
        return pd.DataFrame(
            {"continuation": 1 - change, "change_point": change}, index=values.index
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return posterior continuation and change-point probabilities."""
        return self._probabilities(features)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return incrementing segment labels after probable change points."""
        probability = self._probabilities(features)["change_point"]
        return (probability >= self.threshold).cumsum().rename("regime")

    def visualize(self, features: pd.DataFrame):
        """Return a timeline of Bayesian change-point segments."""
        return regime_figure(self.predict(features))
