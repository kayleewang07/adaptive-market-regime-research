"""Common market-regime model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from amrrp.core.interfaces import RegimeModel


class BaseRegimeModel(RegimeModel, ABC):
    """Base adapter that preserves feature columns and validates fitted state."""

    def _check_fitted(self) -> None:
        """Raise if prediction is requested before fitting."""
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("Fit the regime model before predicting.")

    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select fitted feature columns and reject missing observations."""
        self._check_fitted()
        return features.loc[:, self.feature_names_].dropna()

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> BaseRegimeModel:
        """Fit model on a feature matrix."""
