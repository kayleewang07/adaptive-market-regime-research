"""Abstract contracts for platform components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Produce daily target weights from a market frame.

    Timing convention: the weight at date ``t`` is the position chosen *at the close of
    ``t``* using information available up to and including ``t``. The backtest engine
    holds that position over the following session (``t`` to ``t + 1``), so strategies
    must not lag their own signals.
    """

    name: str

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return target weights indexed identically to ``data``."""


class RegimeModel(ABC):
    """Common model API for all regime estimators."""

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> RegimeModel:
        """Fit the estimator on feature observations."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return integer regime assignments."""

    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return regime (or change-point) probabilities per observation."""

    @abstractmethod
    def visualize(self, features: pd.DataFrame) -> Any:
        """Return a plotly figure visualizing inferred regimes."""
