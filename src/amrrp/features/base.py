"""Base contracts for sklearn-compatible feature transformers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Base class for stateless pandas-to-pandas feature transformers."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> DataFrameTransformer:  # noqa: N803
        """Learn no state and return this transformer."""
        return self
