"""Base contracts for sklearn-compatible feature transformers."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Base class for parameterized pandas-to-pandas feature transformers."""

    def fit(self, X, y=None):  # noqa: N803
        """Learn no state by default and return this transformer."""
        return self
