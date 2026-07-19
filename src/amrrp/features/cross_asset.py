"""Cross-asset feature transformers."""

from __future__ import annotations

import pandas as pd

from .base import DataFrameTransformer


class CrossAssetFeatures(DataFrameTransformer):
    """Generate rolling correlations from a mapping of aligned close series."""

    def __init__(self, benchmark: str = "SPY", window: int = 63) -> None:
        """Store benchmark symbol and correlation window."""
        self.benchmark, self.window = benchmark, window

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return rolling correlations of each asset against the benchmark."""
        returns = X.pct_change()
        if self.benchmark not in returns:
            raise ValueError(f"Benchmark {self.benchmark!r} is absent from input columns.")
        return pd.DataFrame(
            {
                f"corr_{column}_{self.benchmark}": returns[column]
                .rolling(self.window)
                .corr(returns[self.benchmark])
                for column in returns
                if column != self.benchmark
            },
            index=X.index,
        ).shift(1)
