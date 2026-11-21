"""Stable, interpretable regime labels."""

from __future__ import annotations

import pandas as pd


def order_regimes(regimes: pd.Series, score: pd.Series) -> dict[int, int]:
    """Return a mapping that renumbers regimes by ascending mean ``score``.

    Clustering models number their states arbitrarily. Ordering them by, for example,
    mean realized volatility in the training window makes regime 0 the calmest state
    and the highest number the most turbulent, so labels mean the same thing across
    runs and models.
    """
    means = score.groupby(regimes).mean().sort_values()
    return {int(old): new for new, old in enumerate(means.index)}
