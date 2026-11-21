"""Which features drive the regime assignments."""

import pandas as pd


def feature_importance(features: pd.DataFrame, labels: pd.Series, seed: int = 42) -> pd.Series:
    """Fit a surrogate random forest to the regime labels and return its importances.

    The forest learns to reproduce the regime model's labels from the features; its
    impurity-based importances show which inputs separate the regimes most.
    """
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)
    forest.fit(features, labels)
    return pd.Series(forest.feature_importances_, index=features.columns).sort_values(
        ascending=False
    )
