"""Feature-importance analysis."""

import pandas as pd


def feature_importance(model, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Fit a surrogate forest and return impurity-based feature importance."""
    from sklearn.ensemble import RandomForestClassifier

    fitted = RandomForestClassifier(n_estimators=300, random_state=42).fit(features, labels)
    return pd.Series(fitted.feature_importances_, index=features.columns).sort_values(
        ascending=False
    )
