"""Principal-component visualizations."""

import pandas as pd


def pca_projection(features: pd.DataFrame, components: int = 2) -> pd.DataFrame:
    """Return principal-component coordinates for a feature matrix."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    values = StandardScaler().fit_transform(features.dropna())
    return pd.DataFrame(
        PCA(components).fit_transform(values),
        index=features.dropna().index,
        columns=[f"PC{i + 1}" for i in range(components)],
    )
