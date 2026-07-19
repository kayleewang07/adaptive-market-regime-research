"""SHAP-based regime-model explanations."""

import pandas as pd


def shap_values(model, features: pd.DataFrame):
    """Return SHAP explanations for a model supporting the SHAP explainer API."""
    import shap

    return shap.Explainer(model, features)(features)
