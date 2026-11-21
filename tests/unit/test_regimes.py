"""Regime models: recovery of known regimes and absence of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from amrrp.explainability.transitions import transition_matrix
from amrrp.regimes.bayesian_changepoint import BayesianChangePointRegimeModel
from amrrp.regimes.gmm import GMMRegimeModel
from amrrp.regimes.hmm import HMMRegimeModel
from amrrp.regimes.kmeans import KMeansRegimeModel
from amrrp.regimes.labeling import order_regimes


def two_regime_features(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Volatility-like features from alternating calm (0) and turbulent (1) blocks."""
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1, 0, 1, 0], 200)
    vol = np.where(truth == 1, 0.03, 0.01)
    features = pd.DataFrame(
        {
            "rolling_volatility": vol + rng.normal(0, 0.002, len(truth)),
            "abs_return": np.abs(rng.normal(0, vol)),
        },
        index=pd.bdate_range("2010-01-01", periods=len(truth)),
    )
    return features, truth


@pytest.mark.parametrize("model_type", [GMMRegimeModel, KMeansRegimeModel, HMMRegimeModel])
def test_state_models_recover_known_regimes(model_type: type) -> None:
    """After ordering by volatility, labels should match the true regimes."""
    features, truth = two_regime_features()
    model = model_type(n_regimes=2, random_state=0).fit(features)
    labels = model.predict(features)
    labels = labels.map(order_regimes(labels, features["rolling_volatility"]))
    assert (labels.to_numpy() == truth).mean() > 0.95
    probabilities = model.predict_proba(features)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_hmm_labels_do_not_depend_on_future_observations() -> None:
    """Filtered HMM labels before a cutoff ignore everything after it."""
    features, _ = two_regime_features()
    model = HMMRegimeModel(n_regimes=2, random_state=0).fit(features)
    altered = features.copy()
    altered.iloc[500:] *= 3
    before = model.predict_proba(features).iloc[:500]
    after = model.predict_proba(altered).iloc[:500]
    pd.testing.assert_frame_equal(before, after)


def test_changepoint_detects_a_volatility_shift() -> None:
    """A tripling of volatility should be detected within a few weeks, with few false alarms."""
    rng = np.random.default_rng(0)
    returns = np.r_[rng.normal(0, 0.01, 600), rng.normal(0, 0.03, 300), rng.normal(0, 0.01, 600)]
    features = pd.DataFrame(
        {"return_1d": returns}, index=pd.bdate_range("2000-01-03", periods=1500)
    )
    model = BayesianChangePointRegimeModel().fit(features.iloc[:600])
    segments = model.predict(features)
    changes = np.flatnonzero(np.diff(segments.to_numpy())) + 1
    assert any(600 <= c <= 630 for c in changes)
    assert any(900 <= c <= 950 for c in changes)
    assert len(changes) <= 4
    assert model.predict_proba(features)["change_point"].between(0, 1).all()


def test_predict_before_fit_raises() -> None:
    """Unfitted models refuse to predict."""
    with pytest.raises(RuntimeError):
        GMMRegimeModel().predict(pd.DataFrame({"x": [1.0]}))


def test_transition_matrix_rows_sum_to_one() -> None:
    """Each row is a probability distribution over the next regime."""
    matrix = transition_matrix(pd.Series([0, 0, 1, 1, 0, 2, 2, 0]))
    assert np.allclose(matrix.sum(axis=1), 1)
    assert matrix.loc[0, 0] == pytest.approx(1 / 3)
