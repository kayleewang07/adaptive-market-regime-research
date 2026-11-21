"""Hidden Markov Model regime adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .base import BaseRegimeModel
from .visualization import regime_figure


class HMMRegimeModel(BaseRegimeModel):
    """Gaussian hidden-Markov-model regime classifier.

    Prediction uses the *filtered* state probabilities ``P(state_t | x_1..x_t)`` from
    the forward algorithm. hmmlearn's ``predict`` (Viterbi) and ``predict_proba``
    (forward-backward) condition on the whole sequence, which would let future
    observations leak into each day's label.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        """Store HMM configuration."""
        self.n_regimes, self.random_state = n_regimes, random_state

    def fit(self, features: pd.DataFrame) -> HMMRegimeModel:
        """Fit a scaled Gaussian HMM with full covariance matrices."""
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        self.feature_names_ = list(features.columns)
        self.scaler_ = StandardScaler().fit(features)
        self.model_ = GaussianHMM(
            self.n_regimes, covariance_type="full", n_iter=300, random_state=self.random_state
        ).fit(self.scaler_.transform(features))
        return self

    def _filtered(self, features: pd.DataFrame) -> pd.DataFrame:
        """Run the forward algorithm and return normalized filtered probabilities."""
        values = self._prepare(features)
        x = self.scaler_.transform(values)
        log_emission = np.column_stack(
            [
                multivariate_normal.logpdf(
                    x, self.model_.means_[k], self.model_.covars_[k], allow_singular=True
                )
                for k in range(self.n_regimes)
            ]
        ).reshape(len(x), self.n_regimes)
        with np.errstate(divide="ignore"):
            log_start = np.log(self.model_.startprob_)
            log_trans = np.log(self.model_.transmat_)
        log_alpha = np.empty_like(log_emission)
        current = log_start + log_emission[0]
        log_alpha[0] = current - logsumexp(current)
        for t in range(1, len(x)):
            current = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_emission[t]
            log_alpha[t] = current - logsumexp(current)
        return pd.DataFrame(
            np.exp(log_alpha),
            index=values.index,
            columns=[f"regime_{i}" for i in range(self.n_regimes)],
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return the most probable state given information up to each day."""
        probabilities = self._filtered(features)
        return pd.Series(
            probabilities.to_numpy().argmax(axis=1), index=probabilities.index, name="regime"
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return filtered state probabilities."""
        return self._filtered(features)

    def visualize(self, features: pd.DataFrame) -> Any:
        """Return a regime timeline figure."""
        return regime_figure(self.predict(features))
