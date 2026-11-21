"""Bayesian online change-point detection (Adams & MacKay, 2007)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t

from .base import BaseRegimeModel
from .visualization import regime_figure


class BayesianChangePointRegimeModel(BaseRegimeModel):
    """Online change-point detector on a single return series.

    The model tracks the posterior over the *run length* (sessions since the last
    change) under a constant hazard rate. Within a segment, returns are Gaussian with
    unknown mean and variance and a Normal-Inverse-Gamma prior, giving a Student-t
    predictive distribution. Each update only uses data up to the current day.

    Unlike the clustering models, labels are *segment numbers* (0, 1, 2, ...) that
    increase at each detected change; segments are not recurring states.

    Args:
        hazard: prior probability of a change on any given day (1/250 = once a year).
        column: feature column to monitor (lagged daily return by default).
        window: a change is declared when the posterior probability that the current
            run is at most ``window`` sessions old crosses above ``threshold``.
        threshold: posterior probability required to declare a change.
        max_run_length: run-length posterior truncation, for speed.
    """

    def __init__(
        self,
        hazard: float = 1 / 250,
        column: str = "return_1d",
        window: int = 20,
        threshold: float = 0.9,
        max_run_length: int = 750,
    ) -> None:
        """Store the hazard rate, monitored column and detection settings."""
        if not 0 < hazard < 1:
            raise ValueError("hazard must lie strictly between 0 and 1.")
        self.hazard, self.column = hazard, column
        self.window, self.threshold = window, threshold
        self.max_run_length = max_run_length

    def fit(self, features: pd.DataFrame) -> BayesianChangePointRegimeModel:
        """Learn the standardization and prior scale from training returns."""
        if self.column not in features:
            raise ValueError(f"Column {self.column!r} is absent from the features.")
        self.feature_names_ = [self.column]
        series = features[self.column].dropna()
        self.location_ = float(series.mean())
        self.scale_ = float(series.std(ddof=1)) or 1.0
        return self

    def _run_length_posterior(self, features: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
        """Return the index and the (T, max_run_length + 1) run-length posterior."""
        values = self._prepare(features)[self.column]
        x = ((values - self.location_) / self.scale_).to_numpy()
        # Normal-Inverse-Gamma prior centred on the training distribution (unit variance).
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0
        size = self.max_run_length + 1
        posterior = np.zeros((len(x), size))
        log_r = np.array([0.0])
        mu, kappa = np.array([mu0]), np.array([kappa0])
        alpha, beta = np.array([alpha0]), np.array([beta0])
        log_h, log_1mh = np.log(self.hazard), np.log1p(-self.hazard)
        for t, xt in enumerate(x):
            scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
            log_pred = student_t.logpdf(xt, df=2 * alpha, loc=mu, scale=scale)
            growth = log_r + log_pred + log_1mh
            change = logsumexp(log_r + log_pred + log_h)
            log_r = np.append(change, growth)
            mu_new = (kappa * mu + xt) / (kappa + 1)
            beta_new = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
            mu, kappa = np.append(mu0, mu_new), np.append(kappa0, kappa + 1)
            alpha, beta = np.append(alpha0, alpha + 0.5), np.append(beta0, beta_new)
            if len(log_r) > size:
                log_r, mu, kappa = log_r[:size], mu[:size], kappa[:size]
                alpha, beta = alpha[:size], beta[:size]
            log_r = log_r - logsumexp(log_r)
            posterior[t, : len(log_r)] = np.exp(log_r)
        return values.index, posterior

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return the probability that a change happened within the last ``window`` days."""
        index, posterior = self._run_length_posterior(features)
        recent = posterior[:, : self.window + 1].sum(axis=1).clip(0.0, 1.0)
        return pd.DataFrame({"continuation": 1 - recent, "change_point": recent}, index=index)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return segment numbers that increment at each detected change point."""
        index, posterior = self._run_length_posterior(features)
        recent = posterior[:, : self.window + 1].sum(axis=1) > self.threshold
        onset = recent & ~np.r_[True, recent[:-1]]
        return pd.Series(np.cumsum(onset), index=index, name="regime")

    def visualize(self, features: pd.DataFrame) -> Any:
        """Return a timeline of change-point segments."""
        return regime_figure(self.predict(features))
