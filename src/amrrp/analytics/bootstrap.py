"""Bootstrap inference for Sharpe ratios.

Daily returns are autocorrelated in volatility, so resampling single days would
understate uncertainty. The stationary bootstrap (Politis & Romano, 1994) resamples
blocks of random length (mean ``block_length`` days) and keeps short-range dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(
    length: int, n_resamples: int, block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Return an ``(n_resamples, length)`` array of stationary-bootstrap row indices."""
    if length < 2:
        raise ValueError("Need at least two observations to bootstrap.")
    restart = rng.random((n_resamples, length)) < 1 / block_length
    starts = rng.integers(0, length, size=(n_resamples, length))
    indices = np.empty((n_resamples, length), dtype=np.int64)
    indices[:, 0] = starts[:, 0]
    for t in range(1, length):
        indices[:, t] = np.where(restart[:, t], starts[:, t], (indices[:, t - 1] + 1) % length)
    return indices


def _sharpe(samples: np.ndarray, periods_per_year: int) -> np.ndarray:
    """Annualized Sharpe ratio of each row (zero risk-free rate)."""
    std = samples.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, samples.mean(axis=-1) / std * np.sqrt(periods_per_year), np.nan)


def sharpe_inference(
    returns: pd.DataFrame,
    benchmark: str = "buy_and_hold",
    n_resamples: int = 2000,
    block_length: float = 20.0,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """Return Sharpe ratios with bootstrap confidence intervals and tests vs ``benchmark``.

    All strategies are resampled with the *same* indices, so the Sharpe difference to the
    benchmark is a paired comparison. ``p_value`` is a two-sided bootstrap p-value for
    "this strategy's Sharpe equals the benchmark's".
    """
    clean = returns.dropna()
    values = clean.to_numpy().T  # (strategies, days)
    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(values.shape[1], n_resamples, block_length, rng)
    boot = _sharpe(values[:, indices], periods_per_year)  # (strategies, resamples)
    point = _sharpe(values, periods_per_year)
    alpha = (1 - confidence) / 2
    columns = list(clean.columns)
    b = columns.index(benchmark) if benchmark in columns else None
    rows = []
    for i in range(len(columns)):
        row = {
            "sharpe": point[i],
            "ci_low": np.nanquantile(boot[i], alpha),
            "ci_high": np.nanquantile(boot[i], 1 - alpha),
        }
        if b is not None and i != b:
            diff = boot[i] - boot[b]
            row["diff_vs_benchmark"] = point[i] - point[b]
            row["diff_ci_low"] = np.nanquantile(diff, alpha)
            row["diff_ci_high"] = np.nanquantile(diff, 1 - alpha)
            tail = min(np.nanmean(diff <= 0), np.nanmean(diff >= 0))
            row["p_value"] = min(1.0, 2 * tail)
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(columns, name="strategy"))
