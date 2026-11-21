"""Bootstrap inference for Sharpe ratios."""

import numpy as np
import pandas as pd
import pytest

from amrrp.analytics.bootstrap import sharpe_inference, stationary_bootstrap_indices


def test_indices_are_valid_and_form_blocks() -> None:
    """Indices stay in range, and consecutive steps mostly continue the current block."""
    rng = np.random.default_rng(0)
    indices = stationary_bootstrap_indices(500, 200, block_length=20, rng=rng)
    assert indices.shape == (200, 500)
    assert indices.min() >= 0 and indices.max() < 500
    continues = (np.diff(indices, axis=1) % 500) == 1
    assert continues.mean() == pytest.approx(1 - 1 / 20, abs=0.01)


def make_returns(edge: float, seed: int = 0) -> pd.DataFrame:
    """Benchmark returns plus a strategy that earns ``edge`` extra per day."""
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0004, 0.01, 2000)
    strategy = benchmark + edge + rng.normal(0, 0.001, 2000)
    return pd.DataFrame({"buy_and_hold": benchmark, "better": strategy})


def test_identical_strategies_are_not_different() -> None:
    """A strategy equal to the benchmark has zero difference and p-value 1."""
    returns = make_returns(0.0)
    returns["better"] = returns["buy_and_hold"]
    row = sharpe_inference(returns, n_resamples=300).loc["better"]
    assert row["diff_vs_benchmark"] == pytest.approx(0)
    assert row["p_value"] == pytest.approx(1.0)


def test_clear_edge_is_significant_and_intervals_bracket_the_estimate() -> None:
    """A large, steady edge yields a small p-value and an interval above zero."""
    result = sharpe_inference(make_returns(0.0005), n_resamples=500)
    row = result.loc["better"]
    assert row["p_value"] < 0.01
    assert row["diff_ci_low"] > 0
    assert (result["ci_low"] < result["sharpe"]).all()
    assert (result["sharpe"] < result["ci_high"]).all()


def test_results_are_reproducible_with_a_seed() -> None:
    """The same seed gives identical intervals."""
    returns = make_returns(0.0002)
    pd.testing.assert_frame_equal(
        sharpe_inference(returns, n_resamples=200, seed=3),
        sharpe_inference(returns, n_resamples=200, seed=3),
    )
