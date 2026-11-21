"""End-to-end experiment, configuration and dashboard tests on synthetic data."""

import json
from pathlib import Path

import pandas as pd
import pytest

from amrrp.experiments.registry import REGIME_MODELS
from amrrp.experiments.runner import ExperimentRunner
from amrrp.experiments.specification import ExperimentSpec

ROOT = Path(__file__).resolve().parents[2]
SPEC = dict(start="2005-01-01", train_end="2009-01-01", end="2013-01-01")


@pytest.mark.parametrize("model", REGIME_MODELS)
def test_experiment_writes_out_of_sample_artifacts(model: str, market, tmp_path) -> None:
    """Every model runs end to end and writes the documented artifacts."""
    spec = ExperimentSpec(regime_model=model, **SPEC)
    result = ExperimentRunner(tmp_path).run(market, spec)
    files = {p.name for p in result.artifact_dir.iterdir()}
    assert {
        "manifest.json",
        "metrics_test.csv",
        "metrics_train.csv",
        "regime_metrics_test.csv",
        "regime_summary_test.csv",
        "transition_matrix_test.csv",
        "feature_importance.csv",
        "strategy_returns.parquet",
        "regimes.parquet",
        "prices.parquet",
        "report.md",
    } <= files
    expected = list(spec.strategies)
    if model != "bayesian_changepoint":  # change-point segments don't recur, so no switching
        expected.append("regime_switching")
    assert list(result.metrics.index) == expected
    assert "sharpe_inference_test.csv" in files
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text())
    assert manifest["specification"]["regime_model"] == model
    assert manifest["test_rows"] > 0 and manifest["train_rows"] >= 250


def test_out_of_sample_regimes_come_from_a_model_fitted_on_training_data(market, tmp_path) -> None:
    """Changing test-period prices must not change any training-period label."""
    spec = ExperimentSpec(regime_model="gmm", **SPEC)
    altered = market.copy()
    test = altered.index >= pd.Timestamp(spec.train_end)
    altered.loc[test, ["Open", "High", "Low", "Close"]] *= 2
    first = ExperimentRunner(tmp_path / "a").run(market, spec)
    second = ExperimentRunner(tmp_path / "b").run(altered, spec)
    labels_a = pd.read_parquet(first.artifact_dir / "regimes.parquet")["regime"]
    labels_b = pd.read_parquet(second.artifact_dir / "regimes.parquet")["regime"]
    before = labels_a.index < pd.Timestamp(spec.train_end)
    pd.testing.assert_series_equal(labels_a[before], labels_b[before])


def test_regime_zero_is_the_calmest(market, tmp_path) -> None:
    """State models are renumbered so volatility rises with the regime number."""
    result = ExperimentRunner(tmp_path).run(market, ExperimentSpec(regime_model="hmm", **SPEC))
    summary = pd.read_csv(result.artifact_dir / "regime_summary_test.csv", index_col=0)
    assert summary["annualized_volatility"].idxmin() == 0


def test_too_little_training_data_is_rejected(market, tmp_path) -> None:
    """A train_end that leaves under a year of training data fails clearly."""
    spec = ExperimentSpec(start="2005-01-01", train_end="2005-06-01", end="2013-01-01")
    with pytest.raises(ValueError, match="training rows"):
        ExperimentRunner(tmp_path).run(market, spec)


def test_default_config_file_loads() -> None:
    """The shipped YAML matches the specification schema."""
    spec = ExperimentSpec.from_yaml(ROOT / "configs" / "default.yaml")
    assert spec.symbol == "SPY" and spec.train_end == "2015-01-01"


@pytest.mark.parametrize(
    "values",
    [
        {"regime_model": "random_forest"},
        {"strategies": ["lottery"]},
        {"train_end": "2030-01-01"},
        {"n_regimes": 1},
        {"typo_key": 1},
    ],
)
def test_invalid_configuration_is_rejected(values: dict) -> None:
    """Unknown names, impossible dates and unknown keys fail before any work is done."""
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict(values)


def test_dashboard_renders_every_page(market, tmp_path, monkeypatch) -> None:
    """Each dashboard page renders without raising for a real run."""
    from streamlit.testing.v1 import AppTest

    ExperimentRunner(tmp_path).run(market, ExperimentSpec(**SPEC))
    monkeypatch.setenv("AMRRP_ARTIFACTS", str(tmp_path))
    app = AppTest.from_file(
        str(ROOT / "src" / "amrrp" / "dashboard" / "app.py"), default_timeout=60
    )
    app.run()
    assert not app.exception
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_cross_asset_study_summarizes_each_symbol(tmp_path) -> None:
    """The study runs once per symbol and writes one summary row for each."""
    from amrrp.experiments.study import run_study
    from conftest import make_market

    def load(symbol: str, start: str, end: str) -> pd.DataFrame:
        return make_market(seed=sum(map(ord, symbol)))

    directory, table = run_study(["AAA", "BBB"], ExperimentSpec(**SPEC), load, tmp_path)
    assert list(table.index) == ["AAA", "BBB"]
    assert {"switching_sharpe", "p_value", "trend_sharpe_turbulent"} <= set(table.columns)
    assert (directory / "study_summary.md").exists()
    assert len([p for p in directory.iterdir() if p.is_dir()]) == 2
