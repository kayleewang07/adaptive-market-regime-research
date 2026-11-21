"""End-to-end experiment execution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amrrp.analytics.bootstrap import sharpe_inference
from amrrp.analytics.performance import compare_strategies
from amrrp.analytics.regime_comparison import compare_by_regime, regime_summary
from amrrp.backtesting.engine import BacktestEngine
from amrrp.core.types import BacktestConfig, ExperimentResult
from amrrp.data.validation import validate_market_frame
from amrrp.explainability.feature_importance import feature_importance
from amrrp.explainability.transitions import transition_matrix
from amrrp.features.pipeline import FeaturePipeline
from amrrp.regimes.labeling import order_regimes
from amrrp.reporting.report_builder import build_markdown_report
from amrrp.strategies.regime_switching import (
    regime_switching_weights,
    select_strategy_per_regime,
)

from .registry import STATE_MODELS, STRATEGIES, build_regime_model
from .specification import ExperimentSpec
from .tracking import create_run_directory, write_manifest

MIN_TRAINING_DAYS = 250
REGIME_SWITCHING = "regime_switching"


class ExperimentRunner:
    """Detect regimes and evaluate strategies with a strict train/test split.

    1. Build lagged features from the full price history (each row only uses data
       available before that day).
    2. Fit the regime model on rows before ``train_end`` only.
    3. Label every day with that frozen model. All four models label causally.
    4. Backtest every strategy over the full history and report metrics separately
       for the training and out-of-sample (test) periods.
    5. For state models, build a regime-switching strategy that runs, in each regime,
       the strategy with the best training-period Sharpe, and test it out of sample.
    6. Attach bootstrap confidence intervals and tests vs buy-and-hold to the
       out-of-sample Sharpe ratios.
    """

    def __init__(self, artifact_root: Path = Path("artifacts")) -> None:
        """Store the output root for research runs."""
        self.artifact_root = artifact_root

    def run(self, market_data: pd.DataFrame, specification: ExperimentSpec) -> ExperimentResult:
        """Execute the experiment, write artifacts and return the out-of-sample metrics."""
        validate_market_frame(market_data)
        train_end = pd.Timestamp(specification.train_end)
        features = FeaturePipeline().fit_transform(market_data)
        train_features = features[features.index < train_end]
        if len(train_features) < MIN_TRAINING_DAYS:
            raise ValueError(
                f"Only {len(train_features)} training rows before {specification.train_end}; "
                f"need at least {MIN_TRAINING_DAYS}. Move train_end later or start earlier."
            )
        if not (features.index >= train_end).any():
            raise ValueError(f"No data on or after train_end={specification.train_end}.")

        model = build_regime_model(
            specification.regime_model, specification.n_regimes, specification.seed
        )
        model.fit(train_features)
        regimes = model.predict(features)
        if specification.regime_model in STATE_MODELS and "rolling_volatility" in features:
            # Renumber so regime 0 is the calmest state in the training window.
            train_regimes = regimes[regimes.index < train_end]
            mapping = order_regimes(
                train_regimes, features.loc[train_regimes.index, "rolling_volatility"]
            )
            regimes = regimes.map(mapping).astype(int).rename("regime")

        config = BacktestConfig(
            transaction_cost_bps=specification.transaction_cost_bps,
            slippage_bps=specification.slippage_bps,
        )
        engine = BacktestEngine(config)
        strategy_weights = pd.DataFrame(
            {
                name: STRATEGIES[name]().generate_signals(market_data)
                for name in specification.strategies
            }
        )
        strategy_returns = pd.DataFrame(
            {
                name: engine.run(market_data["Close"], strategy_weights[name]).returns
                for name in specification.strategies
            }
        )
        is_test = strategy_returns.index >= train_end
        test_regimes = regimes[regimes.index >= train_end]
        train_regimes = regimes[regimes.index < train_end]

        # Regime-aware allocation: choose a strategy per regime from training data only,
        # then follow that mapping out of sample (trading costs included).
        regime_map: dict[int, str] = {}
        if specification.regime_model in STATE_MODELS:
            default = (
                "buy_and_hold"
                if "buy_and_hold" in specification.strategies
                else specification.strategies[0]
            )
            regime_map = select_strategy_per_regime(
                strategy_returns[~is_test], train_regimes, default=default
            )
            switching = regime_switching_weights(strategy_weights, regimes, regime_map, default)
            strategy_returns[REGIME_SWITCHING] = engine.run(market_data["Close"], switching).returns
        train_returns, test_returns = strategy_returns[~is_test], strategy_returns[is_test]
        asset_returns = market_data["Close"].pct_change().rename("asset_return")

        metrics_test = compare_strategies(test_returns)
        metrics_train = compare_strategies(train_returns)
        regime_metrics_test = compare_by_regime(test_returns, test_regimes)
        regime_metrics_train = compare_by_regime(train_returns, train_regimes)
        summary_test = regime_summary(asset_returns, test_regimes)
        transitions = transition_matrix(test_regimes)
        inference = sharpe_inference(test_returns, seed=specification.seed)
        importance = feature_importance(
            train_features.loc[train_regimes.index], train_regimes, specification.seed
        )

        run_id, directory = create_run_directory(self.artifact_root, specification.regime_model)
        metadata = {
            "run_id": run_id,
            "train_rows": len(train_regimes),
            "test_rows": len(test_regimes),
            "features": list(features.columns),
            "regime_strategy_map": {str(k): v for k, v in regime_map.items()},
        }
        write_manifest(directory, specification, **metadata)
        metrics_test.to_csv(directory / "metrics_test.csv")
        metrics_train.to_csv(directory / "metrics_train.csv")
        regime_metrics_test.to_csv(directory / "regime_metrics_test.csv")
        regime_metrics_train.to_csv(directory / "regime_metrics_train.csv")
        summary_test.to_csv(directory / "regime_summary_test.csv")
        inference.to_csv(directory / "sharpe_inference_test.csv")
        transitions.to_csv(directory / "transition_matrix_test.csv")
        importance.rename("importance").to_csv(directory / "feature_importance.csv")
        strategy_returns.to_parquet(directory / "strategy_returns.parquet")
        regimes.to_frame().to_parquet(directory / "regimes.parquet")
        market_data[["Close"]].to_parquet(directory / "prices.parquet")
        build_markdown_report(
            specification,
            metrics_test=metrics_test,
            metrics_train=metrics_train,
            regime_summary=summary_test,
            regime_metrics=regime_metrics_test,
            inference=inference,
            regime_map=regime_map,
            importance=importance,
            output=directory / "report.md",
        )
        return ExperimentResult(run_id, directory, metrics_test, metadata)
