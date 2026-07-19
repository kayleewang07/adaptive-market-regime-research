"""End-to-end experiment execution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amrrp.analytics.performance import compare_strategies
from amrrp.analytics.regime_comparison import compare_by_regime
from amrrp.backtesting.engine import BacktestEngine
from amrrp.core.types import ExperimentResult
from amrrp.features.pipeline import FeaturePipeline
from amrrp.regimes.bayesian_changepoint import BayesianChangePointRegimeModel
from amrrp.regimes.gmm import GMMRegimeModel
from amrrp.regimes.hmm import HMMRegimeModel
from amrrp.regimes.kmeans import KMeansRegimeModel
from amrrp.strategies.buy_and_hold import BuyAndHold
from amrrp.strategies.ma_crossover import MovingAverageCrossover
from amrrp.strategies.mean_reversion import MeanReversion
from amrrp.strategies.momentum import Momentum
from amrrp.strategies.volatility_breakout import VolatilityBreakout

from .tracking import create_run_directory, write_manifest


class ExperimentRunner:
    """Run regime detection and a collection of strategies from one specification."""

    def __init__(self, artifact_root: Path = Path("artifacts")) -> None:
        """Store the output root for reproducible research runs."""
        self.artifact_root = artifact_root

    def run(self, market_data: pd.DataFrame, specification) -> ExperimentResult:
        """Execute all configured models/strategies and save metrics and returns."""
        features = FeaturePipeline().fit_transform(market_data)
        model_type = {
            "gmm": GMMRegimeModel,
            "hmm": HMMRegimeModel,
            "kmeans": KMeansRegimeModel,
            "bayesian_changepoint": BayesianChangePointRegimeModel,
        }[specification.regime_model]
        model = (
            model_type(n_regimes=specification.n_regimes, random_state=specification.seed)
            if specification.regime_model != "bayesian_changepoint"
            else model_type()
        )
        model.fit(features)
        regimes = model.predict(features)
        strategy_types = {
            "buy_and_hold": BuyAndHold,
            "moving_average_crossover": MovingAverageCrossover,
            "mean_reversion": MeanReversion,
            "volatility_breakout": VolatilityBreakout,
            "momentum": Momentum,
        }
        engine, returns = BacktestEngine(), {}
        for name in specification.strategies:
            strategy = strategy_types[name]()
            result = engine.run(market_data["Close"], strategy.generate_signals(market_data))
            returns[name] = result.returns
        strategy_returns = pd.DataFrame(returns)
        metrics = compare_strategies(strategy_returns)
        run_id, directory = create_run_directory(self.artifact_root)
        write_manifest(directory, specification)
        metrics.to_csv(directory / "metrics.csv")
        strategy_returns.to_parquet(directory / "strategy_returns.parquet")
        regimes.to_frame().to_parquet(directory / "regimes.parquet")
        compare_by_regime(strategy_returns, regimes).to_csv(directory / "regime_metrics.csv")
        return ExperimentResult(
            run_id, directory, metrics, {"regime_model": specification.regime_model}
        )
