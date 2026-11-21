"""Name-to-class registries for configurable components."""

from __future__ import annotations

from amrrp.regimes.base import BaseRegimeModel
from amrrp.regimes.bayesian_changepoint import BayesianChangePointRegimeModel
from amrrp.regimes.gmm import GMMRegimeModel
from amrrp.regimes.hmm import HMMRegimeModel
from amrrp.regimes.kmeans import KMeansRegimeModel
from amrrp.strategies.base import Strategy
from amrrp.strategies.buy_and_hold import BuyAndHold
from amrrp.strategies.ma_crossover import MovingAverageCrossover
from amrrp.strategies.mean_reversion import MeanReversion
from amrrp.strategies.momentum import Momentum
from amrrp.strategies.volatility_breakout import VolatilityBreakout

STATE_MODELS: dict[str, type[GMMRegimeModel | HMMRegimeModel | KMeansRegimeModel]] = {
    "gmm": GMMRegimeModel,
    "hmm": HMMRegimeModel,
    "kmeans": KMeansRegimeModel,
}
REGIME_MODELS: tuple[str, ...] = (*STATE_MODELS, "bayesian_changepoint")

STRATEGIES: dict[str, type[Strategy]] = {
    "buy_and_hold": BuyAndHold,
    "moving_average_crossover": MovingAverageCrossover,
    "mean_reversion": MeanReversion,
    "volatility_breakout": VolatilityBreakout,
    "momentum": Momentum,
}


def build_regime_model(name: str, n_regimes: int, seed: int) -> BaseRegimeModel:
    """Instantiate a regime model by registry name."""
    if name == "bayesian_changepoint":
        return BayesianChangePointRegimeModel()
    if name not in STATE_MODELS:
        raise ValueError(f"Unknown regime model {name!r}; choose from {REGIME_MODELS}.")
    return STATE_MODELS[name](n_regimes=n_regimes, random_state=seed)
