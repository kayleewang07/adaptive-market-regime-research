"""Experiment definitions and validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentSpec:
    """Serializable, reproducible configuration for a market research run."""

    symbols: tuple[str, ...] = ("SPY", "QQQ", "TLT", "GLD", "^VIX")
    start: str = "2005-01-01"
    end: str = "2025-01-01"
    regime_model: str = "gmm"
    n_regimes: int = 3
    strategies: tuple[str, ...] = (
        "buy_and_hold",
        "moving_average_crossover",
        "mean_reversion",
        "volatility_breakout",
        "momentum",
    )
    seed: int = 42
    feature_params: dict[str, object] = field(default_factory=dict)
