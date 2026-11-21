"""Experiment definitions and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .registry import REGIME_MODELS, STRATEGIES


@dataclass(frozen=True)
class ExperimentSpec:
    """Serializable, validated configuration for one research run.

    Regime models are fitted only on data before ``train_end``; every result reported
    as out-of-sample covers ``train_end`` to ``end``.
    """

    symbol: str = "SPY"
    start: str = "2005-01-01"
    end: str = "2025-01-01"
    train_end: str = "2015-01-01"
    regime_model: str = "hmm"
    n_regimes: int = 3
    strategies: tuple[str, ...] = tuple(STRATEGIES)
    seed: int = 42
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        """Reject inconsistent or unknown settings early."""
        start, train_end, end = (pd.Timestamp(v) for v in (self.start, self.train_end, self.end))
        if not start < train_end < end:
            raise ValueError("Dates must satisfy start < train_end < end.")
        if self.regime_model not in REGIME_MODELS:
            raise ValueError(
                f"Unknown regime model {self.regime_model!r}; choose from {REGIME_MODELS}."
            )
        unknown = sorted(set(self.strategies) - set(STRATEGIES))
        if unknown or not self.strategies:
            raise ValueError(
                f"Unknown or empty strategies {unknown}; choose from {list(STRATEGIES)}."
            )
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be at least 2.")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Trading costs cannot be negative.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ExperimentSpec:
        """Build a specification from a mapping, rejecting unknown keys."""
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        data = dict(values)
        if "strategies" in data:
            data["strategies"] = tuple(data["strategies"])
        for key in ("start", "end", "train_end"):
            if key in data:
                data[key] = str(data[key])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentSpec:
        """Load a specification from a YAML file."""
        values = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(values, dict):
            raise ValueError(f"{path} must contain a mapping of settings.")
        return cls.from_dict(values)
