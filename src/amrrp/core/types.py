"""Typed domain objects shared across components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Execution assumptions for one daily backtest."""

    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    periods_per_year: int = 252


@dataclass
class BacktestResult:
    """Portfolio history and summary metrics emitted by a backtest."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.Series
    turnover: pd.Series
    metrics: dict[str, float]


@dataclass(frozen=True)
class ExperimentResult:
    """Addresses of persistent artifacts from one experiment."""

    run_id: str
    artifact_dir: Path
    metrics: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
