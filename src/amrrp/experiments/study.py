"""Multi-asset robustness study: does a finding on one asset hold on others?"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .runner import REGIME_SWITCHING, ExperimentRunner
from .specification import ExperimentSpec
from .tracking import create_run_directory

TREND = ("momentum", "moving_average_crossover")


def summarize_run(directory: Path, symbol: str) -> dict[str, object]:
    """Extract the headline out-of-sample numbers from one experiment's artifacts."""
    metrics = pd.read_csv(directory / "metrics_test.csv", index_col=0)
    inference = pd.read_csv(directory / "sharpe_inference_test.csv", index_col=0)
    by_regime = pd.read_csv(directory / "regime_metrics_test.csv", index_col=[0, 1])["sharpe"]
    summary = pd.read_csv(directory / "regime_summary_test.csv", index_col=0)
    sharpe = by_regime.unstack("strategy")
    calm, turbulent = sharpe.index.min(), sharpe.index.max()
    trend = [name for name in TREND if name in sharpe.columns]
    row: dict[str, object] = {
        "symbol": symbol,
        "buy_and_hold_sharpe": metrics.loc["buy_and_hold", "sharpe"],
        "best_strategy": metrics["sharpe"].idxmax(),
        "calm_volatility": summary.loc[calm, "annualized_volatility"],
        "turbulent_volatility": summary.loc[turbulent, "annualized_volatility"],
        "turbulent_share": summary.loc[turbulent, "share_of_days"],
    }
    if trend:
        row["trend_sharpe_calm"] = sharpe.loc[calm, trend].mean()
        row["trend_sharpe_turbulent"] = sharpe.loc[turbulent, trend].mean()
    if "mean_reversion" in sharpe.columns:
        row["mean_reversion_sharpe_turbulent"] = sharpe.loc[turbulent, "mean_reversion"]
    if REGIME_SWITCHING in metrics.index:
        row.update(
            {
                "switching_sharpe": metrics.loc[REGIME_SWITCHING, "sharpe"],
                "switching_minus_bh": inference.loc[REGIME_SWITCHING, "diff_vs_benchmark"],
                "diff_ci_low": inference.loc[REGIME_SWITCHING, "diff_ci_low"],
                "diff_ci_high": inference.loc[REGIME_SWITCHING, "diff_ci_high"],
                "p_value": inference.loc[REGIME_SWITCHING, "p_value"],
                "buy_and_hold_max_drawdown": metrics.loc["buy_and_hold", "max_drawdown"],
                "switching_max_drawdown": metrics.loc[REGIME_SWITCHING, "max_drawdown"],
            }
        )
    return row


def run_study(
    symbols: Iterable[str],
    specification: ExperimentSpec,
    load_prices: Callable[[str, str, str], pd.DataFrame],
    artifact_root: Path = Path("artifacts"),
) -> tuple[Path, pd.DataFrame]:
    """Run the same experiment on each symbol and write a cross-asset summary table."""
    _, directory = create_run_directory(artifact_root, f"study-{specification.regime_model}")
    runner = ExperimentRunner(directory)
    rows = []
    for symbol in symbols:
        spec = replace(specification, symbol=symbol)
        prices = load_prices(symbol, spec.start, spec.end)
        result = runner.run(prices, spec)
        rows.append(summarize_run(result.artifact_dir, symbol))
    table = pd.DataFrame(rows).set_index("symbol")
    table.to_csv(directory / "study_summary.csv")
    (directory / "study_summary.md").write_text(
        f"# Cross-asset study ({specification.regime_model}, out of sample "
        f"{specification.train_end} to {specification.end})\n\n"
        + table.round(3).to_markdown()
        + "\n"
    )
    return directory, table
