"""Command-line entry points: ``amrrp run`` and ``amrrp dashboard``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="Adaptive market regime research platform.")


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="YAML experiment specification.")] = Path(
        "configs/default.yaml"
    ),
    model: Annotated[
        str | None,
        typer.Option(help="Override regime_model (gmm, hmm, kmeans, bayesian_changepoint)."),
    ] = None,
    symbol: Annotated[str | None, typer.Option(help="Override the ticker symbol.")] = None,
    artifacts: Annotated[Path, typer.Option(help="Output directory.")] = Path("artifacts"),
    cache: Annotated[Path, typer.Option(help="Market-data cache directory.")] = Path("data/cache"),
    refresh: Annotated[bool, typer.Option(help="Ignore cached prices and re-download.")] = False,
) -> None:
    """Download prices, detect regimes and backtest strategies."""
    from dataclasses import replace

    from amrrp.data.cache import MarketDataCache
    from amrrp.data.providers import YFinanceProvider
    from amrrp.data.repository import MarketDataRepository
    from amrrp.experiments.runner import ExperimentRunner
    from amrrp.experiments.specification import ExperimentSpec

    spec = ExperimentSpec.from_yaml(config) if config.exists() else ExperimentSpec()
    if model:
        spec = replace(spec, regime_model=model)
    if symbol:
        spec = replace(spec, symbol=symbol)
    repository = MarketDataRepository(YFinanceProvider(), MarketDataCache(cache))
    prices = repository.get(spec.symbol, spec.start, spec.end, refresh=refresh)
    result = ExperimentRunner(artifacts).run(prices, spec)
    typer.echo(f"Run {result.run_id} written to {result.artifact_dir}")
    typer.echo(f"\nOut-of-sample metrics ({spec.train_end} to {spec.end}):")
    typer.echo(result.metrics.round(3).to_string())


@app.command()
def dashboard(
    artifacts: Annotated[Path, typer.Option(help="Directory of experiment runs.")] = Path(
        "artifacts"
    ),
) -> None:
    """Launch the Streamlit dashboard."""
    from amrrp.dashboard import app as dashboard_app

    script = Path(dashboard_app.__file__)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script), "--", str(artifacts)], check=True
    )


if __name__ == "__main__":
    app()
