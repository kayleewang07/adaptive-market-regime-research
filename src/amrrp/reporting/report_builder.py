"""Research report assembly."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from amrrp.experiments.specification import ExperimentSpec


def build_markdown_report(
    specification: ExperimentSpec,
    *,
    metrics_test: pd.DataFrame,
    metrics_train: pd.DataFrame,
    regime_summary: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    inference: pd.DataFrame,
    regime_map: dict[int, str],
    importance: pd.Series,
    output: Path,
) -> Path:
    """Write a Markdown summary of one experiment and return its path."""
    settings = "\n".join(f"- **{key}**: {value}" for key, value in asdict(specification).items())
    sharpe = regime_metrics["sharpe"].unstack("strategy") if not regime_metrics.empty else None
    sections = [
        f"# Experiment report: {specification.symbol} / {specification.regime_model}",
        f"## Settings\n\n{settings}",
        f"## Out-of-sample strategy metrics ({specification.train_end} to {specification.end})\n\n"
        + metrics_test.round(3).to_markdown(),
        f"## In-sample strategy metrics ({specification.start} to {specification.train_end})\n\n"
        + metrics_train.round(3).to_markdown(),
        "## Out-of-sample regimes (underlying asset)\n\n" + regime_summary.round(3).to_markdown(),
    ]
    sections.append(
        "## Is the out-of-sample Sharpe difference real?\n\n"
        "95% stationary-bootstrap intervals (2,000 resamples, 20-day mean blocks). "
        "`diff_vs_benchmark` compares each strategy with buy-and-hold on the same "
        "resampled days.\n\n" + inference.round(3).to_markdown()
    )
    if regime_map:
        rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(regime_map.items()))
        sections.append(
            "## Regime-switching rule (chosen on training data only)\n\n"
            "| Regime | Strategy |\n|---|---|\n" + rows + "\n\n"
            "Its in-sample metrics are optimistic by construction; judge it out of sample."
        )
    if sharpe is not None:
        sections.append(
            "## Out-of-sample Sharpe ratio by regime\n\n" + sharpe.round(2).to_markdown()
        )
    sections.append(
        "## Top features separating regimes\n\n"
        + importance.head(8).round(3).to_frame().to_markdown()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sections) + "\n")
    return output
