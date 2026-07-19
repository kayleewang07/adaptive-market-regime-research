"""Research report assembly."""

from pathlib import Path

import pandas as pd


def build_markdown_report(metrics: pd.DataFrame, output: Path) -> Path:
    """Write a compact Markdown research report and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# Experiment Report\n\n## Performance Metrics\n\n" + metrics.to_markdown())
    return output
