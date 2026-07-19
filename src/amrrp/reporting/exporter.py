"""Artifact export utilities."""

from pathlib import Path

import pandas as pd


def export_table(table: pd.DataFrame, path: Path) -> Path:
    """Export a table as CSV, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return path
