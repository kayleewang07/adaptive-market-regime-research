"""Market-data cache management."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class MarketDataCache:
    """Filesystem parquet cache keyed by symbol and inclusive date range."""

    def __init__(self, directory: Path) -> None:
        """Initialize a cache rooted at ``directory``."""
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, start: str, end: str) -> Path:
        """Build the deterministic cache path for one request."""
        return self.directory / f"{symbol.upper()}_{start}_{end}.parquet"

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """Load a cached frame, or return ``None`` if absent."""
        path = self.path_for(symbol, start, end)
        return pd.read_parquet(path) if path.exists() else None

    def save(self, frame: pd.DataFrame, symbol: str, start: str, end: str) -> Path:
        """Persist a cleaned market frame and return its path."""
        path = self.path_for(symbol, start, end)
        frame.to_parquet(path)
        return path
