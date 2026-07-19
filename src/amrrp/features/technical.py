"""Technical-indicator feature transformers."""

from __future__ import annotations

import pandas as pd

from .base import DataFrameTransformer


class TechnicalFeatures(DataFrameTransformer):
    """Generate configurable lag-safe technical features from OHLCV data."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        rsi_window: int = 14,
        atr_window: int = 14,
        bollinger_window: int = 20,
        volatility_window: int = 21,
        momentum_window: int = 21,
    ) -> None:
        """Store technical indicator windows."""
        self.fast, self.slow, self.rsi_window, self.atr_window = fast, slow, rsi_window, atr_window
        self.bollinger_window, self.volatility_window, self.momentum_window = (
            bollinger_window,
            volatility_window,
            momentum_window,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return technical features aligned to the input index, without look-ahead."""
        close, high, low, volume = (
            X[key].astype(float) for key in ("Close", "High", "Low", "Volume")
        )
        returns = close.pct_change()
        delta = close.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        rs = gain.rolling(self.rsi_window).mean() / loss.rolling(self.rsi_window).mean().replace(
            0, float("nan")
        )
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
        ).max(axis=1)
        ma = close.rolling(self.bollinger_window).mean()
        std = close.rolling(self.bollinger_window).std()
        fast_ma, slow_ma = close.rolling(self.fast).mean(), close.rolling(self.slow).mean()
        result = pd.DataFrame(index=X.index)
        result["return_1d"] = returns
        result["rolling_volatility"] = returns.rolling(self.volatility_window).std() * (252**0.5)
        result["realized_volatility"] = (
            returns.pow(2).rolling(self.volatility_window).sum() * 252
        ).pow(0.5)
        result["rsi"] = 100 - 100 / (1 + rs)
        result["macd"] = (
            close.ewm(span=self.fast, adjust=False).mean()
            - close.ewm(span=self.slow, adjust=False).mean()
        )
        result["atr"] = tr.rolling(self.atr_window).mean()
        result["bollinger_z"] = (close - ma) / std.replace(0, float("nan"))
        result["ma_fast"] = fast_ma / close - 1
        result["ma_slow"] = slow_ma / close - 1
        result["momentum"] = close.pct_change(self.momentum_window)
        result["volume_zscore"] = (
            volume - volume.rolling(self.volatility_window).mean()
        ) / volume.rolling(self.volatility_window).std()
        return result.shift(1)
