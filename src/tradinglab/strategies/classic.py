"""Classic, literature-standard systematic strategies.

Parameters are fixed at textbook values (not tuned on our data) so that they
cost almost nothing in multiple-testing terms and serve as honest baselines for
the deep models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.data.store import Panel
from tradinglab.features import ewm_vol, macd_response, macd_signal
from tradinglab.strategies.base import Strategy


def _long_only(sig: pd.DataFrame, long_only: bool) -> pd.DataFrame:
    return sig.clip(lower=0.0) if long_only else sig


def _available(panel: Panel) -> pd.DataFrame:
    return panel.close.notna()


class BuyAndHold(Strategy):
    """Volatility-targeted long position in a single asset (benchmark)."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.name = f"hold_{symbol.removesuffix('USDT').lower()}"

    def signals(self, panel: Panel) -> pd.DataFrame:
        sig = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
        if self.symbol in sig:
            sig[self.symbol] = 1.0
        return sig.where(_available(panel))


class EqualWeightLong(Strategy):
    """Volatility-normalised long basket of every available asset (market beta)."""

    name = "basket_long"

    def signals(self, panel: Panel) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=panel.close.index, columns=panel.close.columns).where(_available(panel))


class TSMOM(Strategy):
    """Time-series momentum (Moskowitz, Ooi & Pedersen 2012), blended over lookbacks."""

    def __init__(self, lookbacks=(21, 63, 126, 252), long_only: bool = False, name: str | None = None):
        self.lookbacks = tuple(lookbacks)
        self.long_only = long_only
        self.name = name or ("tsmom_lf" if long_only else "tsmom")

    def signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        parts = [np.sign(close.pct_change(h, fill_method=None)) for h in self.lookbacks]
        sig = sum(parts) / len(parts)
        return _long_only(sig, self.long_only).where(_available(panel))


class MACDTrend(Strategy):
    """Baz et al. (2015) multi-timescale MACD trend signal with φ response."""

    def __init__(self, pairs=((8, 24), (16, 48), (32, 96)), long_only: bool = False, name: str | None = None):
        self.pairs = tuple(pairs)
        self.long_only = long_only
        self.name = name or ("macd_lf" if long_only else "macd")

    def signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        sig = sum(macd_response(macd_signal(close, s, l)) for s, l in self.pairs) / len(self.pairs)
        return _long_only(sig.clip(-1, 1), self.long_only).where(_available(panel))


class Breakout(Strategy):
    """Donchian channel position (Turtle-style), averaged over channel lengths."""

    def __init__(self, windows=(20, 55, 100), long_only: bool = False, name: str | None = None):
        self.windows = tuple(windows)
        self.long_only = long_only
        self.name = name or ("breakout_lf" if long_only else "breakout")

    def signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        parts = []
        for w in self.windows:
            hi = close.rolling(w, min_periods=w).max()
            lo = close.rolling(w, min_periods=w).min()
            rng = (hi - lo).replace(0.0, np.nan)
            parts.append(2.0 * (close - lo) / rng - 1.0)
        sig = sum(parts) / len(parts)
        return _long_only(sig, self.long_only).where(_available(panel))


class CrossSectionalMomentum(Strategy):
    """Cross-sectional momentum: long recent winners, short losers (dollar neutral)."""

    def __init__(self, lookback: int = 63, skip: int = 0, name: str | None = None, reverse: bool = False):
        self.lookback = lookback
        self.skip = skip
        self.reverse = reverse
        self.name = name or (f"xs_rev_{lookback}" if reverse else f"xsmom_{lookback}")

    def signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        vol = ewm_vol(close.pct_change(fill_method=None))
        score = close.shift(self.skip).pct_change(self.lookback, fill_method=None) / (vol * np.sqrt(self.lookback))
        rank = score.rank(axis=1, pct=True)
        n = score.notna().sum(axis=1)
        sig = (rank.sub(rank.mean(axis=1), axis=0) * 2.0)
        sig = sig.where(n >= 5, 0.0)
        if self.reverse:
            sig = -sig
        return sig.clip(-1, 1).where(_available(panel))


def classic_candidates(symbols: list[str]) -> list[Strategy]:
    """The classic strategy zoo evaluated by the research tournament."""
    out: list[Strategy] = [
        BuyAndHold("BTCUSDT"),
        EqualWeightLong(),
        TSMOM(),
        TSMOM(long_only=True),
        MACDTrend(),
        MACDTrend(long_only=True),
        Breakout(),
        Breakout(long_only=True),
        CrossSectionalMomentum(63),
        CrossSectionalMomentum(5, reverse=True),
    ]
    return out
