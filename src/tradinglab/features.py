"""Causal features shared by classic and deep strategies.

All functions take wide frames (index = bar close time, columns = symbols) and
only use information available at each row's timestamp. ``tests/test_features``
verifies this by recomputing on truncated histories.

The deep-learning feature set follows Lim, Zohren & Roberts (2019) "Enhancing
Time-Series Momentum Strategies Using Deep Neural Networks" and Wood et al.
(2021/2026) Momentum Transformer / DeePM: volatility-normalised returns over
several horizons, multi-scale MACD signals and price z-scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Horizons in *bars*. For daily bars these are 1d, 1w, 1m, 3m, 6m, 1y.
RET_HORIZONS = {"r1": 1, "r5": 5, "r21": 21, "r63": 63, "r126": 126, "r252": 252}
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]
VOL_SPAN = 60

DEEP_FEATURES = ["r1", "r5", "r21", "r63", "r126", "r252",
                 "macd_8_24", "macd_16_48", "macd_32_96",
                 "z_price_21", "z_price_252", "z_logvol"]


def ewm_vol(returns: pd.DataFrame, span: int = VOL_SPAN, min_periods: int | None = None) -> pd.DataFrame:
    """Per-bar EWMA volatility of simple returns."""
    return returns.ewm(span=span, min_periods=min_periods or span // 2).std()


def _halflife(timescale: float) -> float:
    return np.log(0.5) / np.log(1 - 1 / timescale)


def macd_signal(close: pd.DataFrame, short: float, long: float) -> pd.DataFrame:
    """Baz et al. (2015) normalised MACD: (EWMA_s - EWMA_l) / std(price, 63) / std(q, 252)."""
    macd = (close.ewm(halflife=_halflife(short)).mean()
            - close.ewm(halflife=_halflife(long)).mean())
    q = macd / close.rolling(63, min_periods=30).std()
    return q / q.rolling(252, min_periods=63).std()


def macd_response(x: pd.DataFrame | np.ndarray):
    """Baz et al. position response φ(x) = x·exp(-x²/4)/0.89, bounded in ~[-1, 1]."""
    return x * np.exp(-(x ** 2) / 4.0) / 0.89


def deep_features(close: pd.DataFrame, clip: float = 5.0) -> dict[str, pd.DataFrame]:
    """Feature frames keyed by feature name (each frame: time × symbol)."""
    rets = close.pct_change(fill_method=None)
    vol = ewm_vol(rets)
    feats: dict[str, pd.DataFrame] = {}
    for name, h in RET_HORIZONS.items():
        feats[name] = (close.pct_change(h, fill_method=None) / (vol * np.sqrt(h))).clip(-clip, clip)
    for s, l in MACD_PAIRS:
        feats[f"macd_{s}_{l}"] = macd_signal(close, s, l).clip(-clip, clip)
    logp = np.log(close)
    for w in (21, 252):
        mu = logp.rolling(w, min_periods=w // 2).mean()
        sd = logp.rolling(w, min_periods=w // 2).std()
        feats[f"z_price_{w}"] = ((logp - mu) / sd).clip(-clip, clip)
    logvol = np.log(vol + 1e-8)
    feats["z_logvol"] = ((logvol - logvol.rolling(252, min_periods=63).mean())
                         / logvol.rolling(252, min_periods=63).std()).clip(-clip, clip)
    return feats


def stack_features(feats: dict[str, pd.DataFrame], names: list[str]) -> np.ndarray:
    """→ array [time, symbol, feature]."""
    return np.stack([feats[n].to_numpy(dtype="float32") for n in names], axis=-1)
