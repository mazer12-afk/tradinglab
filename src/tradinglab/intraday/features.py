"""Causal intraday features for 5-minute perpetual-futures bars.

Every feature at row t uses only bars whose close time is <= t. Families:

* returns over 5 min … 1 day, normalised by a 1-day EWMA volatility;
* volatility regime (short/long vol ratio, vol z-score), bar ranges, close location;
* order flow: taker-buy share of traded notional over several windows (a proxy
  for order-flow imbalance), and its z-score;
* activity: volume and trade-count surprise vs a 1-week EWMA;
* basis: perpetual premium index level and change (funding pressure);
* calendar: time of day and day of week (crypto has strong intraday seasonality);
* market context: cross-sectional average returns and each asset's excess return.

Target: the vol-normalised log return from the next fill (next bar open) to the
fill ``horizon`` bars later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglab.intraday.data import IntradayData

RET_LAGS = (1, 2, 3, 6, 12, 24, 48, 96, 288)
WEEK = 288 * 7
CLIP = 6.0


@dataclass
class FeatureSet:
    index: pd.DatetimeIndex
    symbols: list[str]
    names: list[str]
    X: np.ndarray        # [T, N, F] float32
    y: np.ndarray        # [T, N] vol-normalised forward return (NaN where unknown)
    sigma: np.ndarray    # [T, N] per-bar volatility estimate (fraction)

    def rows(self, t_idx: np.ndarray, need_y: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stack (t, asset) rows for tabular models → (X, y, t, asset) without NaN rows."""
        X = self.X[t_idx]                                     # [n, N, F]
        n, N, F = X.shape
        Xf = X.reshape(n * N, F)
        yf = self.y[t_idx].reshape(n * N)
        tt = np.repeat(t_idx, N)
        aa = np.tile(np.arange(N), n)
        ok = np.isfinite(Xf).all(axis=1)
        if need_y:
            ok &= np.isfinite(yf)
        return Xf[ok], yf[ok], tt[ok], aa[ok]


def _ewm_mean(df: pd.DataFrame, span: int) -> pd.DataFrame:
    return df.ewm(span=span, min_periods=span // 4).mean()


def _z(df: pd.DataFrame, span: int = WEEK) -> pd.DataFrame:
    mu = _ewm_mean(df, span)
    sd = np.sqrt(_ewm_mean((df - mu) ** 2, span))
    return (df - mu) / sd.replace(0.0, np.nan)


def build_features(d: IntradayData, horizon: int = 12) -> FeatureSet:
    close = d.close
    lp = np.log(close)
    r1 = lp.diff()
    var = _ewm_mean(r1 ** 2, 288)
    sigma = np.sqrt(var).clip(lower=1e-5)
    f: dict[str, pd.DataFrame] = {}

    for k in RET_LAGS:
        f[f"ret_{k}"] = (lp - lp.shift(k)) / (sigma * np.sqrt(k))
    f["vol_ratio"] = 0.5 * np.log(_ewm_mean(r1 ** 2, 12) / var)
    f["vol_z"] = _z(np.log(sigma))
    f["range_1"] = np.log(d.high / d.low) / sigma
    hi12 = d.high.rolling(12, min_periods=6).max()
    lo12 = d.low.rolling(12, min_periods=6).min()
    f["range_12"] = np.log(hi12 / lo12) / (sigma * np.sqrt(12))
    f["cloc_12"] = (close - lo12) / (hi12 - lo12).replace(0.0, np.nan) - 0.5

    qv = d.quote_volume
    tb = d.taker_buy_quote
    for k in (1, 6, 24, 96):
        f[f"ofi_{k}"] = 2.0 * tb.rolling(k).sum() / qv.rolling(k).sum().replace(0.0, np.nan) - 1.0
    f["ofi_z_6"] = _z(f["ofi_6"])
    base_vol = _ewm_mean(qv, WEEK)
    for k in (1, 12):
        f[f"volu_{k}"] = np.log((qv.rolling(k).mean() + 1.0) / (base_vol + 1.0))
    f["trades_1"] = np.log((d.trades + 1.0) / (_ewm_mean(d.trades, WEEK) + 1.0))

    prem = d.premium
    f["prem"] = prem * 1e3
    f["prem_z"] = _z(prem)
    f["prem_chg_12"] = (prem - prem.shift(12)) / np.sqrt(_ewm_mean((prem - prem.shift(12)) ** 2, WEEK))

    idx = close.index
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    dow = idx.dayofweek.to_numpy()
    cal = {"tod_sin": np.sin(2 * np.pi * minutes / 1440), "tod_cos": np.cos(2 * np.pi * minutes / 1440),
           "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7)}
    for name, v in cal.items():
        f[name] = pd.DataFrame(np.repeat(v[:, None], close.shape[1], axis=1), index=idx, columns=close.columns)

    for k in (1, 12, 48):
        mkt = f[f"ret_{k}"].mean(axis=1)
        f[f"mkt_{k}"] = pd.DataFrame(np.repeat(mkt.to_numpy()[:, None], close.shape[1], axis=1),
                                     index=idx, columns=close.columns)
    for k in (12, 48, 288):
        f[f"xs_{k}"] = f[f"ret_{k}"].sub(f[f"ret_{k}"].mean(axis=1), axis=0)

    names = list(f)
    X = np.stack([f[n].to_numpy(dtype="float32") for n in names], axis=-1)
    X = np.clip(X, -CLIP * 3, CLIP * 3)
    X[~np.isfinite(X)] = np.nan

    ex = np.log(d.exec_price())
    fwd = (ex.shift(-horizon) - ex) / (sigma * np.sqrt(horizon))
    y = fwd.clip(-CLIP, CLIP).to_numpy(dtype="float32")
    return FeatureSet(idx, list(close.columns), names, X, y, sigma.to_numpy(dtype="float64"))
