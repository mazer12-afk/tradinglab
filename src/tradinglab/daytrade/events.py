"""Extreme-move events with futures-metrics features, and the walk-forward LightGBM forecast on them.

The research behind it is in ``scripts/daytrade2`` and ``reports/daytrade2``. Timing: the 5-minute bar ending
at T is judged at T + 1 minute; the decision uses the close of the minute starting at T, and any trade fills
from the next minute.

Futures metrics and their publication delay: the archive's value stamped X is Binance's live value stamped
X + 5 minutes (open interest, long/short ratios) or X (taker ratio), published about 1-2.3 and 7.6 minutes after
those stamps (measured on the live API on 1 Nov 2026). An archive value stamped X is therefore known only from
about X + 8 minutes, so with ``metrics_lag=2`` a decision at T + 1 minute uses values stamped T - 10 minutes or
earlier. The first research runs used ``metrics_lag=0``, which looked about 6-8 minutes ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradinglab.daytrade.sim import Minutes

HORIZON = 240                      # minutes: the forecast and holding horizon
EVENT_Z = 3.0
PARAMS = {"objective": "huber", "alpha": 0.02, "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1,
          "num_threads": 4, "seed": 0, "deterministic": True}
ROUNDS = 200
FEATURES = ["z", "iz", "abs_z", "doi_z", "sig5", "sig5_rel", "ret_1h", "ret_4h", "ret_1d", "doi_1h", "doi_1d",
            "taker_5m", "taker_1h", "top_ls", "top_ls_1h", "crowd_ls", "funding", "mkt_1h", "mkt_1d", "mkt_3d",
            "hour", "dow", "to_settle"]


METRICS_LAG = 2                    # 5-minute steps between a metrics stamp and the first decision that may use it


def event_table(m: Minutes, metrics: dict[str, pd.DataFrame], funding: dict[str, pd.Series],
                start=None, end=None, metrics_lag: int = METRICS_LAG) -> pd.DataFrame:
    """One row per event (5-minute |z| > EVENT_Z) with its features, the coin, the decision close and sigma,
    and the label (4-hour forward log return; NaN until known). Rows from ``start`` to ``end`` (decision time)."""
    c = m.close
    syms = list(c.columns)
    lc = np.log(c.ffill())
    grid = c.index[(c.index.minute % 5 == 0)]
    L = lc.reindex(grid)
    r5 = (lc.shift(1) - lc.shift(6)).reindex(grid)
    sig = r5.ewm(span=288, min_periods=100).std().shift(1)
    z = r5 / sig
    idio = r5.sub(r5.mean(axis=1), axis=0)
    iz = idio / idio.ewm(span=288, min_periods=100).std().shift(1)
    met = {s: metrics[s].reindex(grid).shift(metrics_lag) for s in syms}
    oi = pd.DataFrame({s: np.log(met[s]["sum_open_interest_value"]) for s in syms})
    doi = oi.diff()
    taker = pd.DataFrame({s: np.log(met[s]["sum_taker_long_short_vol_ratio"]) for s in syms})
    top = pd.DataFrame({s: np.log(met[s]["sum_toptrader_long_short_ratio"]) for s in syms})
    fr = pd.DataFrame({s: funding.get(s, pd.Series(dtype="float64")) for s in syms})
    fr.index = pd.DatetimeIndex(fr.index).round("min")
    mk = lc.diff().mean(axis=1).fillna(0.0).cumsum().reindex(grid)

    def bc(x):
        return pd.DataFrame(np.repeat(np.asarray(x)[:, None], len(syms), axis=1), index=grid, columns=syms)

    feat = {
        "z": z, "iz": iz, "abs_z": z.abs(), "doi_z": doi / doi.ewm(span=288, min_periods=100).std().shift(1),
        "sig5": np.log(sig), "sig5_rel": np.log(sig / sig.rolling(288 * 7, min_periods=288).mean()),
        "ret_1h": L - L.shift(12), "ret_4h": L - L.shift(48), "ret_1d": L - L.shift(288),
        "doi_1h": oi.diff(12), "doi_1d": oi.diff(288),
        "taker_5m": taker, "taker_1h": taker.rolling(12, min_periods=8).mean(),
        "top_ls": top, "top_ls_1h": top.diff(12),
        "crowd_ls": pd.DataFrame({s: np.log(met[s]["count_long_short_ratio"]) for s in syms}),
        "funding": fr.reindex(grid, method="ffill"),
        "mkt_1h": bc(mk - mk.shift(12)), "mkt_1d": bc(mk - mk.shift(288)), "mkt_3d": bc(mk - mk.shift(864)),
        "hour": bc(grid.hour), "dow": bc(grid.dayofweek), "to_settle": bc(((8 - grid.hour % 8) * 60 - grid.minute) % 480),
    }
    ev = z.abs() > EVENT_Z
    if start is not None:
        ev &= bc(grid >= pd.Timestamp(start))
    if end is not None:
        ev &= bc(grid < pd.Timestamp(end))
    st = ev.stack()
    idx = st[st].index
    out = pd.DataFrame({n: feat[n].stack().reindex(idx).to_numpy(dtype="float64") for n in FEATURES})
    out.insert(0, "t", idx.get_level_values(0))
    out.insert(1, "symbol", idx.get_level_values(1))
    out["coin"] = pd.Categorical(out["symbol"], categories=syms).codes
    out["close"] = c.reindex(grid).stack().reindex(idx).to_numpy()
    out["sigma"] = sig.stack().reindex(idx).to_numpy()
    fwd = lc.shift(-HORIZON).reindex(grid) - L
    out["label"] = fwd.stack().reindex(idx).to_numpy()
    return out


def train(events: pd.DataFrame, month_start: pd.Timestamp):
    """The model used during the month starting at ``month_start``: trained on every event whose label was
    complete before it (decision + horizon + 10 minutes)."""
    import lightgbm as lgb

    cut = pd.Timestamp(month_start) - pd.Timedelta(minutes=HORIZON + 10)
    tr = events[(events["t"] < cut) & np.isfinite(events["label"])]
    X = np.column_stack([tr[FEATURES].to_numpy(dtype="float64"), tr["coin"].to_numpy()])
    return lgb.train(PARAMS, lgb.Dataset(X, tr["label"].to_numpy(), categorical_feature=[X.shape[1] - 1]),
                     num_boost_round=ROUNDS)


def predict(model, events: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([events[FEATURES].to_numpy(dtype="float64"), events["coin"].to_numpy()])
    return model.predict(X)


def intents(events: pd.DataFrame, pred: np.ndarray, theta: float, hold: int = HORIZON, size: float = 0.2) -> pd.DataFrame:
    """Market buys for events whose forecast exceeds ``theta`` (strongest first within a minute)."""
    sel = pred > theta
    e = events[sel]
    return pd.DataFrame({"t": pd.DatetimeIndex(e["t"]), "symbol": e["symbol"].to_numpy(), "side": 1, "size": size,
                         "entry": "market", "price": np.nan, "ttl": 0, "tp": 0.0, "sl": 0.0, "max_hold": hold,
                         "priority": pred[sel]})
