"""Train and freeze the live intraday models (spot 5m data, the live feed's source)."""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tradinglab.config import LabConfig
from tradinglab.data.exchanges import BinanceSpot
from tradinglab.data.store import MarketData
from tradinglab.intraday.data import load_intraday
from tradinglab.intraday.features import build_features
from tradinglab.intraday.policy import DEFAULT_POLICIES, PolicySpec

log = logging.getLogger(__name__)


def live_history(cfg: LabConfig, start: str | None = None):
    """Archive spot 5m history topped up with live bars up to the last closed bar."""
    ic = cfg.intraday
    md = MarketData(cfg.paths.data_dir)
    ex = BinanceSpot("https://data-api.binance.vision", "binance-data")
    hist = {s: md.klines(s, ic.bar, start or ic.data_start, market="spot") for s in ic.symbols}
    extra = {}
    for s, h in hist.items():
        since = h.index[-1] + pd.Timedelta(minutes=5)
        extra[s] = ex.klines(s, ic.bar, since, full=True)
    return load_intraday(md, ic.symbols, start or ic.data_start, ic.bar, market="spot", extra=extra)


def train_live_intraday(cfg: LabConfig, policies: list[PolicySpec] = DEFAULT_POLICIES,
                        out_dir: Path | None = None, deep_epochs: int | None = None,
                        kinds: tuple[str, ...] = ("lgbm", "deep")) -> dict:
    """Train the models used by ``policies``; with ``kinds`` restrict to lgbm and/or deep,
    keeping the other models (and their manifest entries) as they are."""
    from tradinglab.intraday.models import LGBMForecaster
    from tradinglab.intraday.research import train_rows

    ic = cfg.intraday
    out_dir = out_dir or Path(cfg.paths.models_dir) / "intraday" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = live_history(cfg)
    train_end = d.index[-1] + pd.Timedelta(minutes=5)
    old = json.loads((out_dir / "manifest.json").read_text()) if (out_dir / "manifest.json").exists() else {}
    manifest = {"trained_at": pd.Timestamp.now(tz="UTC").isoformat(), "data_end": str(d.index[-1]),
                "symbols": d.symbols, "bar": ic.bar, "source": "binance spot 5m (archive + data-api)",
                "models": {k: v for k, v in old.get("models", {}).items()
                           if (k == "deep" and "deep" not in kinds) or (k.startswith("lgbm") and "lgbm" not in kinds)}}
    try:
        manifest["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    for H in sorted({p.horizon for p in policies if p.model == "lgbm" and "lgbm" in kinds}):
        t0 = time.time()
        icH = dataclasses.replace(ic, horizon_bars=H)
        fs = build_features(d, H)
        tr, va = train_rows(fs, icH, train_end)
        rows_all = np.concatenate([tr, va])  # fixed rounds → no early-stopping tail needed
        X, y, _, a = fs.rows(rows_all)
        # Fixed number of rounds: early stopping on a 30-day tail proved unstable (1-150 trees).
        model = LGBMForecaster(learning_rate=0.02)
        model.params["learning_rate"] = 0.02
        import lightgbm as lgb

        Xa = np.column_stack([X, a])
        model.booster = lgb.train(model.params, lgb.Dataset(Xa, y, categorical_feature=[Xa.shape[1] - 1]),
                                  num_boost_round=150)
        model.best_iteration = 150
        model.save(out_dir / f"lgbm_h{H}")
        manifest["features"] = fs.names
        manifest["models"][f"lgbm_h{H}"] = {"rows": int(len(y)), "train_from": str(fs.index[rows_all[0]]),
                                           "train_to_label_end": str(fs.index[rows_all[-1]]
                                                                     + pd.Timedelta(minutes=5 * (H + 1))),
                                           "horizon_bars": H, "rounds": 150,
                                           "seconds": round(time.time() - t0, 1)}
        log.info("trained lgbm_h%d on %d rows in %.0fs", H, len(y), time.time() - t0)
    if "deep" in kinds and any(p.model == "deep" for p in policies):
        from tradinglab.intraday.deep import DeepIntradayConfig, DeepIntradayTrader, prepare

        t0 = time.time()
        fs = build_features(d, 12)
        dcfg = DeepIntradayConfig(kind="lstm_attn", cost_bps=ic.fee_bps + 2.0)
        if deep_epochs:
            dcfg.max_epochs = deep_epochs
        trader = DeepIntradayTrader(dcfg, ic.bar)
        dd = prepare(fs, d, dcfg)
        trader.fit(dd, fs.index, train_end - pd.Timedelta(days=ic.train_days), train_end)
        trader.save(out_dir / "deep")
        manifest["features"] = fs.names
        manifest["models"]["deep"] = {"kind": dcfg.kind, "val_sharpe": [s["val_sharpe"] for s in trader.states],
                                      "seconds": round(time.time() - t0, 1)}
        log.info("trained deep model in %.0fs", time.time() - t0)
    if "features" not in manifest and "features" in old:
        manifest["features"] = old["features"]
    manifest["policies"] = [dataclasses.asdict(p) for p in policies]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
