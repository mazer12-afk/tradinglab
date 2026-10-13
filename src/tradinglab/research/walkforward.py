"""Walk-forward (expanding window) out-of-sample signal generation.

Trainable strategies are refit at every fold start on data strictly before it
and produce signals only for the following ``step_months``. Fold results are
cached on disk keyed by (strategy, fold, data fingerprint, config fingerprint),
so an interrupted run resumes and a re-run after new data only recomputes the
latest fold.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from tradinglab.config import LabConfig, utc
from tradinglab.data.store import Panel
from tradinglab.strategies.base import Strategy

log = logging.getLogger(__name__)


def fold_starts(index: pd.DatetimeIndex, first_test: str, step_months: int) -> list[pd.Timestamp]:
    start = utc(first_test)
    last = index[-1]
    out = []
    t = start
    while t <= last:
        out.append(t)
        t = t + pd.DateOffset(months=step_months)
    return out


def _fingerprint(panel: Panel, cfg: LabConfig, name: str, end: pd.Timestamp | None) -> str:
    close = panel.close if end is None else panel.close[panel.close.index < end]
    data = pd.util.hash_pandas_object(close.fillna(-1.0), index=True).sum()
    conf = {"name": name, "deep": dataclasses.asdict(cfg.deep), "costs": dataclasses.asdict(cfg.costs),
            "avt": cfg.portfolio.asset_vol_target, "vl": cfg.portfolio.vol_lookback,
            "mh": cfg.universe.min_history_bars, "vf": cfg.walkforward.valid_fraction,
            "bar": cfg.market.bar, "lag": cfg.market.exec_lag_hours, "cols": list(close.columns),
            "rows": len(close)}
    return hashlib.sha256((str(int(data)) + json.dumps(conf, sort_keys=True)).encode()).hexdigest()[:16]


def _run_fold(args):
    strategy, panel, train_end, test_end, threads = args
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:  # noqa: BLE001
        pass
    if hasattr(strategy, "cfg"):
        strategy.cfg.threads = threads
    strategy.fit(panel, train_end)
    sig = strategy.signals(panel, start=train_end, end=test_end)
    return train_end, sig


def walk_forward_signals(strategy: Strategy, panel: Panel, cfg: LabConfig,
                         workers: int | None = None, cache_dir: Path | None = None) -> pd.DataFrame:
    """Signals where each bar is produced by a model trained only on earlier data."""
    starts = fold_starts(panel.close.index, cfg.walkforward.first_test, cfg.walkforward.step_months)
    if not strategy.trainable:
        sig = strategy.signals(panel)
        return sig.loc[starts[0]:]
    ends = starts[1:] + [None]
    parts: dict[pd.Timestamp, pd.DataFrame] = {}
    paths: dict[pd.Timestamp, Path] = {}
    jobs = []
    for s, e in zip(starts, ends):
        if cache_dir is not None:
            key = _fingerprint(panel, cfg, strategy.name, e)
            paths[s] = Path(cache_dir) / strategy.name / f"{s:%Y%m%d}_{key}.parquet"
            if paths[s].exists():
                parts[s] = pd.read_parquet(paths[s])
                continue
        jobs.append((s, e))
    if jobs:
        workers = workers or max(1, min(len(jobs), os.cpu_count() or 2))
        workers = min(workers, len(jobs))
        threads = max(1, (os.cpu_count() or 2) // workers)
        args = [(strategy, panel, s, e, threads) for s, e in jobs]

        def done(s, sig):
            parts[s] = sig
            if s in paths:
                paths[s].parent.mkdir(parents=True, exist_ok=True)
                sig.to_parquet(paths[s])
            log.info("%s fold %s done", strategy.name, s.date())

        if workers == 1:
            for a in args:
                done(*_run_fold(a))
        else:
            with ProcessPoolExecutor(workers) as pool:
                futures = [pool.submit(_run_fold, a) for a in args]
                for fut in as_completed(futures):
                    done(*fut.result())
    return pd.concat([parts[s] for s in starts]).sort_index()
