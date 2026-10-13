"""Walk-forward (expanding window) out-of-sample signal generation."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor

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
    return train_end, sig, getattr(strategy, "states", None)


def walk_forward_signals(strategy: Strategy, panel: Panel, cfg: LabConfig,
                         workers: int | None = None) -> pd.DataFrame:
    """Signals where each bar is produced by a model trained only on earlier data."""
    starts = fold_starts(panel.close.index, cfg.walkforward.first_test, cfg.walkforward.step_months)
    if not strategy.trainable:
        sig = strategy.signals(panel)
        return sig.loc[starts[0]:]
    ends = starts[1:] + [None]
    workers = workers or max(1, min(len(starts), (os.cpu_count() or 2)))
    threads = max(1, (os.cpu_count() or 2) // workers)
    jobs = [(strategy, panel, s, e, threads) for s, e in zip(starts, ends)]
    parts = {}
    if workers == 1:
        for job in jobs:
            s, sig, _ = _run_fold(job)
            parts[s] = sig
            log.info("%s fold %s done", strategy.name, s.date())
    else:
        with ProcessPoolExecutor(workers) as pool:
            for s, sig, _ in pool.map(_run_fold, jobs):
                parts[s] = sig
                log.info("%s fold %s done", strategy.name, s.date())
    return pd.concat([parts[s] for s in starts]).sort_index()
