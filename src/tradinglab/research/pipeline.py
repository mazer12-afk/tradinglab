"""End-to-end research run: data → tournament → selection → live models → report."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from tradinglab.config import LabConfig, utc
from tradinglab.data.store import MarketData
from tradinglab.research.tournament import run_tournament, save_report
from tradinglab.strategies.registry import DEEP_KINDS, build_candidates, deep_available, make_deep

log = logging.getLogger(__name__)


def selection_path(cfg: LabConfig) -> Path:
    return Path(cfg.paths.models_dir) / "selection.json"


def train_live_models(cfg: LabConfig, panel, kinds) -> dict[str, str]:
    """Fit deep models on *all* available data for live use; returns name → path."""
    out = {}
    train_end = panel.close.index[-1] + pd.Timedelta(days=2)
    for kind in kinds:
        strat = make_deep(kind, cfg)
        t0 = time.time()
        strat.fit(panel, train_end)
        path = Path(cfg.paths.models_dir) / strat.name
        strat.save(path)
        out[strat.name] = str(path)
        log.info("trained live model %s in %.0fs", strat.name, time.time() - t0)
    return out


def run_research(cfg: LabConfig, include_deep: bool = True, workers: int | None = None,
                 train_live: bool = True) -> dict:
    t0 = time.time()
    md = MarketData(cfg.paths.data_dir)
    panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start,
                     exec_lag_hours=cfg.market.exec_lag_hours,
                     default_funding_8h=cfg.market.default_funding_8h)
    log.info("panel %s → %s, %d symbols", panel.close.index[0], panel.close.index[-1], panel.close.shape[1])
    strategies = build_candidates(cfg, include_deep=include_deep)
    report, results = run_tournament(strategies, panel, cfg, cfg.walkforward.holdout_start, workers=workers,
                                     cache_dir=Path(cfg.paths.models_dir) / "cache" / "walkforward")
    out_dir = Path(cfg.paths.reports_dir) / "research"
    save_report(report, results, out_dir)

    deep_paths = {}
    if include_deep and deep_available():
        kinds = [k for k in DEEP_KINDS if cfg.live.shadow_all or f"dl_{k}" in report.live_choice]
        if train_live:
            deep_paths = train_live_models(cfg, panel, kinds)
        else:  # keep previously trained live models
            for k in kinds:
                path = Path(cfg.paths.models_dir) / f"dl_{k}"
                if (path / "weights.pt").exists():
                    deep_paths[f"dl_{k}"] = str(path)

    selection = {
        "created_at": report.created_at,
        "data_end": report.data_end,
        "strategies": report.live_choice,
        "alphas": report.live_alphas,
        "candidates": list(results),
        "deep_models": deep_paths,
        "holdout": {"window": report.holdout_window, "choice": report.holdout_choice,
                    "metrics": report.holdout_ensemble},
        "live_rules_oos": report.live_ensemble_oos,
        "config": cfg.to_dict(),
    }
    path = selection_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2, default=str))
    try:
        from tradinglab.report.html import research_report

        research_report(report, out_dir / "index.html")
    except Exception as exc:  # noqa: BLE001 - report generation must not fail research
        log.warning("research report failed: %s", exc)
    log.info("research finished in %.0fs; live selection: %s", time.time() - t0, report.live_choice)
    return selection
