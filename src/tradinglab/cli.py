"""Command-line interface: ``tradinglab <command>``.

    fetch        download / update market data into the local cache
    research     run the strategy tournament, select the live portfolio, train live models
    train-live   refit the live deep models on all data (keeps the selection)
    backtest     backtest one strategy (walk-forward for trainable ones)
    probe        check which real-time exchange APIs are reachable
    cycle        run one live paper-trading cycle (idempotent; cron / CI friendly)
    run          autonomous daemon: trade + periodic retraining/research, forever
    status       print the live accounts
    report       write the HTML dashboard (live + research)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tradinglab.config import load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "matplotlib", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def _engine(cfg):
    from tradinglab.data.store import MarketData
    from tradinglab.live.engine import LiveEngine
    from tradinglab.live.feed import LiveFeed
    from tradinglab.live.state import StateStore

    store = StateStore(Path(cfg.paths.state_dir) / "lab.sqlite")
    feed = LiveFeed(MarketData(cfg.paths.data_dir), cfg.universe.symbols, cfg.walkforward.data_start,
                    cfg.live.feed, default_funding_8h=cfg.market.default_funding_8h)
    return LiveEngine(cfg, feed, store), store


def cmd_fetch(cfg, args) -> int:
    from tradinglab.data.store import MarketData

    p = MarketData(cfg.paths.data_dir).panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start,
                                             exec_lag_hours=cfg.market.exec_lag_hours)
    print(f"{p.close.shape[1]} symbols, {len(p.close)} bars, {p.close.index[0]} → {p.close.index[-1]}")
    return 0


def cmd_research(cfg, args) -> int:
    from tradinglab.research.pipeline import run_research

    sel = run_research(cfg, include_deep=not args.no_deep, workers=args.workers, train_live=not args.no_train)
    print(json.dumps({k: sel[k] for k in ("strategies", "alphas", "holdout", "live_rules_oos")}, indent=2,
                     default=str))
    return 0


def cmd_train_live(cfg, args) -> int:
    from tradinglab.data.store import MarketData
    from tradinglab.live.engine import load_selection
    from tradinglab.research.pipeline import selection_path, train_live_models

    sel = load_selection(cfg)
    kinds = [n[3:] for n in sel.get("deep_models", {})]
    if not kinds:
        print("no live deep models in the selection – nothing to retrain")
        return 0
    panel = MarketData(cfg.paths.data_dir).panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start,
                                                 exec_lag_hours=cfg.market.exec_lag_hours)
    sel["deep_models"] = train_live_models(cfg, panel, kinds)
    selection_path(cfg).write_text(json.dumps(sel, indent=2, default=str))
    print("retrained:", ", ".join(sel["deep_models"]))
    return 0


def cmd_backtest(cfg, args) -> int:
    from tradinglab.config import bars_per_year
    from tradinglab.data.store import MarketData
    from tradinglab.research.tournament import evaluate_candidate
    from tradinglab.strategies.registry import by_name

    panel = MarketData(cfg.paths.data_dir).panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start,
                                                 exec_lag_hours=cfg.market.exec_lag_hours)
    res = evaluate_candidate(by_name(cfg, args.strategy), panel, cfg, workers=args.workers)
    print(json.dumps(res.backtest.metrics(bars_per_year(cfg.market.bar)), indent=2, default=str))
    return 0


def cmd_probe(cfg, args) -> int:
    from tradinglab.data.exchanges import default_exchanges, probe

    ok = probe(default_exchanges())
    print("reachable exchange APIs:", ", ".join(ok) if ok else "none (archive mode will be used)")
    return 0


def cmd_cycle(cfg, args) -> int:
    engine, store = _engine(cfg)
    rep = engine.cycle()
    print(json.dumps({"mode": rep.mode, "horizon": str(rep.horizon), "signal_bar": str(rep.signal_bar),
                      "fills": rep.fills, "funding_events": rep.funding_events, "marks": rep.marks,
                      "decisions": rep.decisions, "orders": rep.orders, "warnings": rep.warnings}, indent=2))
    if args.report:
        from tradinglab.report.html import live_report

        live_report(cfg, store, Path(cfg.paths.reports_dir) / "live.html")
    store.close()
    return 0


def cmd_run(cfg, args) -> int:
    from tradinglab.live.daemon import Daemon

    Daemon(cfg, args.config, research_days=args.research_days, max_cycles=args.max_cycles,
           git_commit=args.git_commit).run()
    return 0


def cmd_status(cfg, args) -> int:
    path = Path(cfg.paths.state_dir) / "status.json"
    if not path.exists():
        print("no live state yet – run `tradinglab cycle` first")
        return 1
    st = json.loads(path.read_text())
    print(f"updated {st['updated_at']}  feed={st['feed_mode']}  data horizon={st['data_horizon']}")
    print(f"{'account':28s} {'equity':>12s} {'P&L':>11s} {'return':>8s} {'gross':>6s} {'marked at':>26s}")
    for name, a in sorted(st["accounts"].items(), key=lambda kv: (kv[0] != "main", -kv[1]["return"])):
        print(f"{name:28s} {a['equity']:12,.2f} {a['pnl']:+11,.2f} {100 * a['return']:+7.2f}% "
              f"{a['gross']:6.2f} {str(a['marked_at']):>26s}")
    return 0


def cmd_report(cfg, args) -> int:
    from tradinglab.live.state import StateStore
    from tradinglab.report.html import live_report

    store = StateStore(Path(cfg.paths.state_dir) / "lab.sqlite")
    out = live_report(cfg, store, Path(cfg.paths.reports_dir) / "live.html")
    print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tradinglab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config layered over config/default.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    p = sub.add_parser("research")
    p.add_argument("--no-deep", action="store_true", help="skip the deep-learning candidates")
    p.add_argument("--workers", type=int, default=None, help="parallel walk-forward folds")
    p.add_argument("--no-train", action="store_true", help="reuse existing live deep models instead of refitting")
    sub.add_parser("train-live")
    p = sub.add_parser("backtest")
    p.add_argument("strategy")
    p.add_argument("--workers", type=int, default=None)
    sub.add_parser("probe")
    p = sub.add_parser("cycle")
    p.add_argument("--report", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--research-days", type=int, default=30)
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--git-commit", action="store_true", help="commit + push state/ after trading events")
    sub.add_parser("status")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    handler = {"fetch": cmd_fetch, "research": cmd_research, "train-live": cmd_train_live,
               "backtest": cmd_backtest, "probe": cmd_probe, "cycle": cmd_cycle, "run": cmd_run,
               "status": cmd_status, "report": cmd_report}[args.cmd]
    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
