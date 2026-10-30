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
    intraday-train / intraday-run / intraday-status   5-minute AI day-trading accounts
    experiment-run   one shift of the unattended live experiment (see tradinglab.experiment)
    simulate     the live engines on archived data: weeks of paper trading in minutes (tradinglab.simulate)
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

    import pandas as pd

    Daemon(cfg, args.config, research_days=args.research_days, max_cycles=args.max_cycles,
           git_commit=args.git_commit, maintenance=not args.no_maintenance,
           until=pd.Timestamp(args.until) if args.until else None).run()
    return 0


def cmd_status(cfg, args) -> int:
    path = Path(cfg.paths.state_dir) / "status.json"
    if not path.exists():
        print("no live state yet – run `tradinglab cycle` first")
        return 1
    st = json.loads(path.read_text())
    print(f"updated {st['updated_at']}  feed={st['feed_mode']}  data horizon={st['data_horizon']}")
    print(f"{'account':30s} {'equity':>12s} {'vault':>10s} {'P&L':>11s} {'return':>8s} {'gross':>6s} "
          f"{'marked at':>26s}")
    for name, a in sorted(st["accounts"].items(), key=lambda kv: (kv[0] != "main", -kv[1]["return"])):
        print(f"{name:30s} {a['equity']:12,.2f} {a.get('vault', 0.0):10,.2f} {a['pnl']:+11,.2f} "
              f"{100 * a['return']:+7.2f}% {a['gross']:6.2f} {str(a['marked_at']):>26s}")
    return 0


def cmd_report(cfg, args) -> int:
    from tradinglab.live.state import StateStore
    from tradinglab.report.html import live_report

    store = StateStore(Path(cfg.paths.state_dir) / "lab.sqlite")
    out = live_report(cfg, store, Path(cfg.paths.reports_dir) / "live.html")
    print(out)
    return 0


def cmd_intraday_train(cfg, args) -> int:
    from tradinglab.intraday.train import train_live_intraday

    m = train_live_intraday(cfg, deep_epochs=args.deep_epochs, kinds=tuple(args.kinds.split(",")))
    print(json.dumps({k: m[k] for k in ("trained_at", "data_end", "models")}, indent=2, default=str))
    return 0


def cmd_intraday_run(cfg, args) -> int:
    from tradinglab.intraday.live import run_intraday

    import pandas as pd

    run_intraday(cfg, poll_seconds=args.poll, max_cycles=args.max_cycles,
                 commit_every_minutes=args.commit_every, until=pd.Timestamp(args.until) if args.until else None)
    return 0


def cmd_intraday_status(cfg, args) -> int:
    path = Path(cfg.paths.state_dir) / "intraday_status.json"
    if not path.exists():
        print("no intraday state yet – run `tradinglab intraday-run`")
        return 1
    st = json.loads(path.read_text())
    print(f"updated {st['updated_at']}  last bar {st['last_bar']}")
    print(f"{'account':34s} {'equity':>12s} {'vault':>10s} {'P&L':>10s} {'return':>8s} {'trades':>7s} "
          f"{'fees':>9s} {'gross':>6s}")
    for name, a in st["accounts"].items():
        print(f"{name:34s} {a['equity']:12,.2f} {a.get('vault', 0.0):10,.2f} {a['pnl']:+10,.2f} "
              f"{100 * a['return']:+7.2f}% {a['trades']:7d} {a['fees_paid']:9,.2f} {a['gross']:6.2f}")
    attr = {k: a["attribution"] for k, a in st["accounts"].items() if a.get("attribution")}
    if attr:
        windows = {(x["window_start"], round(x["market_return"], 10)) for x in attr.values()}
        if len(windows) == 1:
            w = next(iter(attr.values()))
            print(f"\nP&L attribution since {w['window_start']} "
                  f"(equal-weight basket {100 * w['market_return']:+.2f}%):")
        else:
            print("\nP&L attribution (windows differ per account – see the last two columns):")
        print(f"{'account':26s} {'avg exp':>8s} {'beta':>6s} {'passive':>10s} {'timing':>10s} {'selection':>10s} "
              f"{'costs':>9s}" + ("" if len(windows) == 1 else f" {'since':>26s} {'basket':>7s}"))
        for name, x in attr.items():
            beta = f"{x['beta']:+6.2f}" if x.get("beta") is not None else f"{'n/a':>6s}"
            row = (f"{name:26s} {x['avg_exposure']:+8.2f} {beta} {x['passive_pnl']:+10,.2f} "
                   f"{x['timing_pnl']:+10,.2f} {x['selection_pnl']:+10,.2f} {-x['costs']:+9,.2f}")
            if len(windows) > 1:
                row += f" {x['window_start']:>26s} {100 * x['market_return']:+6.2f}%"
            print(row)
        print("passive = average start-of-bar exposure held throughout; timing = exposure changes vs the market;\n"
              "selection = asset choice before costs; costs = fees + spread/impact + funding")
    return 0


def cmd_experiment_run(cfg, args) -> int:
    from tradinglab.experiment import run

    if args.records_branch:
        cfg.experiment.records_branch = args.records_branch
    if args.duration_days is not None:
        cfg.experiment.duration_days = args.duration_days
    status = run(cfg, args.config, args.records, minutes=args.minutes, commit_every=args.commit_every)
    print(status)
    return 0


def cmd_simulate(cfg, args) -> int:
    import pandas as pd

    from tradinglab.simulate import simulate

    if args.end is None and args.days is None:
        raise SystemExit("give --end or --days")
    end = args.end or pd.Timestamp(args.start) + pd.Timedelta(days=args.days)
    res = simulate(cfg, args.start, end, Path(args.out), books=tuple(args.books),
                   allow_in_sample=args.allow_in_sample, spreads_path=args.spreads, retrain=args.walk_forward,
                   retrain_deep=args.retrain_deep)
    s = res["summary"].sort_values(["book", "return"], ascending=[True, False])
    print(s.to_string(index=False, formatters={"return": "{:+.2%}".format}, float_format="{:,.2f}".format))
    for note in res["meta"]["in_sample_notes"]:
        print("IN-SAMPLE:", note)
    print(json.dumps(res["meta"]["runs"]))
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
    p.add_argument("--no-maintenance", action="store_true", help="never retrain / re-run research (frozen models)")
    p.add_argument("--until", default=None, help="stop at this ISO UTC time")
    sub.add_parser("status")
    sub.add_parser("report")
    p = sub.add_parser("intraday-train", help="train + freeze the live intraday AI models")
    p.add_argument("--deep-epochs", type=int, default=None)
    p.add_argument("--kinds", default="lgbm,deep", help="which models to (re)train: lgbm,deep")
    p = sub.add_parser("intraday-run", help="real-time intraday paper trading daemon (5m bars)")
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--commit-every", type=float, default=None, help="commit state/ every N minutes")
    p.add_argument("--until", default=None, help="stop at this ISO UTC time")
    sub.add_parser("intraday-status")
    p = sub.add_parser("experiment-run", help="one shift of the unattended live experiment")
    p.add_argument("--records", required=True, help="git checkout of the records branch")
    p.add_argument("--minutes", type=float, default=330.0, help="length of this shift")
    p.add_argument("--commit-every", type=float, default=60.0, help="push a snapshot every N minutes")
    p.add_argument("--records-branch", default=None, help="override experiment.records_branch")
    p.add_argument("--duration-days", type=float, default=None, help="override experiment.duration_days")
    p = sub.add_parser("simulate", help="run the live engines on archived data (fast-forward paper trading)")
    p.add_argument("--start", required=True, help="UTC start, e.g. 2026-10-15T23:47")
    p.add_argument("--end", default=None, help="UTC end (or --days)")
    p.add_argument("--days", type=float, default=None, help="length in days")
    p.add_argument("--out", required=True, help="empty directory for the simulated state and summary")
    p.add_argument("--books", nargs="+", choices=["intraday", "daily"], default=["intraday", "daily"])
    p.add_argument("--spreads", default=None, help="half-spread file (default config/sim_spreads.json)")
    p.add_argument("--walk-forward", action="store_true",
                   help="retrain the models monthly on data before each step (for windows in the past)")
    p.add_argument("--retrain-deep", action="store_true",
                   help="with --walk-forward also retrain the deep models (~30 min per model per month); "
                        "otherwise they are left out")
    p.add_argument("--allow-in-sample", action="store_true",
                   help="run even if the models were trained on part of this window (results flagged)")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    handler = {"fetch": cmd_fetch, "research": cmd_research, "train-live": cmd_train_live,
               "backtest": cmd_backtest, "probe": cmd_probe, "cycle": cmd_cycle, "run": cmd_run,
               "status": cmd_status, "report": cmd_report, "intraday-train": cmd_intraday_train,
               "intraday-run": cmd_intraday_run, "intraday-status": cmd_intraday_status,
               "experiment-run": cmd_experiment_run, "simulate": cmd_simulate}[args.cmd]
    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
