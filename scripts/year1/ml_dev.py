"""Daily LightGBM strategy: walk-forward on data before T0; retraining interval and horizon chosen on 2021-01..2023-10."""
import copy, logging, sys, time
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.backtest.metrics import summarize
from tradinglab.research.tournament import evaluate_candidate
from tradinglab.strategies.ml import GBMForecast

logging.basicConfig(level=logging.WARNING)
T0 = pd.Timestamp("2025-10-29", tz="UTC")
cfg = load_config()
md = MarketData(cfg.paths.data_dir)
panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=T0,
                 exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
variants = [(5, True, 1), (5, True, 3), (5, True, 6), (5, True, 12), (5, False, 3), (21, True, 3)]
rets = {}
for h, lo, step in variants:
    c = copy.deepcopy(cfg)
    c.walkforward.step_months = step
    s = GBMForecast(h, lo)
    s.threads = 1
    t0 = time.time()
    r = evaluate_candidate(s, panel, c, workers=4)
    name = f"{s.name}_retrain{step}m"
    rets[name] = r.backtest.returns
    out = []
    for a, b in [("2021-01-01", "2023-10-29"), ("2023-10-29", "2025-10-29"), ("2021-01-01", "2025-10-29")]:
        x = r.backtest.returns[(r.backtest.returns.index >= pd.Timestamp(a, tz="UTC")) & (r.backtest.returns.index < pd.Timestamp(b, tz="UTC"))]
        m = summarize(x, 365)
        out.append((round(m["sharpe"], 2), round(m["cagr"], 2)))
    w = r.weights[r.weights.index >= "2021-01-01"]
    print(f"{name:26s} A {out[0]} B {out[1]} all {out[2]} mdd {summarize(r.backtest.returns[r.backtest.returns.index>='2021-01-01'], 365)['max_drawdown']:.2f} "
          f"turnover {w.diff().abs().sum(axis=1).mean()*365:.1f} net {w.sum(axis=1).mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(rets).to_csv(sys.argv[1])
