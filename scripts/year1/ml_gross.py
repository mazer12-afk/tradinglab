"""Daily LightGBM in the test year: monthly retraining from the start, return after and before costs."""
import copy, json, logging, sys
import pandas as pd
from tradinglab.config import load_config, CostConfig
from tradinglab.data.store import MarketData
from tradinglab.backtest.engine import run_backtest
from tradinglab.research.tournament import evaluate_candidate
from tradinglab.strategies.ml import GBMForecast

logging.basicConfig(level=logging.WARNING)
T0, T1 = pd.Timestamp("2025-10-29", tz="UTC"), pd.Timestamp("2026-10-29", tz="UTC")
cfg = load_config()
md = MarketData(cfg.paths.data_dir)
panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=T1,
                 exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
c = copy.deepcopy(cfg); c.walkforward.first_test = str(T0.date()); c.walkforward.step_months = 1
g = GBMForecast(5, True); g.threads = 1
r = evaluate_candidate(g, panel, c, workers=4)
free = run_backtest(r.weights, panel, CostConfig(0, 0), band=cfg.portfolio.rebalance_band, start=str(T0.date()))
nofund = panel.__class__(panel.close, panel.exec_price, panel.volume, panel.funding * 0, panel.bar, panel.exec_lag_hours)
free_nf = run_backtest(r.weights, nofund, CostConfig(0, 0), band=cfg.portfolio.rebalance_band, start=str(T0.date()))
f = lambda x: float((1 + x[(x.index >= T0) & (x.index < T1)]).prod() - 1)
w = r.weights[(r.weights.index >= T0) & (r.weights.index < T1)]
out = {"return": f(r.backtest.returns), "before_fees_spread": f(free.returns), "before_all_costs": f(free_nf.returns),
       "turnover": float(w.diff().abs().sum(axis=1).sum()), "net_exposure": float(w.sum(axis=1).mean())}
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
