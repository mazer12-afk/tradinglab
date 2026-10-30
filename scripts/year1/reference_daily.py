"""Test-year reference (after the roster was frozen): backtests of every daily strategy, kept or pruned."""
import copy, logging, sys
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.backtest.metrics import summarize
from tradinglab.research.tournament import evaluate_candidate, ensemble_backtest
from tradinglab.strategies.registry import by_name
from tradinglab.strategies.ml import GBMForecast
import json

logging.basicConfig(level=logging.WARNING)
T0, T1 = pd.Timestamp("2025-10-29", tz="UTC"), pd.Timestamp("2026-10-29", tz="UTC")
cfg = load_config()
md = MarketData(cfg.paths.data_dir)
panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=T1,
                 exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
names = ["hold_btc", "basket_long", "tsmom", "tsmom_lf", "macd_lf", "breakout_lf", "trend_ens_lf",
         "breakout_slow_lf", "xs_top_lf", "macd", "breakout", "xsmom_63", "xs_rev_5"]
res = {n: evaluate_candidate(by_name(cfg, n), panel, cfg) for n in names}
c = copy.deepcopy(cfg); c.walkforward.first_test = str(T0.date()); c.walkforward.step_months = 1
g = GBMForecast(5, True); g.threads = 1
res["ml_gbm5_lf"] = evaluate_candidate(g, panel, c, workers=4)
sel = json.load(open("models/year1/selection.json"))
main = ensemble_backtest(res, sel["strategies"], sel["alphas"], panel, cfg, start=str(T0.date()))
rets = {n: r.backtest.returns for n, r in res.items()}
rets["main"] = main.returns
oos = pd.read_csv("reports/research/oos_returns.csv", index_col=0, parse_dates=True)
for n in ("dl_lstm", "dl_mom_trans", "dl_deepm"):
    rets[n] = oos[n]
rows = {}
for n, r in rets.items():
    x = r[(r.index >= T0) & (r.index < T1)].dropna()
    m = summarize(x, 365)
    rows[n] = {"return": (1 + x).prod() - 1, "sharpe": m["sharpe"], "mdd": m["max_drawdown"], "days": len(x)}
t = pd.DataFrame(rows).T.sort_values("return", ascending=False)
print(t.round(3).to_string())
t.to_csv(sys.argv[1])
