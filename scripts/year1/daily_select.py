import json, logging
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.research.tournament import evaluate_candidate, select, ensemble_backtest
from tradinglab.backtest.metrics import summarize
from tradinglab.strategies.registry import by_name

logging.basicConfig(level=logging.WARNING)
T0 = pd.Timestamp("2025-10-29", tz="UTC")
cfg = load_config()
md = MarketData(cfg.paths.data_dir)
panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=T0,
                 exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
kept = ["tsmom", "tsmom_lf", "macd_lf", "breakout_lf"]
new = ["trend_ens_lf", "breakout_slow_lf", "xs_top_lf"]
bench = ["hold_btc", "basket_long"]
res = {n: evaluate_candidate(by_name(cfg, n), panel, cfg) for n in kept + new + bench}
chosen, alphas, table = select(res, cfg, "2021-01-01", T0)
pd.set_option("display.width", 250)
print(table[["sharpe", "psr", "dsr", "cluster", "eligible", "cluster_sharpe", "cluster_dsr", "selected"]].round(3).to_string())
print("chosen", chosen, alphas)
bt = ensemble_backtest(res, chosen, alphas, panel, cfg)
for a, b in [("2021-01-01", "2023-10-29"), ("2023-10-29", "2025-10-29"), ("2021-01-01", "2025-10-29")]:
    x = bt.returns[(bt.returns.index >= pd.Timestamp(a, tz="UTC")) & (bt.returns.index < T0) & (bt.returns.index < pd.Timestamp(b, tz="UTC"))]
    m = summarize(x, 365)
    print("main", a, b, "sharpe %.2f cagr %.2f mdd %.2f" % (m["sharpe"], m["cagr"], m["max_drawdown"]))
sel = {"created_at": pd.Timestamp.now(tz="UTC").isoformat(), "data_end": str(T0),
       "experiment": "year1: pruned + improved strategies, designed on data before data_end",
       "selection_window": ["2021-01-01", str(T0)],
       "strategies": chosen, "alphas": alphas, "candidates": bench + kept + new, "deep_models": {},
       "dev_sharpe_2021_2025": {n: round(float(table.loc[n, "sharpe"]), 3) for n in table.index}}
open("selection_year1.json", "w").write(json.dumps(sel, indent=2))
