"""Is 'keep what made money, prune what lost' a good rule? Apply it at each year start, measure the next year."""
import copy, logging
import numpy as np
import pandas as pd
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
from tradinglab.research.tournament import evaluate_candidate
from tradinglab.strategies.classic import classic_candidates

logging.basicConfig(level=logging.WARNING)
T1 = pd.Timestamp("2026-10-29", tz="UTC")
cfg = load_config(); cfg.walkforward.first_test = "2019-01-01"
md = MarketData(cfg.paths.data_dir)
panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=T1,
                 exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
bench = {"hold_btc", "basket_long"}
strats = [s for s in classic_candidates(cfg.universe.symbols) if s.name not in bench]
R = pd.DataFrame({s.name: evaluate_candidate(s, panel, cfg).backtest.returns for s in strats})
R = R[R.index >= "2019-01-01"]
B = evaluate_candidate([s for s in classic_candidates([]) if s.name == "basket_long"][0], panel, cfg).backtest.returns


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan


rows = []
origins = [pd.Timestamp(f"{y}-01-01", tz="UTC") for y in range(2020, 2026)] + [pd.Timestamp("2025-10-29", tz="UTC")]
for o in origins:
    past2 = R[(R.index >= o - pd.Timedelta(days=730)) & (R.index < o)]
    past1 = R[(R.index >= o - pd.Timedelta(days=365)) & (R.index < o)]
    keep = [c for c in R if sharpe(past2[c]) >= 0.5 and (1 + past1[c]).prod() > 1]
    drop = [c for c in R if c not in keep]
    nxt = R[(R.index >= o) & (R.index < o + pd.Timedelta(days=365))]
    yr = (1 + nxt).prod() - 1
    rows.append({"origin": str(o.date()), "kept": len(keep), "pruned": len(drop),
                 "kept_avg": yr[keep].mean() if keep else np.nan, "pruned_avg": yr[drop].mean() if drop else np.nan,
                 "all_avg": yr.mean(), "kept_beat_pruned": (yr[keep].mean() > yr[drop].mean()) if keep and drop else None,
                 "basket": (1 + B[(B.index >= o) & (B.index < o + pd.Timedelta(days=365))]).prod() - 1,
                 "kept_names": ",".join(keep)})
t = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 120)
print(t.round(3).to_string(index=False))
import sys
if len(sys.argv) > 1:
    t.to_csv(sys.argv[1], index=False)
