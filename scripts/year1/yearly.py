"""Calendar-year returns of every daily strategy (research backtester, net of costs) and of collecting funding."""
import logging, sys
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
R = pd.DataFrame({s.name: evaluate_candidate(s, panel, cfg).backtest.returns for s in classic_candidates(cfg.universe.symbols)})
oos = pd.read_csv("reports/research/oos_returns.csv", index_col=0, parse_dates=True)
for n in ("dl_lstm", "dl_mom_trans", "dl_deepm"):
    R[n] = oos[n]
R = R[(R.index >= "2019-01-01") & (R.index < T1)]
Y = R.groupby(R.index.year).agg(lambda x: (1 + x.dropna()).prod() - 1 if x.notna().sum() > 30 else np.nan)
Y.index = [str(y) + ("*" if y == 2026 else "") for y in Y.index]
pos = (Y > 0).sum(); n = Y.notna().sum()
if len(sys.argv) > 1:
    Y.T.round(4).assign(positive_years=pos, years=n).to_csv(sys.argv[1])
pd.set_option("display.width", 250)
print((Y.T * 100).round(0).assign(**{"years +": pos.astype(str) + "/" + n.astype(str)}).sort_values("years +", ascending=False).to_string())

# funding: what a short perpetual (hedged with spot) collects per year, before costs
rates = {s: md.funding_8h(s, pd.Timestamp("2019-01-01")) for s in cfg.universe.symbols}
F = pd.DataFrame({s: r for s, r in rates.items() if len(r)})
F = F[F.index < T1]
for name, cols in [("BTC", ["BTCUSDT"]), ("BTC+ETH", ["BTCUSDT", "ETHUSDT"]), ("30 coins", list(F.columns))]:
    f = F[cols].mean(axis=1)
    yr = f.groupby(f.index.year).sum()
    mo = f.groupby([f.index.year, f.index.month]).sum()
    print(f"funding collected by a short perp, {name}: " + ", ".join(f"{y}: {100*v:+.1f}%" for y, v in yr.items()),
          f"| negative months {int((mo < 0).sum())}/{len(mo)} | last print {f.dropna().index[-1].date()}")
