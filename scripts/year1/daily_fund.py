"""Funding filter with the funding a live run sees (Binance archive, published monthly)."""
import logging
import numpy as np
import pandas as pd
from tradinglab.config import load_config, bars_per_year
from tradinglab.data.store import MarketData, _funding_on_grid
from tradinglab.backtest.metrics import summarize
from tradinglab.research.tournament import evaluate_candidate
from tradinglab.strategies.base import Strategy
from tradinglab.strategies.classic import TSMOM, MACDTrend, Breakout, _available

logging.basicConfig(level=logging.WARNING)
T0 = pd.Timestamp("2025-10-29", tz="UTC")
cfg = load_config()
md = MarketData(cfg.paths.data_dir)
panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=T0,
                 exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
rates = {s: md.funding_8h(s, pd.Timestamp(cfg.walkforward.data_start)) for s in panel.symbols}
grid = panel.close.index
lag = pd.Timedelta(hours=cfg.market.exec_lag_hours)
seen = pd.DataFrame(np.nan, index=grid, columns=panel.symbols)
months = pd.date_range(grid[0].floor("D") - pd.offsets.MonthBegin(1), T0, freq="MS", tz="UTC")
for m0 in months:
    rows = (grid >= m0) & (grid < m0 + pd.offsets.MonthBegin(1))
    if not rows.any():
        continue
    hist = grid[grid < m0 + pd.offsets.MonthBegin(1)]
    for s in panel.symbols:
        r = rates[s]
        cut = r[r.index < m0] if len(r) else r
        f = _funding_on_grid(cut, hist, "1d", lag, cfg.market.default_funding_8h)
        f7 = f.shift(2).rolling(7, min_periods=3).mean()
        seen.loc[rows, s] = f7.reindex(grid[rows]).to_numpy()
true7 = panel.funding.shift(2).rolling(7, min_periods=3).mean()
print("corr seen vs true (stacked):", np.corrcoef(seen.stack().dropna().align(true7.stack().dropna(), join="inner")[0],
                                                 true7.stack().dropna().align(seen.stack().dropna(), join="inner")[0])[0, 1])


class Ens(Strategy):
    name = "trend_ens_lf"
    def signals(self, p):
        parts = [TSMOM(long_only=True), MACDTrend(long_only=True), Breakout(long_only=True)]
        return sum(s.signals(p).fillna(0.0) for s in parts).div(3).where(_available(p))


class Filt(Strategy):
    def __init__(self, name, f, th):
        self.name, self.f, self.th = name, f, th
    def signals(self, p):
        sig = Ens().signals(p)
        return sig.where(~(self.f.reindex(sig.index) > self.th), 0.0).where(_available(p))


bpy = bars_per_year("1d")
for s in [Ens(), Filt("fund_true", true7, 0.0009), Filt("fund_seen", seen, 0.0009),
          Filt("fund_seen_6bp", seen, 0.0006), Filt("fund_seen_12bp", seen, 0.0012)]:
    ret = evaluate_candidate(s, panel, cfg).backtest.returns
    out = []
    for a, b in [("2021-01-01", "2023-10-29"), ("2023-10-29", "2025-10-29"), ("2021-01-01", "2025-10-29")]:
        x = ret[(ret.index >= pd.Timestamp(a, tz="UTC")) & (ret.index < pd.Timestamp(b, tz="UTC"))]
        out.append(round(summarize(x, bpy)["sharpe"], 2))
    print(f"{s.name:16s} sharpe A/B/all {out}")
print("share of coin-days filtered (seen, 9bp):", float((seen > 0.0009).sum().sum() / seen.notna().sum().sum()))
