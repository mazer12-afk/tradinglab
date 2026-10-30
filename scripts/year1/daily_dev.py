"""Daily book development on data before the test year (T0 = 2025-10-29): prune + improve + new strategies."""
import json, logging, sys
import numpy as np
import pandas as pd
from tradinglab.config import load_config, bars_per_year
from tradinglab.data.store import MarketData, Panel
from tradinglab.backtest.metrics import summarize
from tradinglab.research.tournament import evaluate_candidate, select
from tradinglab.strategies.base import Strategy
from tradinglab.strategies.classic import (TSMOM, MACDTrend, Breakout, CrossSectionalMomentum, BuyAndHold,
                                           EqualWeightLong, _available)
from tradinglab.features import ewm_vol

logging.basicConfig(level=logging.WARNING)
T0 = pd.Timestamp("2025-10-29", tz="UTC")
A0, A1, B1 = "2021-01-01", "2023-10-29", "2025-10-29"
cfg = load_config()
md = MarketData(cfg.paths.data_dir)
panel = md.panel(cfg.universe.symbols, cfg.market.bar, cfg.walkforward.data_start, end=T0,
                 exec_lag_hours=cfg.market.exec_lag_hours, default_funding_8h=cfg.market.default_funding_8h)
print("panel", panel.close.index[0], panel.close.index[-1], panel.close.shape, flush=True)


class Ensemble(Strategy):
    def __init__(self, name, parts):
        self.name, self.parts = name, parts

    def signals(self, p):
        return sum(s.signals(p).fillna(0.0) for s in self.parts).div(len(self.parts)).where(_available(p))


class MarketRegime(Strategy):
    """Inner strategy, switched off while the equal-weight market's blended trend is negative."""
    def __init__(self, name, inner, lookbacks=(21, 63, 126, 252)):
        self.name, self.inner, self.lookbacks = name, inner, lookbacks

    def signals(self, p):
        r = p.close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
        idx = (1 + r).cumprod()
        regime = sum(np.sign(idx.pct_change(h)) for h in self.lookbacks) / len(self.lookbacks)
        on = (regime > 0).astype(float)
        return self.inner.signals(p).mul(on, axis=0)


class TopRotation(Strategy):
    """Long-only rotation: the top third of coins by vol-adjusted momentum, among coins in an uptrend."""
    def __init__(self, name="xs_top_lf", lookback=63, frac=1/3):
        self.name, self.lookback, self.frac = name, lookback, frac

    def signals(self, p):
        close = p.close
        vol = ewm_vol(close.pct_change(fill_method=None))
        score = close.pct_change(self.lookback, fill_method=None) / (vol * np.sqrt(self.lookback))
        up = TSMOM(long_only=True).signals(p) > 0
        score = score.where(up)
        rank = score.rank(axis=1, ascending=False)
        n = close.notna().sum(axis=1)
        top = rank.le((n * self.frac).clip(lower=1), axis=0)
        return top.astype(float).where(_available(p))


class FundingFilter(Strategy):
    """Inner strategy without coins whose trailing 7-day funding is > 3x the 1 bp/8h baseline (crowded longs)."""
    def __init__(self, name, inner, thresh=0.0009):
        self.name, self.inner, self.thresh = name, inner, thresh

    def signals(self, p):
        f = p.funding.shift(2).rolling(7, min_periods=3).mean()   # funding already paid (no look-ahead)
        sig = self.inner.signals(p)
        return sig.where(~(f > self.thresh), 0.0).where(_available(p))


ens = Ensemble("trend_ens_lf", [TSMOM(long_only=True), MACDTrend(long_only=True), Breakout(long_only=True)])
cands = [
    BuyAndHold("BTCUSDT"), EqualWeightLong(), TSMOM(), TSMOM(long_only=True), MACDTrend(), MACDTrend(long_only=True),
    Breakout(), Breakout(long_only=True), CrossSectionalMomentum(63), CrossSectionalMomentum(5, reverse=True),
    ens,
    MarketRegime("trend_regime_lf", ens),
    TSMOM((63, 126, 252), long_only=True, name="tsmom_slow_lf"),
    Breakout((55, 100, 200), long_only=True, name="breakout_slow_lf"),
    TopRotation(),
    FundingFilter("trend_fund_lf", ens),
    MarketRegime("basket_regime", EqualWeightLong()),
]
bpy = bars_per_year(cfg.market.bar)
res, rows = {}, {}
for s in cands:
    r = evaluate_candidate(s, panel, cfg)
    res[s.name] = r
    ret = r.backtest.returns
    row = {}
    for lab, a, b in [("A 21-23", A0, A1), ("B 23-25", A1, B1), ("all", A0, B1), ("last1y", "2024-10-29", B1)]:
        x = ret[(ret.index >= pd.Timestamp(a, tz="UTC")) & (ret.index < pd.Timestamp(b, tz="UTC"))]
        m = summarize(x, bpy)
        row[(lab, "sharpe")] = m["sharpe"]; row[(lab, "cagr")] = m["cagr"]
        if lab == "all":
            row[(lab, "mdd")] = m["max_drawdown"]
    w = r.weights[r.weights.index >= A0]
    row[("all", "turnover")] = float(w.diff().abs().sum(axis=1).mean() * 365)
    row[("all", "net_exp")] = float(w.sum(axis=1).mean())
    rows[s.name] = row
t = pd.DataFrame(rows).T
t.columns = pd.MultiIndex.from_tuples(t.columns)
pd.set_option("display.width", 250)
print(t.round(2).to_string())
rets = pd.DataFrame({n: r.backtest.returns for n, r in res.items()})
rets.to_csv(sys.argv[1] if len(sys.argv) > 1 else "daily_dev_returns.csv")
print(rets[rets.index >= A0][["tsmom_lf", "macd_lf", "breakout_lf", "trend_ens_lf", "trend_regime_lf", "xs_top_lf", "basket_long"]].corr().round(2))
