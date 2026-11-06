"""Judge Kronos's 24-hour forecasts (scripts/kronos/evaluate.py) against simple baselines; writes summary.json.

    python scripts/kronos/analyse.py [--dir reports/kronos]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tradinglab.config import load_config
from tradinglab.data.store import MarketData

COST = 2 * (5 + 1) * 1e-4          # taker fee and 1 bp beyond the quote, in and out
PERIODS = {"Jul-Oct 2025": ("2025-07-01", "2025-10-29"), "test year": ("2025-10-29", "2026-10-29")}


def load(d: Path) -> pd.DataFrame:
    p = d / "forecasts.parquet"
    f = pd.read_parquet(p if p.exists() else d / "forecasts.partial.parquet")
    f["origin"] = pd.to_datetime(f["origin"], utc=True)
    f["act24"] = f["act"].map(lambda a: float(a[-1]))
    f["q50_24"] = f["q50"].map(lambda a: float(a[-1]))
    f["q10_24"] = f["q10"].map(lambda a: float(a[-1]))
    f["q90_24"] = f["q90"].map(lambda a: float(a[-1]))
    return f


def binom_p(k: int, n: int) -> float:
    return float(stats.binomtest(k, n, 0.5).pvalue) if n else float("nan")


def hit(pred, act, days=None) -> dict:
    """Share of signs called right. With ``days``, also a p-value that treats each day as one observation (the
    12 coins move together, so their forecasts on one day are not independent tries)."""
    pred, act = np.asarray(pred, dtype=float), np.asarray(act, dtype=float)
    m = (np.sign(pred) != 0) & (np.sign(act) != 0)
    ok = np.sign(pred[m]) == np.sign(act[m])
    k, n = int(ok.sum()), int(m.sum())
    out = {"hit_rate": round(k / n, 4) if n else None, "n": n, "p_value_naive": round(binom_p(k, n), 4)}
    if days is not None and n:
        daily = pd.Series(ok).groupby(np.asarray(days)[m]).mean()
        t = (daily.mean() - 0.5) / (daily.std(ddof=1) / math.sqrt(len(daily)))
        out.update({"days": int(len(daily)), "t_by_day": round(float(t), 2),
                    "p_value": round(float(2 * stats.t.sf(abs(t), len(daily) - 1)), 4)})
    return out


def crps_samples(x: np.ndarray, y: float) -> float:
    x = np.sort(x)
    n = len(x)
    return float(np.abs(x - y).mean() - (2 * np.arange(1, n + 1) - n - 1).dot(x) / n ** 2)


def crps_normal(mu, sd, y) -> np.ndarray:
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))


def pinball(q, y, tau) -> np.ndarray:
    return np.maximum(tau * (y - q), (tau - 1) * (y - q))


def funding_24h(md, syms, origins) -> pd.DataFrame:
    """Funding paid by a long held over (t, t + 24 h] for each origin and coin (shorts receive it)."""
    out = {}
    for s in syms:
        r = md.funding_8h(s, pd.Timestamp(origins.min()).tz_localize(None) - pd.Timedelta(days=2))
        r.index = pd.DatetimeIndex(r.index).tz_convert("UTC") if r.index.tz else pd.DatetimeIndex(r.index).tz_localize("UTC")
        r.index = r.index.round("min")
        out[s] = [float(r[(r.index > t) & (r.index <= t + pd.Timedelta(hours=24))].sum()) for t in origins]
    return pd.DataFrame(out, index=origins)


def book(weights: pd.DataFrame, rets: pd.DataFrame, fund: pd.DataFrame) -> dict:
    """Daily rebalanced book: weights per origin and coin, held 24 h; costs on every position opened and closed."""
    w = weights.fillna(0.0)
    gross = (w * rets).sum(axis=1) - (w * fund).sum(axis=1) - w.abs().sum(axis=1) * COST
    eq = (1 + gross).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sd = gross.std()
    return {"total_return": round(float(eq.iloc[-1] - 1), 4), "sharpe": round(float(gross.mean() / sd * math.sqrt(365)), 2) if sd > 0 else None,
            "max_drawdown": round(dd, 4), "days": int(len(gross)), "avg_gross_exposure": round(float(w.abs().sum(axis=1).mean()), 2),
            "daily": gross}


def past_returns(md, f: pd.DataFrame, hours=(1, 4, 12, 24)) -> pd.DataFrame:
    """Log return over the ``h`` hours before each forecast (from the archive's hourly closes)."""
    from tradinglab import kronos as K

    out = pd.DataFrame(index=f.index)
    for s, g in f.groupby("symbol"):
        c = np.log(K.hourly(md, s, (g["origin"].min() - pd.Timedelta(days=3)).tz_localize(None))["close"])
        last = c.reindex(g["origin"] - pd.Timedelta(hours=1)).to_numpy()
        for h in hours:
            out.loc[g.index, f"past{h}"] = last - c.reindex(g["origin"] - pd.Timedelta(hours=1 + h)).to_numpy()
    return out


def analyse(f: pd.DataFrame, md, syms) -> dict:
    out = {"forecasts": int(len(f)), "days": int(f["origin"].nunique()), "coins": int(f["symbol"].nunique()),
           "first": str(f["origin"].min().date()), "last": str(f["origin"].max().date())}
    a, q50, mean = f["act24"].to_numpy(), f["q50_24"].to_numpy(), f["mean24"].to_numpy()
    day = f["origin"].to_numpy()
    past = past_returns(md, f)

    # direction
    out["direction_24h"] = {"kronos_median": hit(q50, a, day), "kronos_mean": hit(mean, a, day),
                            "momentum_24h": hit(f["ctx_ret24"].to_numpy(), a, day), "reversal_24h": hit(-f["ctx_ret24"].to_numpy(), a, day),
                            "always_up": hit(np.ones_like(a), a, day)}
    at = lambda col, h: f[col].map(lambda x: float(x[h - 1])).to_numpy()                 # noqa: E731
    out["direction_by_hour"] = {h: hit(at("q50", h), at("act", h), day)["hit_rate"] for h in (1, 4, 12, 24)}
    out["direction_by_hour_detail"] = {h: {"kronos": hit(at("q50", h), at("act", h), day),
                                           "reversal_same_span": hit(-past[f"past{h}"].to_numpy(), at("act", h), day),
                                           "momentum_same_span": hit(past[f"past{h}"].to_numpy(), at("act", h), day),
                                           "agree_with_reversal": round(float((np.sign(at("q50", h)) == -np.sign(past[f"past{h}"])).mean()), 3)}
                                       for h in (1, 4, 12, 24)}
    # correlation with what happened
    ic_daily = f.groupby("origin").apply(lambda g: stats.spearmanr(g["mean24"], g["act24"])[0] if len(g) > 3 else np.nan,
                                         include_groups=False).dropna()
    out["correlation"] = {"pearson_mean_vs_actual": round(float(np.corrcoef(mean, a)[0, 1]), 4),
                          "spearman_mean_vs_actual": round(float(stats.spearmanr(mean, a)[0]), 4),
                          "daily_rank_ic_mean": round(float(ic_daily.mean()), 4),
                          "daily_rank_ic_t": round(float(ic_daily.mean() / ic_daily.std() * math.sqrt(len(ic_daily))), 2)}
    # point accuracy of the median against "no change"
    out["point_24h"] = {"mae_kronos": round(float(np.abs(q50 - a).mean()), 5), "mae_no_change": round(float(np.abs(a).mean()), 5),
                        "skill_vs_no_change": round(float(1 - np.abs(q50 - a).mean() / np.abs(a).mean()), 4),
                        "mean_forecast_return": round(float(mean.mean()), 5), "mean_actual_return": round(float(a.mean()), 5)}
    # bands: coverage, width, scores; baseline = normal band from recent hourly volatility (EWMA, 3 days)
    cov, base_cov, width, base_width = {}, {}, {}, {}
    z = stats.norm.ppf(0.9)
    for h in (1, 4, 12, 24):
        act_h = f["act"].map(lambda x: float(x[h - 1])).to_numpy()
        lo, hi = f["q10"].map(lambda x: float(x[h - 1])).to_numpy(), f["q90"].map(lambda x: float(x[h - 1])).to_numpy()
        sd = f["ctx_vol_ewm"].to_numpy() * math.sqrt(h)
        cov[h] = round(float(((act_h >= lo) & (act_h <= hi)).mean()), 4)
        base_cov[h] = round(float((np.abs(act_h) <= z * sd).mean()), 4)
        width[h] = round(float(np.median(hi - lo)), 5)
        base_width[h] = round(float(np.median(2 * z * sd)), 5)
    sd24 = f["ctx_vol_ewm"].to_numpy() * math.sqrt(24)
    crps_k = np.array([crps_samples(s, y) for s, y in zip(f["ret24_samples"], a)])
    crps_b = crps_normal(0.0, sd24, a)
    pin_k = np.mean([pinball(f[c].to_numpy(), a, t).mean() for c, t in (("q10_24", .1), ("q50_24", .5), ("q90_24", .9))])
    pin_b = np.mean([pinball(z_ * sd24, a, t).mean() for z_, t in ((-z, .1), (0.0, .5), (z, .9))])
    pit = np.array([(s < y).mean() for s, y in zip(f["ret24_samples"], a)])
    out["bands"] = {"inside_p10_p90_kronos": cov, "inside_p10_p90_baseline": base_cov, "target": 0.8,
                    "median_width_kronos": width, "median_width_baseline": base_width,
                    "crps_24h_kronos": round(float(crps_k.mean()), 6), "crps_24h_baseline": round(float(crps_b.mean()), 6),
                    "crps_skill_vs_baseline": round(float(1 - crps_k.mean() / crps_b.mean()), 4),
                    "pinball_24h_kronos": round(float(pin_k), 6), "pinball_24h_baseline": round(float(pin_b), 6),
                    "pit_deciles": np.histogram(pit, bins=10, range=(0, 1))[0].tolist()}
    # volatility: does the forecast's spread tell how much the next 24 hours will move?
    lv, lp, lb = np.log(f["act_vol"]), np.log(f["pred_vol"]), np.log(f["ctx_vol_ewm"])
    X = np.column_stack([np.ones(len(f)), lb, lp])
    beta = np.linalg.lstsq(X, lv, rcond=None)[0]
    resid = lv - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
    out["volatility"] = {"corr_kronos": round(float(np.corrcoef(lp, lv)[0, 1]), 4), "corr_baseline_ewm": round(float(np.corrcoef(lb, lv)[0, 1]), 4),
                         "corr_last_24h": round(float(np.corrcoef(np.log(f["ctx_vol24"]), lv)[0, 1]), 4),
                         "kronos_beta_given_baseline": round(float(beta[2]), 3), "kronos_t_given_baseline": round(float(beta[2] / se[2]), 1),
                         "median_ratio_predicted_to_actual": round(float(np.median(f["pred_vol"] / f["act_vol"])), 3)}

    # trading: daily at 00:00 UTC, hold 24 h, costs and funding
    piv = lambda c: f.pivot(index="origin", columns="symbol", values=c)            # noqa: E731
    rets = np.exp(piv("act24")) - 1
    origins = rets.index
    fund = funding_24h(md, list(rets.columns), origins).reindex(columns=rets.columns).fillna(0.0)
    m50, mn, mom = piv("q50_24"), piv("mean24"), piv("ctx_ret24")
    n = rets.notna().sum(axis=1)
    rk = mn.rank(axis=1)
    books = {
        "long_short_by_forecast_sign": np.sign(m50).div(n, axis=0),
        "long_only_when_forecast_up": (m50 > 0).astype(float).div(n, axis=0),
        "top3_long_bottom3_short": ((rk > n.values[:, None] - 3).astype(float) - (rk <= 3).astype(float)) / 6,
        "confident_only_mean_beyond_half_band": (np.sign(mn) * ((mn.abs() > 0.5 * (piv("q90_24") - piv("q10_24")) / 2))).div(n, axis=0),
        "baseline_momentum_sign": np.sign(mom).div(n, axis=0),
        "baseline_reversal_sign": (-np.sign(mom)).div(n, axis=0),
        "buy_and_hold_basket": rets.notna().astype(float).div(n, axis=0),
    }
    r4 = np.exp(f.assign(a4=at("act", 4)).pivot(index="origin", columns="symbol", values="a4")) - 1
    q4 = f.assign(k4=at("q50", 4)).pivot(index="origin", columns="symbol", values="k4")
    p4 = f.assign(p4=past["past4"]).pivot(index="origin", columns="symbol", values="p4")
    no_fund = fund * 0.0                                   # 4-hour holds from 00:00 cross no settlement
    res, daily = {}, {}
    for k, w, rr in (("kronos_4h_long_short", np.sign(q4).div(n, axis=0), r4),
                     ("reversal_4h_long_short", (-np.sign(p4)).div(n, axis=0), r4)):
        b = book(w.reindex_like(rr), rr.fillna(0.0), no_fund.reindex_like(rr).fillna(0.0))
        daily[k] = b.pop("daily")
        res[k] = b
        res[k]["by_period"] = {p: round(float((1 + daily[k][(daily[k].index >= s) & (daily[k].index < e)]).prod() - 1), 4)
                               for p, (s, e) in PERIODS.items()}
    for k, w in books.items():
        b = book(w.reindex_like(rets), rets.fillna(0.0), fund)
        daily[k] = b.pop("daily")
        res[k] = b
        res[k]["by_period"] = {p: round(float((1 + daily[k][(daily[k].index >= s) & (daily[k].index < e)]).prod() - 1), 4)
                               for p, (s, e) in PERIODS.items()}
    out["trading"] = res
    out["by_period_direction"] = {p: hit(q50[(f["origin"] >= s) & (f["origin"] < e)], a[(f["origin"] >= s) & (f["origin"] < e)],
                                         day[(f["origin"] >= s) & (f["origin"] < e)])
                                  for p, (s, e) in PERIODS.items()}
    out["by_coin_direction"] = {s: hit(g["q50_24"].to_numpy(), g["act24"].to_numpy())["hit_rate"] for s, g in f.groupby("symbol")}
    return out, pd.DataFrame(daily)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/kronos")
    d = Path(ap.parse_args().dir)
    cfg = load_config()
    f = load(d)
    res, daily = analyse(f, MarketData(cfg.paths.data_dir), cfg.intraday.symbols)
    (d / "summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (1 + daily).cumprod().to_csv(d / "books_equity.csv")
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_coin_direction",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
