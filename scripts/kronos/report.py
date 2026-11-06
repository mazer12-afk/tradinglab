"""Build reports/kronos/report.html from the evaluation (summary.json, forecasts, books) and the live forecast.

    python scripts/kronos/report.py [--dir reports/kronos]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts as C  # noqa: E402

CSS = (Path(__file__).resolve().parents[2] / "reports" / "daytrade" / "report.html").read_text()
CSS = CSS[CSS.index("<style>"):CSS.index("</style>") + len("</style>")]
EXTRA = """<style>
.fan-zone { fill: var(--accent-wash); opacity: .55; }
.wick { stroke-width: 1; } .wick.up, .body.up { stroke: var(--s1); fill: var(--s1); } .wick.down, .body.down { stroke: var(--neg); fill: var(--neg); }
.fan-path { fill: none; stroke: var(--s1); stroke-width: 1; opacity: .16; }
.fan-band { fill: var(--s1); opacity: .10; }
.fan-q { fill: none; stroke: var(--s1); stroke-width: 1.3; stroke-dasharray: 4 3; }
.fan-med { fill: none; stroke: var(--s2); stroke-width: 2.2; }
.fan-act { fill: none; stroke: var(--ink); stroke-width: 2; }
.fan-now { fill: var(--s2); }
.fan-lab { font: 11px var(--f-mono); fill: var(--ink-2); } .fan-lab.med { fill: var(--s2); }
.viz-refline { stroke: var(--ink-2); stroke-width: 1.2; stroke-dasharray: 5 4; }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.verdict-head { font: 600 13px/1.4 var(--f-mono); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }
</style>"""


def pct(x, d=0, sign=False) -> str:
    return f"{x:+.{d}%}" if sign else f"{x:.{d}%}"


def build(d: Path, copy: dict) -> str:
    s = json.loads((d / "summary.json").read_text())
    eq = pd.read_csv(d / "books_equity.csv", index_col=0, parse_dates=True)
    f = pd.read_parquet(d / "forecasts.parquet")
    f["origin"] = pd.to_datetime(f["origin"], utc=True)
    live = json.loads((d / "live_BTCUSDT.json").read_text())
    b = s["bands"]
    dr = s["direction_24h"]
    tr = s["trading"]

    # a historical example: BTC on the day with the biggest 24-hour move, its fan against what happened
    from tradinglab import kronos as K
    from tradinglab.config import load_config
    from tradinglab.data.store import MarketData

    btc = f[f["symbol"] == "BTCUSDT"].copy()
    btc["abs"] = btc["act"].map(lambda x: abs(float(x[-1])))
    ex = btc.sort_values("abs").iloc[-1]
    hh = K.hourly(MarketData(load_config().paths.data_dir), "BTCUSDT", (ex["origin"] - pd.Timedelta(days=4)).tz_localize(None))
    hh = hh[(hh.index >= ex["origin"] - pd.Timedelta(hours=72)) & (hh.index < ex["origin"])]
    q = lambda k: (np.exp(np.asarray(ex[k], dtype=float)) * ex["last"]).tolist()                 # noqa: E731
    ex_chart = C.fan({"t": [str(t) for t in hh.index], **{k: hh[k].tolist() for k in ("open", "high", "low", "close")}},
                     [str(t) for t in pd.date_range(ex["origin"], periods=24, freq="1h")], [], q("q10"), q("q50"), q("q90"),
                     float(ex["last"]), actual=q("act"))

    ar = autoresearch_section(copy)
    fan = C.fan(live["history"], live["future_t"], live["path_close"], live["p10"], live["p50"], live["p90"], live["last_close"])
    hz = ["1", "4", "12", "24"]
    cov = C.bars([f"{h} h ahead" for h in hz],
                 {"Kronos": [b["inside_p10_p90_kronos"][h] for h in hz], "Volatility band": [b["inside_p10_p90_baseline"][h] for h in hz]},
                 {"Kronos": "--s1", "Volatility band": "--s4"}, ref=0.8, ref_label="80 %: the target", hi=1.05, W=460, H=260, ref_left=True)
    pit = [x / sum(b["pit_deciles"]) for x in b["pit_deciles"]]
    pit_chart = C.bars([f"{10 * i}" for i in range(10)], {"share": pit}, {"share": "--s1"}, ref=0.1,
                       ref_label="10 % each if right", fmt="{:.0%}", H=260, W=460)
    names = {"kronos_median": "Kronos (median path)", "momentum_24h": "Last 24 h continues", "reversal_24h": "Last 24 h reverses",
             "always_up": "Always up"}
    dir_chart = C.bars([names[k] for k in names], {"hit": [dr[k]["hit_rate"] for k in names]}, {"hit": "--s1"},
                       ref=0.5, ref_label="50 %: a coin toss", lo=min(0.4, min(dr[k]["hit_rate"] for k in names) - 0.05),
                       hi=max(0.6, max(dr[k]["hit_rate"] for k in names) + 0.05), H=230)
    book_names = {"long_short_by_forecast_sign": "Kronos long/short", "long_only_when_forecast_up": "Kronos long-only",
                  "top3_long_bottom3_short": "Kronos top 3 vs bottom 3", "confident_only_mean_beyond_half_band": "Kronos confident only",
                  "baseline_momentum_sign": "Last day continues (no model)", "buy_and_hold_basket": "Buy and hold"}
    table_names = {**book_names, "kronos_4h_long_short": "Kronos long/short, 4-hour hold",
                   "reversal_4h_long_short": "Last 4 h reverses, 4-hour hold (no model)"}
    cols = {"long_short_by_forecast_sign": "--s1", "long_only_when_forecast_up": "--s3", "top3_long_bottom3_short": "--s5",
            "confident_only_mean_beyond_half_band": "--s2", "baseline_momentum_sign": "--base", "buy_and_hold_basket": "--muted"}
    eq_chart = C.lines(eq[list(book_names)], cols, labels=book_names)

    rows = []
    for k, n in table_names.items():
        r = tr[k]
        bp = r["by_period"]
        cls = lambda v: "pos" if v > 0 else "neg"                       # noqa: E731
        rows.append(f'<tr{" class=hl" if k == "long_short_by_forecast_sign" else ""}><td>{n}</td>'
                    f'<td class="{cls(r["total_return"])}">{pct(r["total_return"], 1, True)}</td><td>{r["sharpe"]}</td>'
                    f'<td>{pct(r["max_drawdown"], 1)}</td><td>{r["avg_gross_exposure"]:.2f}</td>'
                    + "".join(f'<td class="{cls(v)}">{pct(v, 1, True)}</td>' for v in bp.values()) + "</tr>")
    periods = list(next(iter(tr.values()))["by_period"])
    v = s["volatility"]
    c = s["correlation"]
    pt = s["point_24h"]
    bh = {h: s["direction_by_hour"][h] for h in s["direction_by_hour"]}
    hit = dr["kronos_median"]

    page = f"""<title>Kronos Forecast Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
{CSS}
{EXTRA}
<div class="wrap">
<header>
  <div class="eyebrow">Foundation-model test · Kronos-small (24.7 M parameters) · 12 crypto perpetuals · 1-hour candles · {s['first']} → {s['last']}</div>
  <h1>{copy['h1']}</h1>
  <p class="lede">{copy['lede']}</p>
</header>

<section aria-labelledby="verdict-h">
  <div class="sec-head"><div class="eyebrow">Verdict</div><h2 id="verdict-h">{copy['verdict_h']}</h2></div>
  <div class="verdict">{''.join(f'<p{" class=big" if i == 0 else ""}>{p}</p>' for i, p in enumerate(copy['verdict']))}</div>
  <div class="tiles">
    <div class="tile"><div class="k">Up or down over 24 h, called right</div><div class="v">{pct(hit['hit_rate'], 1)}</div><div class="s">{hit['n']:,} forecasts · a coin toss is 50 % · always-up got {pct(dr['always_up']['hit_rate'], 1)}</div></div>
    <div class="tile"><div class="k">Actual price inside the p10–p90 fan, 24 h ahead</div><div class="v neg">{pct(b['inside_p10_p90_kronos']['24'])}</div><div class="s">should be 80 % · simple volatility band: {pct(b['inside_p10_p90_baseline']['24'])}</div></div>
    <div class="tile"><div class="k">Forecast error vs "no change"</div><div class="v {'neg' if pt['skill_vs_no_change'] < 0 else 'pos'}">{pct(-pt['skill_vs_no_change'], 0, True)}</div><div class="s">typical miss {pct(pt['mae_kronos'], 2)} vs {pct(pt['mae_no_change'], 2)} for "price stays put"</div></div>
    <div class="tile"><div class="k">Trading its direction, after costs</div><div class="v {'neg' if tr['long_short_by_forecast_sign']['total_return'] < 0 else 'pos'}">{pct(tr['long_short_by_forecast_sign']['total_return'], 1, True)}</div><div class="s">long/short all 12 coins daily · buy and hold {pct(tr['buy_and_hold_basket']['total_return'], 1, True)}</div></div>
  </div>
</section>

{ar}
<section aria-labelledby="now-h">
  <div class="sec-head"><div class="eyebrow">Now · the picture from the post</div><h2 id="now-h">What Kronos says about Bitcoin for the next 24 hours</h2>
  <p class="lede">{copy['now']}</p></div>
  <div class="fig">
    <div class="fig-title">BTCUSDT perpetual, 1-hour candles · {len(live['path_close'])} sampled paths from {pd.Timestamp(live['last_candle']) + pd.Timedelta(hours=1):%d %b %H:%M} UTC</div>
    <div class="legend"><span><i style="background:var(--s1)"></i>sampled paths and p10 / p90</span><span><i style="background:var(--s2)"></i>median path (p50)</span></div>
    {fan}
    <div class="fig-note">Last close {live['last_close']:,.0f}. 24 hours on: p10 {live['p10'][-1]:,.0f} · p50 {live['p50'][-1]:,.0f} · p90 {live['p90'][-1]:,.0f}; {pct(live['share_of_paths_up_24h'])} of paths end higher.</div>
  </div>
</section>

<section aria-labelledby="dir-h">
  <div class="sec-head"><div class="eyebrow">Test 1 · Direction</div><h2 id="dir-h">{copy['dir_h']}</h2>
  <p class="lede">{copy['dir']}</p></div>
  <div class="fig"><div class="fig-title">Share of 24-hour moves called in the right direction</div>{dir_chart}
  <div class="fig-note">{s['forecasts']:,} forecasts on {s['days']} days. Kronos's median path, 1 / 4 / 12 / 24 hours ahead: {' / '.join(pct(bh[h], 1) for h in bh)}. Ranking the 12 coins against each other each day: rank correlation {c['daily_rank_ic_mean']:+.3f} (t = {c['daily_rank_ic_t']}).</div></div>
</section>

<section aria-labelledby="band-h">
  <div class="sec-head"><div class="eyebrow">Test 2 · The fan</div><h2 id="band-h">{copy['band_h']}</h2>
  <p class="lede">{copy['band']}</p></div>
  <div class="two">
    <div class="fig"><div class="fig-title">How often the actual price ended inside the p10–p90 band</div>
      <div class="legend"><span><i class="box" style="background:var(--s1)"></i>Kronos</span><span><i class="box" style="background:var(--s4)"></i>Band from the last 3 days' volatility</span></div>{cov}</div>
    <div class="fig"><div class="fig-title">Where the actual 24-hour return fell among Kronos's 100 paths</div>{pit_chart}
      <div class="fig-note">Bars are deciles: 0 = below all but 10 % of paths, 90 = above 90 % of them. A right fan gives 10 % in each; piles at both ends mean the fan is too narrow.</div></div>
  </div>
  <p class="meta">Scores for the whole 24-hour distribution (lower is better): CRPS {b['crps_24h_kronos']:.4f} for Kronos vs {b['crps_24h_baseline']:.4f} for the volatility band ({pct(-b['crps_skill_vs_baseline'], 0, True)}); quantile loss {b['pinball_24h_kronos']:.4f} vs {b['pinball_24h_baseline']:.4f}. Median band width 24 h ahead: {pct(b['median_width_kronos']['24'], 1)} for Kronos, {pct(b['median_width_baseline']['24'], 1)} for the volatility band.</p>
</section>

<section aria-labelledby="vol-h">
  <div class="sec-head"><div class="eyebrow">Test 3 · Volatility</div><h2 id="vol-h">{copy['vol_h']}</h2>
  <p class="lede">{copy['vol']}</p></div>
  <div class="table-wrap"><table>
    <thead><tr><th>Predictor of the next 24 hours' volatility</th><th>Correlation</th></tr></thead>
    <tbody>
      <tr class="hl"><td>Kronos (spread of its paths)</td><td>{v['corr_kronos']:.2f}</td></tr>
      <tr><td>Volatility of the last 3 days (EWMA)</td><td>{v['corr_baseline_ewm']:.2f}</td></tr>
      <tr><td>Volatility of the last 24 hours</td><td>{v['corr_last_24h']:.2f}</td></tr>
    </tbody></table></div>
  <p class="meta">Log volatility regressed on both: Kronos's coefficient {v['kronos_beta_given_baseline']} (t = {v['kronos_t_given_baseline']}) given the 3-day average. Its paths move {pct(1 - v['median_ratio_predicted_to_actual'])} less per hour than the market did (median).</p>
</section>

<section aria-labelledby="trade-h">
  <div class="sec-head"><div class="eyebrow">Test 4 · Trading it</div><h2 id="trade-h">{copy['trade_h']}</h2>
  <p class="lede">{copy['trade']}</p></div>
  <div class="fig"><div class="fig-title">Account value, daily rebalancing at 00:00 UTC, after costs and funding</div>{eq_chart}</div>
  <div class="table-wrap"><table>
    <thead><tr><th>Rule</th><th>Return</th><th>Sharpe</th><th>Worst DD</th><th>Exposure</th>{''.join(f'<th>{p}</th>' for p in periods)}</tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
  <p class="meta">Each position is held 24 hours and pays 12 bp to enter and leave (5 bp taker fee and 1 bp slippage each way) plus the real funding. Exposure is the average sum of absolute weights.</p>
</section>

<section aria-labelledby="ex-h">
  <div class="sec-head"><div class="eyebrow">One day, up close</div><h2 id="ex-h">Bitcoin on {ex['origin']:%d %b %Y}: the biggest 24-hour move in the test</h2></div>
  <div class="fig"><div class="fig-title">Kronos's 24-hour forecast made at 00:00 UTC vs what happened</div>
    <div class="legend"><span><i style="background:var(--s1)"></i>p10 / p90</span><span><i style="background:var(--s2)"></i>median (p50)</span><span><i style="background:var(--ink)"></i>actual price</span></div>{ex_chart}
  <div class="fig-note">Its p10–p90 band ran from {pct(float(ex['q10'][-1]), 1, True)} to {pct(float(ex['q90'][-1]), 1, True)}; Bitcoin moved {pct(float(ex['act'][-1]), 1, True)}.</div></div>
</section>

<section aria-labelledby="how-h" style="border-bottom:0">
  <div class="sec-head"><div class="eyebrow">Method</div><h2 id="how-h">How it was tested</h2></div>
  <ul class="points">{''.join(f'<li>{x}</li>' for x in copy['method'])}</ul>
</section>
<footer><div>Code: <code>src/tradinglab/kronos.py</code>, <code>scripts/kronos/</code>. Results: <code>reports/kronos/</code>. Kronos: github.com/shiyu-coder/Kronos (MIT licence).</div></footer>
</div>"""
    return page


def autoresearch_section(copy: dict) -> str:
    """The Karpathy-style autoresearch loop on Kronos (research/kronos_autoresearch), if it has been run."""
    d = Path(__file__).resolve().parents[2] / "research" / "kronos_autoresearch"
    if not (d / "holdout.json").exists():
        return ""
    res = pd.read_csv(d / "results.tsv", sep="\t")
    hold = json.loads((d / "holdout.json").read_text())
    sig = json.loads((d / "significance.json").read_text())
    names = list(hold)
    cols = dict(zip(names, ["--s1", "--s4", "--muted"]))
    curves = {}
    for n in names:
        bm = {**hold[n]["dev"]["by_month"], **hold[n]["holdout"]["by_month"]}
        idx = pd.to_datetime([k + "-01" for k in bm]) + pd.offsets.MonthEnd(0)
        curves[n] = pd.Series([1.0, *(1 + pd.Series(list(bm.values()))).cumprod()],
                              index=[idx[0] - pd.offsets.MonthEnd(1), *idx])
    eq = pd.DataFrame(curves)
    short = {names[0]: "Kronos volatility rule", names[1]: "Same rule, no Kronos", names[2]: "Baseline"}
    chart = C.lines(eq, cols, labels=short)
    st = {"keep": "kept", "discard": "discarded", "control": "control", "crash": "crashed"}
    rows = "".join(f'<tr{" class=hl" if r.status == "keep" else ""}><td>{int(r.run)}</td><td style="white-space:normal;text-align:left">{C.esc(r.description)}</td>'
                   f'<td>{r.score:+.2f}</td><td>{st.get(r.status, r.status)}</td></tr>' for r in res.itertuples())
    hrows = "".join(f'<tr{" class=hl" if i == 0 else ""}><td>{C.esc(short[n])}</td><td>{hold[n]["dev"]["sharpe"]:+.2f}</td>'
                    f'<td>{pct(hold[n]["dev"]["total_return"], 1, True)}</td><td>{hold[n]["holdout"]["sharpe"]:+.2f}</td>'
                    f'<td class="{"pos" if hold[n]["holdout"]["total_return"] > 0 else "neg"}">{pct(hold[n]["holdout"]["total_return"], 1, True)}</td>'
                    f'<td>{pct(hold[n]["holdout"]["max_drawdown"], 1)}</td><td>{hold[n]["holdout"]["turnover_per_day"]:.2f}</td></tr>'
                    for i, n in enumerate(names))
    paras = "".join(f"<p>{x}</p>" for x in copy["ar"])
    return f"""<section aria-labelledby="ar-h">
  <div class="sec-head"><div class="eyebrow">Autoresearch · can Kronos be made profitable?</div><h2 id="ar-h">{copy['ar_h']}</h2></div>
  <div class="verdict">{paras}</div>
  <div class="table-wrap"><table>
    <thead><tr><th>Rule</th><th>Scored Sharpe</th><th>Scored return</th><th>Locked Sharpe</th><th>Locked return</th><th>Worst DD (locked)</th><th>Turnover / day</th></tr></thead>
    <tbody>{hrows}</tbody></table></div>
  <div class="fig"><div class="fig-title">Account value by month: scored months Jul 2025 – Mar 2026, then the locked months Apr – Oct 2026</div>{chart}
  <div class="fig-note">Market-neutral, gross exposure at most 1, 6 bp per unit traded plus funding. Over all 16 months the Kronos rule's Sharpe is {sig['both_periods_sharpe']:.2f} ({pct(sig['both_periods_return'], 0, True)}); probability that its true Sharpe is above zero: {pct(sig['both_periods_psr_vs_0'])}. Deflated Sharpe of the scored-period pick after {sig['trials']} ideas: {sig['dev_dsr']:.2f} (0.95 would be convincing).</div></div>
  <details><summary>All {len(res)} runs of the loop</summary><div class="table-wrap"><table>
    <thead><tr><th>Run</th><th style="text-align:left">Idea</th><th>Scored Sharpe</th><th>Result</th></tr></thead>
    <tbody>{rows}</tbody></table></div></details>
</section>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/kronos")
    ap.add_argument("--copy", default="scripts/kronos/copy.json")
    a = ap.parse_args()
    d = Path(a.dir)
    (d / "report.html").write_text(build(d, json.loads(Path(a.copy).read_text())))
    print("wrote", d / "report.html")


if __name__ == "__main__":
    main()
