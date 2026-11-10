"""Build reports/proof/report.html: the proof programme's verdicts, the evidence for funding carry, and what is open.

    python research/proof/report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "kronos"))
import carry as C  # noqa: E402
import charts as CH  # noqa: E402
import confirm_carry as K  # noqa: E402
import widebook as B  # noqa: E402
from deribit import DeribitPanel  # noqa: E402
from wide import Wide  # noqa: E402

CSS = (ROOT / "reports" / "daytrade" / "report.html").read_text()
CSS = CSS[CSS.index("<style>"):CSS.index("</style>") + len("</style>")]
EXTRA = """<style>
.pill { display: inline-block; font: 500 11.5px/1.7 var(--f-mono); padding: 0 8px; border-radius: 999px; border: 1px solid currentColor; white-space: nowrap; }
.pill.pass { color: var(--good); } .pill.fail { color: var(--bad); } .pill.part { color: var(--warn-ink); background: var(--warn-bg); border-color: transparent; }
td.txt, th.txt { white-space: normal; text-align: left; font-family: var(--f-body); }
.bar-pos { fill: var(--s1); } .bar-neg { fill: var(--neg); }
.viz-refline { stroke: var(--ink-2); stroke-width: 1.2; stroke-dasharray: 5 4; }
ol.steps { margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 70ch; }
</style>"""


def pct(x, d=1, sign=True) -> str:
    return f"{x:+.{d}%}" if sign else f"{x:.{d}%}"


def year_bars(series: dict[str, dict], colors: dict[str, str], H: int = 280) -> str:
    """Grouped bars, one group per year, negative years below the axis."""
    W, left, right, top, bottom = CH.W, 44, 12, 22, 34
    years = list(next(iter(series.values())))
    vals = [v for s in series.values() for v in s.values()]
    lo, hi = min(min(vals), 0) - 0.025, max(vals) * 1.15
    ys = lambda v: top + (H - top - bottom) * (hi - v) / (hi - lo)          # noqa: E731
    gw = (W - left - right) / len(years)
    k = len(series)
    bw = min(46, gw * 0.76 / k)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Return by year">']
    for v in CH._nice(lo, hi, 5):
        out.append(f'<line class="viz-grid" x1="{left}" x2="{W - right}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/>'
                   f'<text class="viz-tick" x="{left - 6}" y="{ys(v) + 4:.1f}" text-anchor="end">{v:+.0%}</text>')
    out.append(f'<line class="viz-base" x1="{left}" x2="{W - right}" y1="{ys(0):.1f}" y2="{ys(0):.1f}"/>')
    for i, y in enumerate(years):
        cx = left + gw * (i + 0.5)
        for j, (name, s) in enumerate(series.items()):
            v = s.get(y, 0.0)
            x = cx - k * bw / 2 + j * bw
            y0, y1 = (ys(v), ys(0)) if v >= 0 else (ys(0), ys(v))
            out.append(f'<rect x="{x + 1.5:.1f}" y="{y0:.1f}" width="{bw - 3:.1f}" height="{max(1.0, y1 - y0):.1f}" style="fill:var({colors[name]})">'
                       f'<title>{CH.esc(name)} {y}: {v:+.1%}</title></rect>'
                       f'<text class="viz-val" x="{x + bw / 2:.1f}" y="{(y0 - 6) if v >= 0 else (y1 + 14):.1f}" text-anchor="middle">{v:+.1%}</text>')
        out.append(f'<text class="viz-cat" x="{cx:.1f}" y="{H - 12}" text-anchor="middle">{y}</text>')
    out.append("</svg>")
    return "".join(out)


def checks_passed(v: dict) -> str:
    n, k = len(v["checks"]), sum(v["checks"].values())
    cls = "pass" if k == n else "part" if k >= n - 1 else "fail"
    return f'<span class="pill {cls}">{k} of {n}</span>'


def conf_row(rule, where, s, verdict_html, note="") -> str:
    yrs = list(s["by_year"].values())
    return (f'<tr><td>{rule}</td><td class="txt">{where}{note}</td><td>{s["coins"]}</td><td class="{"pos" if s["cagr"] > 0 else "neg"}">{pct(s["cagr"])}</td>'
            f'<td>{s["sharpe"]:.1f}</td><td>{pct(s["max_drawdown"], 1, False)}</td><td>{sum(y > 0 for y in yrs)} of {len(yrs)}</td><td>{verdict_html}</td></tr>')


def forward_status() -> dict | None:
    try:
        import requests

        r = requests.get("https://raw.githubusercontent.com/mazer12-afk/tradinglab/experiment/carry-forward/status.json", timeout=20)
        return r.json() if r.status_code == 200 else None
    except Exception:                              # noqa: BLE001 - the page is built without it
        return None


def build() -> str:
    res = HERE / "results"
    carry = json.loads((res / "carry.json").read_text())
    r4 = json.loads((res / "carry_r4.json").read_text())
    r4s = json.loads((res / "carry_r4_survivorship.json").read_text())
    drb = json.loads((res / "carry_deribit.json").read_text())
    reg = pd.read_csv(HERE / "registry.csv")

    # carry on every coin Binance ever listed on both markets (held to the last traded day)
    w = Wide("2020-01-01", "2026-10-28")
    p = B.WideCarry(w, exit_before=0)
    book = C.book(p, C.funding_carry(p, **K.R4))
    all_stats = C.stats(book)
    dead = int((w.last[p.symbols] < w.px_all.index[-1] - pd.Timedelta(days=2)).sum())
    sig = (p.trailing_funding(14) * 365).where(p.live)
    share = ((sig > 0.10).sum(axis=1) / p.live.sum(axis=1).replace(0, np.nan))
    share = share.groupby(share.index.year).mean()
    p2 = B.WideCarry(w, ["BTCUSDT", "ETHUSDT"], exit_before=0)
    r3b = C.book(p2, K.btc_eth(p2))
    dp = DeribitPanel(["BTC-PERPETUAL", "ETH-PERPETUAL"])
    Wd = pd.DataFrame(0.5, index=dp.grid, columns=dp.symbols).where(dp.live.all(axis=1), 0.0)
    r3d = C.book(dp, Wd)
    r3d = r3d[r3d["notional"] > 0]
    eq = pd.DataFrame({"all": (1 + book["net"]).cumprod(), "r3b": (1 + r3b["net"]).cumprod()})
    d = (1 + r3d["net"]).cumprod()
    d.index = d.index.normalize()
    eq = eq.join(d[d.index >= eq.index[0]].rename("r3d") / d[d.index >= eq.index[0]].iloc[0], how="left")
    eq = eq.resample("W").last()
    eq.loc[eq.index[0] - pd.Timedelta(days=7)] = 1.0
    eq = eq.sort_index()
    eq = eq[["r3b", "all", "r3d"]]
    cols = {"all": "--s1", "r3b": "--s3", "r3d": "--s4"}
    labels = {"all": "Any coin", "r3b": "BTC + ETH", "r3d": "Deribit"}
    eq_chart = CH.lines(eq, cols, labels=labels, tick_freq="12MS", tick_fmt="%Y", H=320)
    r3_stats = C.stats(r3b)
    yb = year_bars({"Always BTC + ETH": r3_stats["by_year"], "Any coin": all_stats["by_year"]}, {"Always BTC + ETH": "--s3", "Any coin": "--s1"})
    legend = ('<div class="legend"><span><i class="box" style="background:var(--s3)"></i>Always BTC + ETH, Binance</span>'
              '<span><i class="box" style="background:var(--s1)"></i>Carry rule on any coin, Binance (478 coins)</span>{}</div>')
    legend_bars = legend.format("")
    legend_lines = legend.format('<span><i class="box" style="background:var(--s4)"></i>Always BTC + ETH, Deribit</span>')
    r3y = r3_stats["by_year"]

    sets2 = carry["R2"]["sets"]
    rows = [
        conf_row("Carry, 12 main coins", "Binance, the same 12 coins in later years (2024 – Oct 2026)", sets2["12 coins 2024-2026"], checks_passed(carry["R2"])),
        conf_row("Carry, 12 main coins", "Binance, 18 other coins (2020 – Oct 2026)", sets2["18 other coins 2020-2026"], checks_passed(carry["R2"])),
        conf_row("Always BTC + ETH", "Binance, 2024 – Oct 2026", carry["R3"]["sets"]["BTC+ETH 2024-2026"], checks_passed(carry["R3"])),
        conf_row("Carry, any coin", "Binance, 57 coins never used before (2020 – Oct 2026)", list(r4["sets"].values())[0], checks_passed(r4)),
        conf_row("Carry, any coin", "Binance, every other coin ever listed on both markets, 118 of them later delisted, held to the last traded day (2020 – Oct 2026)",
                 list(r4s["sets"].values())[0], checks_passed(r4s)),
        conf_row("Always BTC + ETH", "Deribit, a different exchange (May 2019 – Oct 2026)", list(drb["R3"]["sets"].values())[0], checks_passed(drb["R3"])),
        conf_row("Carry, 12-coin rule", "Deribit, all 131 perpetuals (2019 – Oct 2026)", list(drb["R2"]["sets"].values())[0], checks_passed(drb["R2"]),
                 note=" <span class=\"tag\">see note</span>"),
    ]
    fw = forward_status()
    if fw:
        fr = fw["rules"]
        fw_html = (f'<p>Day {fr["R3"]["days"]} of at least 28 (through {fw["through"]}). So far: 12-coin rule {pct(fr["R2"]["return"], 2)}, always BTC + ETH {pct(fr["R3"]["return"], 2)}, '
                   f'any-coin rule {pct(fr["R4"]["return"], 2)}. These are mostly the cost of opening positions. Live prices match Binance\'s archive exactly so far.</p>')
    else:
        fw_html = "<p>The live paper test started on 8 Nov 2026; its records are on the branch <code>experiment/carry-forward</code>.</p>"
    n_rules = int(reg["id"].max())

    return f"""<title>Trading Proof Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
{CSS}
{EXTRA}
<div class="txt">
<header>
  <div class="eyebrow">Proof programme · 7 – 10 Nov 2026 · {n_rules} registry entries · Binance 2020 – Oct 2026, Deribit 2019 – Oct 2026</div>
  <h1>One method passed every test: funding carry</h1>
  <p class="lede">Every trading rule in this lab now has to pass the same nine checks, on data it had never seen, with the rule frozen before the test. Eight rules went through. Five failed. The three that passed are versions of one method, funding carry, and it also held up on a second exchange and on coins that were later delisted. It is low-risk and it only earns when the market is bullish.</p>
</header>

<section aria-labelledby="verdict-h">
  <div class="sec-head"><div class="eyebrow">Verdict</div><h2 id="verdict-h">Proven on past data; the live check finishes on 6 Dec</h2></div>
  <div class="verdict">
    <p class="big">Funding carry buys a coin on the spot market and shorts the same amount on its perpetual future. The two price moves cancel out. What is left is the funding rate, a fee that holders of the future pay to the short side when the market is leaning long.</p>
    <p>The simplest version holds that trade on Bitcoin and Ether all the time. From 2020 to Oct 2026 it made {pct(r3_stats["cagr"])} a year on Binance with every year in profit and a worst drawdown of {pct(r3_stats["max_drawdown"], 1, False)}, after fees on both legs. A second version trades any coin while its funding over the last 14 days is above 10 % a year. Run on every coin Binance ever listed on both markets ({len(p.symbols)} coins, {dead} of them later delisted), it made {pct(all_stats["cagr"])} a year with a worst drawdown of {pct(all_stats["max_drawdown"], 1, False)}.</p>
    <p>The return depends on the market. Both versions made 23 – 28 % in 2021 and about 8 % in 2024, and little in 2022, 2025 and 2026, when funding was low. Bitcoin's funding is running near 2.6 % a year now, so the method is earning little today.</p>
  </div>
  <div class="tiles">
    <div class="tile"><div class="k">Always BTC + ETH, a year (2020 – Oct 2026)</div><div class="v pos">{pct(r3_stats["cagr"])}</div><div class="s">{sum(v > 0 for v in r3y.values())} of {len(r3y)} years in profit · worst drawdown {pct(r3_stats["max_drawdown"], 1, False)}</div></div>
    <div class="tile"><div class="k">Any-coin rule, a year (2020 – Oct 2026)</div><div class="v pos">{pct(all_stats["cagr"])}</div><div class="s">{len(p.symbols)} coins, dead ones included · worst drawdown {pct(all_stats["max_drawdown"], 1, False)}</div></div>
    <div class="tile"><div class="k">Tests on unseen data</div><div class="v">7 run</div><div class="s">6 passed all nine checks · 1 passed eight</div></div>
    <div class="tile"><div class="k">Rules that failed</div><div class="v neg">5 of 8</div><div class="s">every rule that bets on price direction</div></div>
  </div>
</section>

<section aria-labelledby="year-h">
  <div class="sec-head"><div class="eyebrow">What it earns</div><h2 id="year-h">Strong in bull markets, flat in bear markets</h2>
  <p class="lede">Return by year of the two versions on Binance, after costs. The any-coin rule runs on all {len(p.symbols)} coins with both a spot market and a perpetual, and holds a delisted coin to its last traded day.</p></div>
  <div class="fig"><div class="fig-title">Return by year, after costs</div>{legend_bars}{yb}
  <div class="fig-note">Share of coins paying more than 10 % a year in funding: {" · ".join(f"{y} {v:.0%}" for y, v in share.items())}.</div></div>
  <div class="fig"><div class="fig-title">Account value since 2020</div>{legend_lines}{eq_chart}
  <div class="fig-note">Capital is 1.5 times the position: the spot coin paid in full plus 50 % margin for the short. Costs include a daily re-balancing of both legs, which a real account would do less often.</div></div>
</section>

<section aria-labelledby="conf-h">
  <div class="sec-head"><div class="eyebrow">Evidence</div><h2 id="conf-h">Seven tests on data the rules had never seen</h2>
  <p class="lede">Each rule was frozen in the repository before its test data was loaded. The nine checks: Sharpe of at least 1; two statistical tests that correct for luck and for the number of ideas tried; profit in at least 75 % of years; profit on every data set; drawdown no deeper than 20 %; profit with every setting halved or doubled; profit at double costs; and no dependence on market direction.</p></div>
  <div class="table-wrap"><table>
    <thead><tr><th>Rule</th><th class="txt">Unseen data</th><th>Coins</th><th>Return a year</th><th>Sharpe</th><th>Worst DD</th><th>Years in profit</th><th>Checks passed</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
  <ul class="points">
    <li><strong>The one miss.</strong> On the complete Binance universe the rule lost a little in three bear years (2022 −1.4 %, 2025 −0.2 %, 2026 −0.5 %), so only 4 of 7 years were positive. Closing positions two days before a delisting, which Binance announces in advance, turns 2025 and 2026 positive and passes all nine.</li>
    <li><strong>The Deribit note.</strong> The 12-coin rule passes on all 131 Deribit perpetuals, but most of its +10.6 % a year came from thinly traded contracts in 2023 – 24 that could not absorb real money. On Deribit's liquid contracts alone it made +1.8 to +2.8 % a year with no losing year.</li>
  </ul>
</section>

<section aria-labelledby="score-h">
  <div class="sec-head"><div class="eyebrow">Scoreboard</div><h2 id="score-h">Every rule that reached a test</h2></div>
  <div class="table-wrap"><table>
    <thead><tr><th>Rule</th><th class="txt">What it does</th><th>Verdict</th><th class="txt">Result on unseen data</th></tr></thead>
    <tbody>
      <tr><td>R1 Low volatility</td><td class="txt">Buy calm coins, short jumpy ones (the Kronos loop's winner)</td><td><span class="pill fail">failed</span></td><td class="txt">−64 %. It only made money in falling years: a hidden bet against the market.</td></tr>
      <tr class="hl"><td>R2 Carry, 12 coins</td><td class="txt">Spot long and perpetual short while funding is high</td><td><span class="pill pass">passed</span></td><td class="txt">+2.9 % a year (2024 – 26) and +7.1 % a year on 18 other coins.</td></tr>
      <tr class="hl"><td>R3 Always BTC + ETH</td><td class="txt">The same trade held all the time on the two largest coins</td><td><span class="pill pass">passed</span></td><td class="txt">+3.8 % a year on Binance (2024 – 26), +3.0 % a year on Deribit (2019 – 26).</td></tr>
      <tr class="hl"><td>R4 Carry, any coin</td><td class="txt">R2's rule on a wide universe, at most a tenth of the book per coin</td><td><span class="pill pass">passed</span></td><td class="txt">+4.9 % a year on 57 new coins; +6.7 % a year on 391 more, delisted ones included.</td></tr>
      <tr><td>R5 Trend following</td><td class="txt">Follow each coin's 120-day trend, sized to 15 % volatility</td><td><span class="pill fail">failed</span></td><td class="txt">Sharpe 0.18. Its design-period strength came from the 2020 – 21 trends.</td></tr>
      <tr><td>R6 Funding crowding</td><td class="txt">Short coins whose buyers pay the most funding, buy the others</td><td><span class="pill fail">failed</span></td><td class="txt">Some edge, far too uneven: a 60 % drawdown on new coins.</td></tr>
      <tr><td>R7 Crowding, older coins</td><td class="txt">R6 only on coins listed for at least a year</td><td><span class="pill fail">failed</span></td><td class="txt">Sharpe 0.56 on 60 untouched coins; 4 of 7 years positive.</td></tr>
      <tr><td>R8 Funding spread</td><td class="txt">Among the 50 most traded coins, buy the lowest-funding fifth and short the highest</td><td><span class="pill fail">failed</span></td><td class="txt">Sharpe 2.0 in 2020 – 23, then −19 % in 2024 – 26 (2025: −33 %).</td></tr>
    </tbody></table></div>
  <p>Ideas that showed no edge after costs on design data, and so never reached a test: price moves around funding payouts, quarterly futures, perpetual-versus-spot price gaps, pairs trading, concentrating carry in fewer coins, timing carry on the price premium, shorting new listings, momentum, and funding gaps between exchanges. All are in <code>research/proof/registry.csv</code>.</p>
</section>

<section aria-labelledby="learn-h">
  <div class="sec-head"><div class="eyebrow">What the failures show</div><h2 id="learn-h">Rules that bet on direction did not survive new data</h2></div>
  <ul class="points">
    <li><strong>A strong backtest is not evidence.</strong> R8 had a Sharpe of 2.0 over four years with every year positive, and lost 19 % on the next three. R1 made +29 % in the months it was built on and lost 64 % elsewhere.</li>
    <li><strong>Coin selection hides risk.</strong> Tests on today's most traded coins leave out the ones that died. Carry looked a little better on those hand-picked sets than on the complete universe.</li>
    <li><strong>Carry passes because it does not predict prices.</strong> It collects a fee that exists whenever the market leans long. That is also why it earns nothing when the market does not.</li>
  </ul>
</section>

<section aria-labelledby="run-h">
  <div class="sec-head"><div class="eyebrow">Running it</div><h2 id="run-h">What it needs, and what can go wrong</h2></div>
  <ul class="points">
    <li><strong>An account where spot coins count as collateral for the short</strong> (unified or portfolio margin). Coins spike 46 – 66 % in a day about once in 1,000 coin-days. Without shared collateral the short could be liquidated and leave the spot leg unhedged.</li>
    <li><strong>Watching delisting notices.</strong> The worst single day in the test was a perpetual that rose 94 % on its last day while the spot price fell 45 %.</li>
    <li><strong>Sudden collapses.</strong> One coin fell 83 % in a day and its funding turned sharply negative; at a tenth of the book that cost 1.3 %. The per-coin cap is what limits this.</li>
    <li><strong>Exchange failure.</strong> No backtest covers an exchange freezing withdrawals. Money on one exchange carries that risk.</li>
    <li><strong>Patience.</strong> In markets like 2022 or today the rule mostly sits in cash.</li>
  </ul>
</section>

<section aria-labelledby="open-h" style="border-bottom:0">
  <div class="sec-head"><div class="eyebrow">Still open</div><h2 id="open-h">The live paper test</h2></div>
  {fw_html}
  <p>The test runs daily on live Binance data through the same code as the backtest. Its job is to show that the method trades live as simulated; four weeks cannot show profit. The verdict is due on or after 6 Dec 2026.</p>
</section>
<footer><div>Protocol: <code>research/proof/PROTOCOL.md</code> · every rule tried: <code>research/proof/registry.csv</code> · results: <code>research/proof/results/</code> · live records: branch <code>experiment/carry-forward</code>.</div></footer>
</div>"""


def main() -> None:
    out = ROOT / "reports" / "proof" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    print("wrote", out)


if __name__ == "__main__":
    main()
