"""Self-contained HTML dashboards (no external assets) for live trading and research."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

# Reference categorical palette (fixed order, light / dark steps).
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--surface-2:#f3f2ef;--text:#0b0b0b;--text-2:#52514e;--muted:#8a8984;
--grid:#e6e5e1;--good:#008300;--bad:#c4302b;%LIGHT%}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--surface-2:#242422;
--text:#f0efec;--text-2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;--good:#4cb04c;--bad:#e66767;%DARK%}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--surface-2:#242422;--text:#f0efec;--text-2:#c3c2b7;
--muted:#8f8e86;--grid:#33332f;--good:#4cb04c;--bad:#e66767;%DARK%}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.sub{color:var(--text-2);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface-2);border-radius:10px;padding:12px 14px}
.tile .l{color:var(--text-2);font-size:12px}.tile .v{font-size:22px;font-weight:600;margin-top:2px}
.tile .d{font-size:12px;color:var(--text-2)}
.hero .v{font-size:44px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;text-align:right;white-space:nowrap}
th{color:var(--text-2);font-weight:500;border-bottom:1px solid var(--grid)}td{border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
tr.sel td:first-child{font-weight:600}
.chart{position:relative;background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:8px 8px 4px}
.chart svg{display:block;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:6px 4px 2px;color:var(--text-2);font-size:12px}
.legend i{display:inline-block;width:14px;height:2px;margin-right:6px;vertical-align:middle;border-radius:1px}
.tip{position:absolute;pointer-events:none;background:var(--surface-2);color:var(--text);border:1px solid var(--grid);
border-radius:8px;padding:6px 8px;font-size:12px;display:none;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.12)}
.tip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.note{color:var(--text-2);font-size:13px;max-width:900px}
code{background:var(--surface-2);padding:1px 4px;border-radius:4px}
"""

JS = r"""
function fmtPct(v){return (v>=0?'+':'')+(100*v).toFixed(1)+'%'}
function renderLines(id, data){
  const el=document.getElementById(id); if(!el) return;
  const W=1100,H=360,L=56,R=16,T=12,B=28;
  const xs=data.x.map(s=>Date.parse(s)); const n=xs.length; if(n<2){el.innerHTML='<p class="note">Not enough data yet.</p>';return}
  let lo=Infinity,hi=-Infinity; data.series.forEach(s=>s.y.forEach(v=>{if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v)}}));
  if(data.base!=null){lo=Math.min(lo,data.base);hi=Math.max(hi,data.base)}
  const pad=(hi-lo)*0.06||1; lo-=pad; hi+=pad;
  const x0=xs[0],x1=xs[n-1]; const X=t=>L+(W-L-R)*(t-x0)/((x1-x0)||1); const Y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  const step=Math.pow(10,Math.floor(Math.log10((hi-lo)/4))); const m=[1,2,2.5,5,10].find(k=>(hi-lo)/(k*step)<=6)*step;
  let g='';
  for(let v=Math.ceil(lo/m)*m; v<=hi; v+=m){g+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/>`+
    `<text x="${L-8}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)">${data.fmt==='pct'?fmtPct(v):(Math.abs(v)>=1000?(v/1000).toFixed(v%1000?1:0)+'K':v.toFixed(2))}</text>`}
  const span=x1-x0, days=span/864e5; const nt=6;
  for(let i=0;i<=nt;i++){const t=x0+span*i/nt; const d=new Date(t);
    const lab=days>120?d.toISOString().slice(0,7):(days>3?d.toISOString().slice(5,10):d.toISOString().slice(5,16).replace('T',' '));
    g+=`<text x="${X(t)}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--muted)">${lab}</text>`}
  if(data.base!=null){g+=`<line x1="${L}" x2="${W-R}" y1="${Y(data.base)}" y2="${Y(data.base)}" stroke="var(--muted)" stroke-width="1"/>`}
  data.series.forEach((s,k)=>{let d='',pen=false; s.y.forEach((v,i)=>{if(v==null){pen=false;return} d+=(pen?'L':'M')+X(xs[i]).toFixed(1)+','+Y(v).toFixed(1); pen=true});
    g+=`<path d="${d}" fill="none" stroke="var(--s${k+1})" stroke-width="${s.emph?2.5:1.5}" stroke-linejoin="round" stroke-linecap="round" opacity="${s.emph?1:0.9}"/>`});
  g+=`<line id="${id}-cx" y1="${T}" y2="${H-B}" stroke="var(--muted)" stroke-width="1" visibility="hidden"/>`;
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${data.title||'chart'}">${g}<rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" id="${id}-hit"/></svg>`+
    `<div class="legend">${data.series.map((s,k)=>`<span><i style="background:var(--s${k+1})"></i>${s.name}</span>`).join('')}</div><div class="tip" id="${id}-tip"></div>`;
  const svg=el.querySelector('svg'), tip=document.getElementById(id+'-tip'), cx=document.getElementById(id+'-cx');
  svg.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(); const px=(ev.clientX-r.left)*W/r.width;
    if(px<L||px>W-R){tip.style.display='none';cx.setAttribute('visibility','hidden');return}
    const t=x0+(px-L)/(W-L-R)*(x1-x0); let i=0,best=Infinity; xs.forEach((v,j)=>{const d=Math.abs(v-t); if(d<best){best=d;i=j}});
    cx.setAttribute('x1',X(xs[i]));cx.setAttribute('x2',X(xs[i]));cx.setAttribute('visibility','visible');
    const rows=data.series.map((s,k)=>({k,name:s.name,v:s.y[i]})).filter(o=>o.v!=null).sort((a,b)=>b.v-a.v);
    tip.innerHTML=`<div style="color:var(--text-2);margin-bottom:3px">${data.x[i].replace('T',' ').slice(0,16)}</div>`+
      rows.map(o=>`<div><i style="background:var(--s${o.k+1})"></i>${o.name}: <b>${data.fmt==='pct'?fmtPct(o.v):o.v.toLocaleString(undefined,{maximumFractionDigits:2})}</b></div>`).join('');
    tip.style.display='block'; const left=(X(xs[i])/W)*r.width; tip.style.left=Math.min(left+12, r.width-tip.offsetWidth-4)+'px'; tip.style.top='16px'});
  svg.addEventListener('mouseleave',()=>{tip.style.display='none';cx.setAttribute('visibility','hidden')});
}
"""


def _css() -> str:
    light = "".join(f"--s{i + 1}:{c};" for i, c in enumerate(SERIES_LIGHT))
    dark = "".join(f"--s{i + 1}:{c};" for i, c in enumerate(SERIES_DARK))
    return CSS.replace("%LIGHT%", light).replace("%DARK%", dark)


def _page(title: str, body: str, charts: dict[str, dict]) -> str:
    init = "\n".join(f"renderLines({json.dumps(cid)}, {json.dumps(data)});" for cid, data in charts.items())
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_css()}</style>"
            f"</head><body><main>{body}</main><script>{JS}\n{init}</script></body></html>")


def _num(v, kind="f2"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    if kind == "pct":
        return f"{100 * v:+.1f}%"
    if kind == "pct0":
        return f"{100 * v:.1f}%"
    if kind == "money":
        return f"${v:,.0f}" if round(v) != 0 else "$0"
    return f"{v:.2f}"


def _opt(v, spec: str) -> str:
    return "" if v is None or pd.isna(v) else format(float(v), spec)


def _series_payload(frame: pd.DataFrame, names: list[str], emph: set[str], fmt: str, title: str,
                    base: float | None = None, max_points: int = 1500) -> dict:
    f = frame[names].copy()
    if len(f) > max_points:
        f = f.iloc[:: math.ceil(len(f) / max_points)]
    return {"x": [t.isoformat() for t in f.index], "fmt": fmt, "title": title, "base": base,
            "series": [{"name": n, "emph": n in emph,
                        "y": [None if pd.isna(v) else round(float(v), 6) for v in f[n]]} for n in names]}


# ---------------------------------------------------------------------- live
def live_report(cfg, store, out_path: Path) -> Path:
    status_path = Path(cfg.paths.state_dir) / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"accounts": {}}
    accounts = status.get("accounts", {})
    main = accounts.get("main", {})
    curves = {}
    for name in accounts:
        eq = store.equity_curve(name)
        if not eq.empty:
            curves[name] = eq["equity"] / accounts[name]["initial_capital"] - 1.0
    body = [f"<h1>TradingLab · live paper trading</h1><p class='sub'>Updated {html.escape(str(status.get('updated_at')))}"
            f" · feed <b>{html.escape(str(status.get('feed_mode')))}</b> · market data known up to "
            f"{html.escape(str(status.get('data_horizon')))} UTC</p>"]
    if main:
        cls = "pos" if main["pnl"] >= 0 else "neg"
        body.append("<div class='tiles'>"
                    f"<div class='tile hero'><div class='l'>Main account P&amp;L</div><div class='v {cls}'>"
                    f"{'+' if main['pnl'] >= 0 else '−'}${abs(main['pnl']):,.2f}</div>"
                    f"<div class='d'>{_num(main['return'], 'pct')} on {_num(main['initial_capital'], 'money')}</div></div>"
                    f"<div class='tile'><div class='l'>Equity</div><div class='v'>{_num(main['equity'], 'money')}</div>"
                    f"<div class='d'>marked {html.escape(str(main.get('marked_at')))}</div></div>"
                    f"<div class='tile'><div class='l'>Gross exposure</div><div class='v'>{main['gross']:.2f}×</div>"
                    f"<div class='d'>net {main['net']:+.2f}×</div></div>"
                    f"<div class='tile'><div class='l'>Strategy</div><div class='v' style='font-size:15px'>"
                    f"{html.escape(main['strategy'])}</div><div class='d'>funding paid {_num(main['funding_paid'], 'money')}"
                    f"</div></div></div>")
    charts = {}
    if curves:
        frame = pd.DataFrame(curves).sort_index().ffill()
        ranked = ["main"] + sorted([n for n in frame if n != "main"], key=lambda n: -frame[n].iloc[-1])
        names = ranked[:8]
        charts["live_eq"] = _series_payload(frame, names, {"main"}, "pct", "Live account returns", base=0.0)
        body.append("<h2>Account returns since inception</h2><div class='chart' id='live_eq'></div>")
    rows = "".join(
        f"<tr class='{'sel' if n == 'main' else ''}'><td>{html.escape(n)}</td><td>{html.escape(a['strategy'])}</td>"
        f"<td>{_num(a['equity'], 'money')}</td><td class='{'pos' if a['pnl'] >= 0 else 'neg'}'>{a['pnl']:+,.2f}</td>"
        f"<td>{_num(a['return'], 'pct')}</td><td>{a['gross']:.2f}</td><td>{a['net']:+.2f}</td>"
        f"<td>{a['pending_orders']}</td></tr>"
        for n, a in sorted(accounts.items(), key=lambda kv: (kv[0] != "main", -kv[1]["return"])))
    body.append("<h2>Accounts</h2><div class='tablewrap'><table><tr><th>Account</th><th>Strategy</th><th>Equity</th>"
                f"<th>P&amp;L</th><th>Return</th><th>Gross</th><th>Net</th><th>Pending</th></tr>{rows}</table></div>")
    if main.get("positions"):
        weights = main.get("weights", {})
        prow = "".join(
            f"<tr><td>{html.escape(s)}</td><td>{u:,.6f}</td><td>{_num(weights.get(s, float('nan')) * main['equity'], 'money')}"
            f"</td><td>{_num(weights.get(s), 'pct')}</td></tr>"
            for s, u in sorted(main["positions"].items(), key=lambda kv: -abs(weights.get(kv[0], 0.0))))
        body.append(f"<h2>Main positions</h2><div class='tablewrap'><table><tr><th>Symbol</th><th>Units</th>"
                    f"<th>Value</th><th>Weight</th></tr>{prow}</table></div>")
    orders = store.orders("main", limit=40)
    if not orders.empty:
        orow = "".join(
            f"<tr><td>{int(o['id'])}</td><td>{html.escape(o['symbol'])}</td><td>{o['qty']:+,.6f}</td>"
            f"<td>{html.escape(str(o['submit_ts'])[:19])}</td><td>{html.escape(str(o['status']))}</td>"
            f"<td>{_opt(o['fill_price'], ',.6g')}</td><td>{_opt(o['fee'], ',.2f')}</td></tr>"
            for _, o in orders.iterrows())
        body.append("<h2>Recent main-account orders</h2><div class='tablewrap'><table><tr><th>#</th><th>Symbol</th>"
                    f"<th>Qty</th><th>Submitted</th><th>Status</th><th>Fill</th><th>Fee</th></tr>{orow}</table></div>")
    ev = store.events(25)
    if ev:
        erow = "".join(f"<tr><td>{html.escape(e['ts'][:19])}</td><td>{html.escape(e['level'])}</td>"
                       f"<td>{html.escape(e['account'] or '')}</td><td style='text-align:left'>"
                       f"{html.escape(e['message'])}</td></tr>" for e in ev)
        body.append(f"<h2>Event journal</h2><div class='tablewrap'><table><tr><th>Time</th><th>Level</th>"
                    f"<th>Account</th><th style='text-align:left'>Message</th></tr>{erow}</table></div>")
    body.append("<p class='note'>Paper trading on real market prices. Orders are decided with information available "
                "at decision time and filled at the first trade price after submission (plus fees and slippage). "
                "In archive mode those prices are published by the exchange with a delay, so fills and marks appear "
                "later — they are never back-filled with hindsight.</p>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_page("TradingLab live", "".join(body), charts))
    return out_path


# ------------------------------------------------------------------ research
def research_report(report, out_path: Path) -> Path:
    cand = report.candidates.copy()
    sel = set(report.live_choice)
    cols = [("sharpe", "Sharpe", "f2"), ("cagr", "CAGR", "pct"), ("ann_vol", "Vol", "pct0"),
            ("max_drawdown", "Max DD", "pct"), ("psr", "PSR", "f2"), ("turnover_ann", "Turnover/yr", "f2"),
            ("cost_drag_ann", "Costs/yr", "pct0"), ("funding_drag_ann", "Funding/yr", "pct0")]
    cand = cand.sort_values("sharpe", ascending=False)
    rows = "".join(
        f"<tr class='{'sel' if n in sel else ''}'><td>{html.escape(n)}{' ★' if n in sel else ''}</td>"
        + "".join(f"<td>{_num(r.get(c), k)}</td>" for c, _, k in cols)
        + f"<td style='text-align:left;white-space:normal;min-width:260px'>{html.escape(str(r.get('description', '')))}</td></tr>"
        for n, r in cand.iterrows())
    head = "".join(f"<th>{h}</th>" for _, h, _ in cols)
    sel_t = report.selection_table
    srows = "".join(
        f"<tr class='{'sel' if r.get('selected') else ''}'><td>{html.escape(n)}</td><td>{_num(r.get('sharpe'))}</td>"
        f"<td>{_num(r.get('psr'))}</td><td>{_num(r.get('dsr'))}</td><td>{_num(r.get('dsr_xs'))}</td>"
        f"<td>{'yes' if r.get('eligible') else ''}</td>"
        f"<td>{'★' if r.get('selected') else ''}</td></tr>"
        for n, r in sel_t.sort_values("dsr", ascending=False).iterrows())
    h = report.holdout_ensemble or {}
    l = report.live_ensemble_oos or {}
    body = [f"<h1>TradingLab · research tournament</h1><p class='sub'>Data through {html.escape(report.data_end)} · "
            f"walk-forward out-of-sample from {html.escape(report.selection_window[0])} · created "
            f"{html.escape(report.created_at[:19])}</p>",
            "<div class='tiles'>"
            f"<div class='tile hero'><div class='l'>Holdout Sharpe (selection frozen before "
            f"{html.escape(report.holdout_window[0][:10])})</div><div class='v'>{_num(h.get('sharpe'))}</div>"
            f"<div class='d'>CAGR {_num(h.get('cagr'), 'pct')} · max DD {_num(h.get('max_drawdown'), 'pct')}</div></div>"
            f"<div class='tile'><div class='l'>Holdout ensemble</div><div class='v' style='font-size:15px'>"
            f"{html.escape(', '.join(report.holdout_choice) or 'none')}</div></div>"
            f"<div class='tile'><div class='l'>Live selection</div><div class='v' style='font-size:15px'>"
            f"{html.escape(', '.join(report.live_choice) or 'none (stay flat)')}</div>"
            f"<div class='d'>OOS Sharpe {_num(l.get('sharpe'))} · CAGR {_num(l.get('cagr'), 'pct')}</div></div></div>"]
    charts = {}
    curves = report.curves.copy()
    if not curves.empty:
        names = [c for c in curves.columns if c.startswith("ENSEMBLE")]
        names += [c for c in ["basket_long", "hold_btc"] if c in curves]
        others = cand.index.tolist()
        names += [c for c in others if c in curves and c not in names][: 8 - len(names)]
        frame = curves[names] - 1.0
        charts["res_eq"] = _series_payload(frame, names, {n for n in names if n.startswith("ENSEMBLE")},
                                           "pct", "Out-of-sample cumulative return", base=0.0)
        body.append("<h2>Out-of-sample cumulative return (net of costs and funding)</h2><div class='chart' id='res_eq'></div>")
    body.append(f"<h2>All candidates, full out-of-sample period</h2><div class='tablewrap'><table><tr><th>Strategy</th>"
                f"{head}<th style='text-align:left'>Description</th></tr>{rows}</table></div>")
    body.append(f"<h2>Selection window {html.escape(report.selection_window[0])} → "
                f"{html.escape(report.selection_window[1])}</h2><div class='tablewrap'><table><tr><th>Strategy</th>"
                f"<th>Sharpe</th><th>PSR</th><th>DSR</th><th>DSR (cross-sectional)</th><th>Eligible</th><th>Chosen</th></tr>"
                f"{srows}</table></div>")
    body.append("<p class='note'>Sharpe ratios are annualised on daily bars. PSR = probability the true Sharpe exceeds 0; "
                "DSR = PSR against the best Sharpe expected from luck alone given the (correlation-adjusted) number of "
                "strategies tried, under the null that every strategy has zero true Sharpe (Bailey &amp; López de "
                "Prado); the stricter cross-sectional variant uses the observed dispersion of all candidates' Sharpe "
                "ratios as the luck variance. Benchmarks (hold_btc, basket_long) are shown but not eligible. "
                "Universe: today's liquid Binance listings, so results carry survivorship bias (delisted coins "
                "such as LUNA/FTT are absent).</p>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_page("TradingLab research", "".join(body), charts))
    return out_path
