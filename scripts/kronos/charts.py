"""Small SVG chart helpers for the Kronos report (styled by the page's CSS classes and colour tokens)."""

from __future__ import annotations

import html
import math

import pandas as pd

W = 920


def _nice(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo or abs(hi) or 1.0
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (m * step) <= n:
            step *= m
            break
    first = math.ceil(lo / step) * step
    return [round(first + i * step, 10) for i in range(int((hi - first) / step + 1e-9) + 1)]


def esc(s) -> str:
    return html.escape(str(s))


def fan(hist: dict, future_t: list[str], paths: list[list[float]], p10, p50, p90, last_close: float,
        actual: list[float] | None = None, H: int = 360, label_fmt="{:,.0f}") -> str:
    """Candles of the recent past, then the sampled paths with their p10 / p50 / p90 (and what happened)."""
    ht = pd.to_datetime(hist["t"])
    ft = pd.to_datetime(future_t)
    n_h, n_f = len(ht), len(ft)
    left, right, top, bottom = 8, 78, 14, 30
    pw = W - left - right
    xs = lambda i: left + pw * (i + 0.5) / (n_h + n_f)                       # noqa: E731
    vals = list(hist["low"]) + list(hist["high"]) + [v for p in paths for v in p] + (actual or [])
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    ys = lambda v: top + (H - top - bottom) * (hi - v) / (hi - lo)          # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Forecast fan">']
    for v in _nice(lo, hi, 6):
        out.append(f'<line class="viz-grid" x1="{left}" x2="{W - right}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/>'
                   f'<text class="viz-tick" x="{W - right + 6}" y="{ys(v) + 4:.1f}">{esc(label_fmt.format(v))}</text>')
    x0 = xs(n_h - 0.5)
    out.append(f'<rect x="{x0:.1f}" y="{top}" width="{W - right - x0:.1f}" height="{H - top - bottom}" class="fan-zone"/>')
    allt = list(ht) + list(ft)
    for i, t in enumerate(allt):
        if t.hour == 0:
            out.append(f'<line class="viz-grid" x1="{xs(i):.1f}" x2="{xs(i):.1f}" y1="{top}" y2="{H - bottom}"/>'
                       f'<text class="viz-tick" x="{xs(i):.1f}" y="{H - 10}" text-anchor="middle">{t:%d %b}</text>')
    bw = max(1.5, pw / (n_h + n_f) * 0.62)
    for i in range(n_h):
        o, h_, l_, c = (hist[k][i] for k in ("open", "high", "low", "close"))
        cls = "up" if c >= o else "down"
        out.append(f'<line class="wick {cls}" x1="{xs(i):.1f}" x2="{xs(i):.1f}" y1="{ys(h_):.1f}" y2="{ys(l_):.1f}"/>'
                   f'<rect class="body {cls}" x="{xs(i) - bw / 2:.1f}" y="{ys(max(o, c)):.1f}" width="{bw:.1f}" '
                   f'height="{max(1.0, abs(ys(o) - ys(c))):.1f}"/>')
    def poly(vs, start=n_h - 1, first=last_close):
        pts = [(xs(start), ys(first))] + [(xs(n_h + j), ys(v)) for j, v in enumerate(vs)]
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    for p in paths:
        out.append(f'<polyline class="fan-path" points="{poly(p)}"/>')
    band = [(xs(n_h - 1), ys(last_close))] + [(xs(n_h + j), ys(v)) for j, v in enumerate(p90)] + \
           [(xs(n_h + j), ys(v)) for j, v in reversed(list(enumerate(p10)))]
    out.append(f'<polygon class="fan-band" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in band)}"/>')
    out.append(f'<polyline class="fan-q" points="{poly(p10)}"/><polyline class="fan-q" points="{poly(p90)}"/>')
    out.append(f'<polyline class="fan-med" points="{poly(p50)}"/>')
    if actual:
        out.append(f'<polyline class="fan-act" points="{poly(actual)}"/>')
    xe = xs(n_h + n_f - 1)
    for v, lab, cls in ((p90[-1], "p90", "q"), (p50[-1], "p50", "med"), (p10[-1], "p10", "q")):
        out.append(f'<text class="fan-lab {cls}" x="{xe + 6:.1f}" y="{ys(v) + 4:.1f}">{lab}</text>')
    out.append(f'<circle class="fan-now" cx="{xs(n_h - 1):.1f}" cy="{ys(last_close):.1f}" r="3.5"/>')
    out.append("</svg>")
    return "".join(out)


def bars(cats: list[str], series: dict[str, list[float]], colors: dict[str, str], ref: float | None = None,
         ref_label: str = "", fmt="{:.0%}", H: int = 250, lo: float = 0.0, hi: float | None = None, W: int = W,
         ref_left: bool = False) -> str:
    """Grouped vertical bars, one group per category, with an optional reference line."""
    left, right, top, bottom = 44, 12, 18, 34
    hi = hi if hi is not None else max(max(v) for v in series.values()) * 1.15
    ys = lambda v: top + (H - top - bottom) * (hi - v) / (hi - lo)          # noqa: E731
    gw = (W - left - right) / len(cats)
    k = len(series)
    bw = min(46, gw * 0.7 / k)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for v in _nice(lo, hi, 4):
        out.append(f'<line class="viz-grid" x1="{left}" x2="{W - right}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/>'
                   f'<text class="viz-tick" x="{left - 6}" y="{ys(v) + 4:.1f}" text-anchor="end">{esc(fmt.format(v))}</text>')
    for i, c in enumerate(cats):
        cx = left + gw * (i + 0.5)
        for j, (name, vals) in enumerate(series.items()):
            v = vals[i]
            x = cx - k * bw / 2 + j * bw
            y0, y1 = ys(max(v, lo)), ys(lo)
            out.append(f'<rect x="{x + 1:.1f}" y="{y0:.1f}" width="{bw - 2:.1f}" height="{max(0.5, y1 - y0):.1f}" '
                       f'style="fill:var({colors[name]})"><title>{esc(name)}: {esc(fmt.format(v))}</title></rect>'
                       f'<text class="viz-val" x="{x + bw / 2:.1f}" y="{y0 - 5:.1f}" text-anchor="middle">{esc(fmt.format(v))}</text>')
        out.append(f'<text class="viz-cat" x="{cx:.1f}" y="{H - 12}" text-anchor="middle">{esc(c)}</text>')
    if ref is not None:
        out.append(f'<line class="viz-refline" x1="{left}" x2="{W - right}" y1="{ys(ref):.1f}" y2="{ys(ref):.1f}"/>'
                   f'<text class="viz-reflabel" x="{left + 4 if ref_left else W - right}" y="{ys(ref) - 6:.1f}" '
                   f'text-anchor="{"start" if ref_left else "end"}">{esc(ref_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def lines(df: pd.DataFrame, colors: dict[str, str], H: int = 300, fmt="{:+.0%}", base: float = 1.0,
          labels: dict[str, str] | None = None) -> str:
    """Account values over time (one line per column, indexed to ``base``), with end labels."""
    left, right, top, bottom = 48, 170, 14, 30
    t = pd.to_datetime(df.index)
    x0, x1 = t.min().value, t.max().value
    xs = lambda v: left + (W - left - right) * (v - x0) / (x1 - x0)          # noqa: E731
    lo, hi = float(df.min().min()), float(df.max().max())
    lo, hi = min(lo, base) - 0.02, max(hi, base) + 0.02
    ys = lambda v: top + (H - top - bottom) * (hi - v) / (hi - lo)          # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for v in _nice(lo, hi, 5):
        out.append(f'<line class="viz-grid" x1="{left}" x2="{W - right}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/>'
                   f'<text class="viz-tick" x="{left - 6}" y="{ys(v) + 4:.1f}" text-anchor="end">{esc(fmt.format(v - base))}</text>')
    out.append(f'<line class="viz-base" x1="{left}" x2="{W - right}" y1="{ys(base):.1f}" y2="{ys(base):.1f}"/>')
    for m in pd.date_range(t.min().normalize() + pd.offsets.MonthBegin(0), t.max(), freq="3MS"):
        out.append(f'<text class="viz-tick" x="{xs(m.value):.1f}" y="{H - 10}" text-anchor="middle">{m:%b %Y}</text>')
    ends = []
    for c in df.columns:
        s = df[c].dropna()
        pts = " ".join(f"{xs(pd.Timestamp(i).value):.1f},{ys(v):.1f}" for i, v in s.items())
        out.append(f'<polyline points="{pts}" style="fill:none;stroke:var({colors[c]});stroke-width:{2 if c == df.columns[0] else 1.6}"/>')
        ends.append([ys(s.iloc[-1]), c, s.iloc[-1]])
    ends.sort()
    for i in range(1, len(ends)):                                           # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for y, c, v in ends:
        out.append(f'<text class="viz-endlabel" x="{W - right + 6}" y="{y + 4:.1f}" style="fill:var({colors[c]})">'
                   f'{esc((labels or {}).get(c, c))} {esc(fmt.format(v - base))}</text>')
    out.append("</svg>")
    return "".join(out)
