"""Event signals on the 5-minute grid with futures metrics (decision one minute after each 5-minute close)."""
from pathlib import Path
import numpy as np
import pandas as pd


def event_frame(m, met_dir):
    c = m.close
    syms = list(c.columns)
    lc = np.log(c.ffill())
    grid = c.index[(c.index.minute % 5 == 0)]
    r5 = (lc.shift(1) - lc.shift(6)).reindex(grid)
    sig = r5.ewm(span=288, min_periods=100).std().shift(1)
    z = r5 / sig
    mr5 = r5.mean(axis=1)
    idio = r5.sub(mr5, axis=0)
    iz = idio / idio.ewm(span=288, min_periods=100).std().shift(1)
    met = {s: pd.read_parquet(Path(met_dir) / f"{s}.parquet") for s in syms}
    oi = pd.DataFrame({s: met[s]["sum_open_interest_value"] for s in syms}).reindex(grid)
    doi = np.log(oi).diff()
    doi_z = doi / doi.ewm(span=288, min_periods=100).std().shift(1)
    return {"grid": grid, "z": z, "iz": iz, "sig": sig, "doi_z": doi_z, "close": c.reindex(grid)}


def intents_from_mask(ev, mask, side, entry="market", offset=0.0, ttl=0, hold=240, size=0.2, tp=0.0, sl=0.0):
    st = mask.stack()
    st = st[st]
    if st.empty:
        return pd.DataFrame(columns=["t", "symbol", "side", "size", "entry", "price", "ttl", "tp", "sl", "max_hold"])
    s5 = ev["sig"].stack().reindex(st.index).to_numpy()
    close = ev["close"].stack().reindex(st.index).to_numpy()
    price = close * np.exp(-side * offset * s5) if entry == "limit" else np.full(len(st), np.nan)
    return pd.DataFrame({"t": st.index.get_level_values(0), "symbol": st.index.get_level_values(1), "side": side,
                         "size": size, "entry": entry, "price": price, "ttl": ttl, "tp": tp * s5, "sl": sl * s5,
                         "max_hold": hold})


def market_hedge(trades, m, hedge_cost):
    """Hedged net: coin trade minus the equal-weight 12-coin market return over the same window, less hedge costs."""
    lc = np.log(m.close.ffill())
    mkt = lc.diff().mean(axis=1).fillna(0.0).cumsum()        # log index of the equal-weight market
    t = trades.copy()
    ok = t["filled"]
    a = mkt.reindex(t.loc[ok, "entry_time"]).to_numpy()
    b = mkt.reindex(t.loc[ok, "exit_time"]).to_numpy()
    t.loc[ok, "gross"] = t.loc[ok, "gross"] - t.loc[ok, "side"].to_numpy() * (np.exp(b - a) - 1.0)
    t.loc[ok, "fees"] = t.loc[ok, "fees"] + hedge_cost
    return t
