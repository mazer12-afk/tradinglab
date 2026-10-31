"""Day-trading strategies → trade intents for :func:`tradinglab.daytrade.sim.simulate_trades`.

Every signal is computed from bars up to the close of the decision minute; the simulator fills from the
next minute on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradinglab.config import utc
from tradinglab.daytrade.sim import Minutes


def five_minute_z(m: Minutes, span: int = 288) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(z, sigma, on5): the 5-minute log return in units of its EWMA volatility over the previous ~day of
    5-minute bars (excluding the bar itself), evaluated at 5-minute closes; ``on5`` marks those minutes."""
    c = m.close
    lc = np.log(c.ffill())
    r5 = lc - lc.shift(5)
    minute = (m.index.minute.to_numpy() % 5) == 4               # minute 4, 9, ... closes a 5-minute bar
    on5 = pd.DataFrame(np.repeat(minute[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    sig = r5.where(on5).ewm(span=span, min_periods=100, ignore_na=True).std().shift(5).ffill()
    return r5 / sig, sig, on5


@dataclass
class CascadeReversal:
    """Fade extreme 5-minute moves (liquidation cascades overshoot): after a move beyond ``k`` sigma, take the
    other side, by market order or with a limit ``offset`` sigma further into the move; exit at a take-profit
    of ``tp`` sigma, a stop of ``sl`` sigma (0 = none) or after ``hold`` minutes."""

    k: float = 5.0
    entry: str = "market"           # "market" | "limit"
    offset: float = 0.0             # limit distance beyond the decision close, in 5-minute sigmas
    ttl: int = 10                   # limit life, minutes
    tp: float = 0.0                 # take-profit, in 5-minute sigmas (0 = none)
    sl: float = 0.0                 # stop, in 5-minute sigmas (0 = none)
    hold: int = 60                  # maximum holding time, minutes
    size: float = 0.2               # notional per trade, fraction of equity
    sides: str = "both"             # "both" | "long" (buy crashes only) | "short" (sell spikes only)
    name: str = "cascade_reversal"

    def intents(self, m: Minutes, start=None, end=None) -> pd.DataFrame:
        z, sig, on5 = five_minute_z(m)
        hit = on5 & (z.abs() > self.k)
        if self.sides == "long":
            hit &= z < 0
        elif self.sides == "short":
            hit &= z > 0
        if start is not None:
            hit &= pd.DataFrame(np.repeat((m.index >= utc(start))[:, None], hit.shape[1], axis=1),
                                index=hit.index, columns=hit.columns)
        if end is not None:
            hit &= pd.DataFrame(np.repeat((m.index < utc(end))[:, None], hit.shape[1], axis=1),
                                index=hit.index, columns=hit.columns)
        st = hit.stack()
        st = st[st]
        if st.empty:
            return pd.DataFrame(columns=["t", "symbol", "side", "size", "entry", "price", "ttl", "tp", "sl", "max_hold"])
        t = st.index.get_level_values(0)
        sym = st.index.get_level_values(1)
        zz = z.stack().reindex(st.index).to_numpy()
        s5 = sig.stack().reindex(st.index).to_numpy()            # log-return sigma ≈ fractional sigma
        close = m.close.stack().reindex(st.index).to_numpy()
        side = -np.sign(zz).astype(int)
        price = close * np.exp(-side * self.offset * s5) if self.entry == "limit" else np.full(len(st), np.nan)
        return pd.DataFrame({"t": t, "symbol": sym, "side": side, "size": self.size, "entry": self.entry,
                             "price": price, "ttl": self.ttl if self.entry == "limit" else 0,
                             "tp": self.tp * s5, "sl": self.sl * s5, "max_hold": self.hold, "z": zz, "sigma": s5})
