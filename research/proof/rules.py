"""Frozen rules for the proof tests. A rule here is never edited once committed; a change is a new rule.

Each rule: (panel, **params) -> target weights (origins x coins) before the book's gross cap; ``DEFAULTS`` holds its
frozen parameters and ``NEIGHBOURS`` the settings checked for robustness (each parameter halved and doubled).
"""

from __future__ import annotations

import xs


# ------------------------------------------------------------------ R1, frozen 7 Nov 2026 (registry id 2)
def lowvol_rv(p: xs.Panel, vol_hours: int = 24, span: int = 72, step: float = 0.05):
    """Market-neutral low volatility: long the coins with the lowest realised volatility (hourly returns over the
    last ``vol_hours``, averaged with an EWM of ``span`` 4-hour periods), short the highest; cross-sectional z-score
    weights at gross 1; positions move ``step`` of the way to target every 4 hours. Found by the Kronos
    autoresearch loop (control run 15) on 12 coins, Jul 2025 - Oct 2026."""
    sig = -p.hourly_vol(vol_hours).ewm(span=span).mean()
    return xs.ease(xs.zscore_weights(sig, p.live), step)


RULES = {
    "lowvol_rv": {"fn": lowvol_rv, "defaults": {"vol_hours": 24, "span": 72, "step": 0.05}},
}
