"""Builds the candidate strategy zoo from the lab configuration."""

from __future__ import annotations

from tradinglab.config import LabConfig, bars_per_year
from tradinglab.strategies.base import Strategy
from tradinglab.strategies.classic import classic_candidates

DEEP_KINDS = ("lstm", "mom_trans", "deepm")


def deep_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - torch missing or broken install
        return False
    return True


def make_deep(kind: str, cfg: LabConfig) -> Strategy:
    from tradinglab.strategies.deep.strategy import DeepStrategy

    return DeepStrategy(kind, cfg.deep, cfg.costs.total_bps, bars_per_year(cfg.market.bar),
                        cfg.portfolio.asset_vol_target, cfg.portfolio.vol_lookback,
                        cfg.universe.min_history_bars, cfg.walkforward.valid_fraction)


def build_candidates(cfg: LabConfig, include_deep: bool = True,
                     deep_kinds: tuple[str, ...] = DEEP_KINDS) -> list[Strategy]:
    out = classic_candidates(cfg.universe.symbols)
    if include_deep and deep_available():
        out += [make_deep(k, cfg) for k in deep_kinds]
    return out


def by_name(cfg: LabConfig, name: str) -> Strategy:
    for s in classic_candidates(cfg.universe.symbols):
        if s.name == name:
            return s
    if name.startswith("dl_"):
        return make_deep(name[3:], cfg)
    raise KeyError(f"unknown strategy {name!r}")
