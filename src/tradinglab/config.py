"""Typed configuration loaded from YAML.

Every tunable of the lab lives here so that research, backtests and the live
engine run on exactly the same assumptions (costs, lag, sizing, risk).
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class PathsConfig:
    data_dir: str = "data/cache"
    state_dir: str = "state"
    models_dir: str = "models"
    reports_dir: str = "reports"

    def resolve(self, root: Path = PROJECT_ROOT) -> "PathsConfig":
        return PathsConfig(**{k: str((root / v).resolve()) if not Path(v).is_absolute() else v
                              for k, v in dataclasses.asdict(self).items()})


@dataclass
class UniverseConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    # An asset only enters the tradable cross-section after this much history.
    min_history_bars: int = 300


@dataclass
class MarketConfig:
    # "perp": linear perpetual-style positions (long/short, funding applies).
    # "spot": long-only, no funding.
    instrument: str = "perp"
    # Bar size used for signals (research + live). "1d" or "1h"/"4h".
    bar: str = "1d"
    # Hours after the bar close at which rebalancing trades execute.
    exec_lag_hours: int = 0
    # Funding assumed when the archive has no rate yet (per 8h, Binance baseline).
    default_funding_8h: float = 0.0001


@dataclass
class CostConfig:
    fee_bps: float = 5.0        # exchange fee per side, bps of traded notional
    slippage_bps: float = 5.0   # spread + impact per side, bps of traded notional

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.slippage_bps


@dataclass
class PortfolioConfig:
    initial_capital: float = 100_000.0
    target_vol: float = 0.25          # annualised ex-ante portfolio volatility target
    asset_vol_target: float = 0.60    # per-asset vol used to normalise signals
    vol_lookback: int = 60            # bars for EWMA vol / covariance
    max_gross: float = 1.5            # max sum |w|
    max_weight: float = 0.35          # max |w| per asset
    rebalance_band: float = 0.005     # skip trades smaller than this fraction of equity
    signal_span: int = 5              # EMA span applied to rule-based signals (turnover control)
    min_trade_notional: float = 10.0  # exchange-style minimum order size (quote ccy)


@dataclass
class RiskConfig:
    dd_derisk_start: float = 0.25   # start scaling exposure down at this drawdown
    dd_halt: float = 0.40           # flatten and halt new risk at this drawdown
    dd_floor_scale: float = 0.5     # exposure multiplier just before the halt
    stale_data_hours: float = 60.0  # refuse to trade on data older than this
    max_price_jump: float = 0.6     # reject bars with |log return| above this (bad ticks)


@dataclass
class WalkForwardConfig:
    data_start: str = "2018-01-01"
    first_test: str = "2021-01-01"
    holdout_start: str = "2025-01-01"  # selection uses OOS data before this; after = untouched holdout
    step_months: int = 6
    valid_fraction: float = 0.15    # tail of each training window used for early stopping


@dataclass
class SelectionConfig:
    min_oos_sharpe: float = 0.5
    min_psr: float = 0.80            # probabilistic Sharpe vs 0, OOS
    min_dsr: float = 0.50            # deflated Sharpe ratio (multiple-testing adjusted)
    cluster_corr: float = 0.7        # strategies correlated above this form one cluster
    max_clusters: int = 4
    weighting: str = "inverse_vol"   # across clusters: inverse_vol | equal | sharpe


@dataclass
class DeepConfig:
    seq_len: int = 63
    burn_in: int = 21               # steps excluded from the loss (warm-up)
    hidden: int = 32
    heads: int = 4
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 10
    seeds: int = 3
    turnover_penalty: float = 1.0   # multiplies estimated cost in the loss
    softmin_lambda: float = 0.1     # DeePM-style robust worst-window penalty weight
    softmin_beta: float = 5.0
    threads: int = 4


@dataclass
class LiveConfig:
    feed: str = "auto"               # auto | realtime | archive
    poll_seconds: int = 300          # daemon sleep between cycles
    retrain_days: int = 7            # retrain deep models / re-run selection cadence
    fill_latency_seconds: int = 60   # order → fill delay used to price fills
    shadow_all: bool = True          # paper-trade every candidate in shadow books


@dataclass
class LabConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: dict[str, Any] | None):
    data = data or {}
    kwargs = {}
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name in known:
        if name not in data:
            continue
        value = data[name]
        ftype = hints[name]
        if dataclasses.is_dataclass(ftype):
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> LabConfig:
    """Load the default config, then layer an optional YAML file and overrides on top."""
    raw: dict[str, Any] = {}
    if DEFAULT_CONFIG.exists():
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        raw = _deep_merge(raw, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg = _build(LabConfig, raw)
    cfg.paths = cfg.paths.resolve()
    return cfg


def bars_per_year(bar: str) -> float:
    """Number of bars per year for a 24/7 market."""
    unit, n = bar[-1], int(bar[:-1])
    hours = {"m": n / 60, "h": n, "d": 24 * n, "w": 24 * 7 * n}[unit]
    return 365.0 * 24.0 / hours


def bar_timedelta(bar: str):
    import pandas as pd

    unit, n = bar[-1], int(bar[:-1])
    return pd.Timedelta(**{{"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[unit]: n})


def utc(ts) -> "pd.Timestamp":
    """Coerce anything timestamp-like to a tz-aware UTC pandas Timestamp."""
    import pandas as pd

    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
